"""Retrieval-augmented critique of a paper PDF.

Replaces main.py + prompts.py. No LangChain, no FAISS: for a 50-chunk paper the
retrieval is a dot product against a 50x384 matrix, which numpy does instantly,
and the framework was hiding the bug that made this thing useless.

The bug: RetrievalQA(chain_type="stuff") pasted 4 x 2000-character chunks into
google/flan-t5-base, whose encoder accepts 512 tokens. Roughly 75% of the
context was silently dropped before the model saw it, so every answer was
generated from a fragment. Nothing errored. Here the context is measured
against the tokenizer and fitted to the real budget, and the number of chunks
that actually made it in is printed with every answer.
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np
import torch
from pypdf import PdfReader
from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-base")
# T5 uses relative position bias, so it accepts sequences longer than the
# tokeniser's legacy 512 default. 1024 roughly triples how much context fits.
MAX_INPUT = int(os.getenv("MAX_INPUT", "1024"))
CHUNK_CHARS = 1200
OVERLAP = 200
TOP_K = 8          # candidates; how many actually fit is decided by the budget
ANSWER_TOKENS = 128

# "in a complete sentence" is load-bearing on an instruction-tuned model this
# small. Without it, "Which ImageNet top-5 error does SE-ResNet-50 achieve?"
# returns the noun phrase "single-crop"; with it, "6.62%".
PROMPT = (
    "Read the excerpts from a research paper and answer the question in a "
    "complete sentence. Use only the excerpts. If they do not contain the "
    "answer, say so.\n\nExcerpts:\n{context}\n\nQuestion: {question}\nAnswer:"
)

QUESTIONS = [
    "What is the main novelty or contribution of the paper?",
    "What are the key limitations the authors acknowledge?",
    "What future work or improvements do the authors suggest?",
    "What datasets and evaluation metrics are used?",
]


# --------------------------------------------------------------------- chunking

def load_chunks(pdf_path: str) -> list[str]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)
    pages = [(p.extract_text() or "") for p in PdfReader(pdf_path).pages]
    text = re.sub(r"[ \t]+", " ", "\n".join(pages))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Drop the bibliography. Reference entries are dense with paper titles, so
    # they out-score real prose on any query about contributions or future work.
    # Before this, "what is the main contribution" retrieved the reference list.
    half = len(text) // 2
    tail = re.search(r"\n\s*(REFERENCES|References|Bibliography)\s*\n", text[half:])
    if tail:
        text = text[: half + tail.start()]

    chunks, i = [], 0
    while i < len(text):
        chunk = text[i:i + CHUNK_CHARS].strip()
        if len(chunk) > 100:                       # drop headers/page-number scraps
            chunks.append(chunk)
        i += CHUNK_CHARS - OVERLAP
    if not chunks:
        raise ValueError(f"No extractable text in {pdf_path}. Is it a scanned PDF?")
    return chunks


# -------------------------------------------------------------------- embedding

class Embedder:
    """MiniLM sentence embeddings: mean-pool over non-padding tokens, L2 normalise.

    ponytail: this is what sentence-transformers does for this model. Eight lines
    against a dependency. Swap it back in if a model needs a different pooling head.
    """

    def __init__(self, name: str = EMBED_MODEL):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).eval()

    @torch.no_grad()
    def __call__(self, texts: list[str], batch: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok(texts[i:i + batch], padding=True, truncation=True,
                           max_length=256, return_tensors="pt")
            hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(vec, dim=1).numpy())
        return np.vstack(out)


def retrieve(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> list[int]:
    """Indices of the k nearest chunks. Vectors are unit-norm, so dot == cosine."""
    return np.argsort(-(matrix @ query_vec))[:k].tolist()


# ------------------------------------------------------------------ the context

def fit_context(chunks: list[str], question: str, tok, max_input: int) -> tuple[str, int]:
    """Pack chunks into the prompt until the tokeniser says we are at the limit.

    Returns the context and how many chunks fitted. This function is the fix:
    the previous version assumed the context fitted and it did not.
    """
    scaffold = len(tok(PROMPT.format(context="", question=question)).input_ids)
    budget = max_input - scaffold - 8               # 8 tokens of slack
    if budget <= 0:
        raise ValueError(f"Question alone exceeds the {max_input}-token input limit.")

    used, kept = 0, []
    for c in chunks:
        n = len(tok(c).input_ids)
        if used + n > budget:
            break
        kept.append(c)
        used += n
    if not kept:                                     # first chunk alone is too big
        ids = tok(chunks[0]).input_ids[:budget]
        kept = [tok.decode(ids, skip_special_tokens=True)]
    return "\n\n".join(kept), len(kept)


# ---------------------------------------------------------------------- pipeline

def critique(pdf_path: str, questions: list[str] | None = None,
             interactive: bool = False) -> dict[str, str]:
    """Answer `questions` about the paper, then optionally take more from stdin.

    Chunking, embedding and model load happen once. Each further question is
    one MiniLM forward pass plus one generate, so the interactive loop is
    effectively free once the fixed critique has printed.
    """
    questions = questions or QUESTIONS

    chunks = load_chunks(pdf_path)
    print(f"{len(chunks)} chunks from {os.path.basename(pdf_path)}")

    embed = Embedder()
    matrix = embed(chunks)

    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    llm = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL).eval()
    max_input = MAX_INPUT
    print(f"{LLM_MODEL}: {max_input}-token input budget")

    def ask(question: str) -> str:
        qvec = embed([question])[0]
        top = [chunks[i] for i in retrieve(qvec, matrix, TOP_K)]
        context, n_fit = fit_context(top, question, tok, max_input)
        enc = tok(PROMPT.format(context=context, question=question),
                  return_tensors="pt", truncation=True, max_length=max_input)
        with torch.no_grad():
            # no_repeat_ngram_size is not optional here: beam search on a 250M
            # model given table-like context degenerates into repeating one
            # phrase for the whole answer budget.
            ids = llm.generate(**enc, max_new_tokens=ANSWER_TOKENS, num_beams=4,
                               no_repeat_ngram_size=3, repetition_penalty=1.2,
                               early_stopping=True)
        out = tok.decode(ids[0], skip_special_tokens=True).strip()
        print(f"\n- {question}\n  [{n_fit}/{TOP_K} retrieved chunks fitted the budget]"
              f"\n  > {out}")
        return out

    answers = {q: ask(q) for q in questions}

    # ponytail: each question is answered independently against the paper.
    # No conversation history: the context budget is already full of excerpts,
    # and prior turns would displace the evidence the answer is grounded in.
    while interactive:
        try:
            q = input("\nask (blank to quit) > ").strip()
        except EOFError:
            break
        if not q:
            break
        answers[q] = ask(q)

    return answers


def demo() -> None:
    """Self-check. Runs the parts that can break without downloading the LLM."""
    here = os.path.dirname(os.path.abspath(__file__))
    pdf = os.path.join(here, "data", "SE Net paper.pdf")

    chunks = load_chunks(pdf)
    assert len(chunks) > 5, f"only {len(chunks)} chunks"
    assert all(len(c) <= CHUNK_CHARS for c in chunks)

    embed = Embedder()
    matrix = embed(chunks)
    assert matrix.shape[0] == len(chunks)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5), "not unit-norm"

    # retrieval must rank a chunk's own text first when used as the query
    probe = 3
    assert retrieve(matrix[probe], matrix, 1)[0] == probe, "self-retrieval failed"

    # the regression guard: the packed prompt must never exceed the model limit
    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    limit = MAX_INPUT
    for q in QUESTIONS:
        top = [chunks[i] for i in retrieve(embed([q])[0], matrix, TOP_K)]
        ctx, n = fit_context(top, q, tok, limit)
        total = len(tok(PROMPT.format(context=ctx, question=q)).input_ids)
        assert total <= limit, f"prompt {total} > limit {limit}"
        assert n >= 1
    print(f"demo OK: {len(chunks)} chunks, prompts fit within {limit} tokens")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    if "--demo" in sys.argv:
        demo()
        sys.exit()

    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    pdf = args[0] if args and args[0].lower().endswith(".pdf") else \
        os.path.join(here, "data", "SE Net paper.pdf")
    # Anything else on the command line is treated as a question, replacing
    # the standard critique set.
    asked = [a for a in args if a is not pdf and not a.lower().endswith(".pdf")]
    critique(pdf, questions=asked or None, interactive="--chat" in sys.argv)
