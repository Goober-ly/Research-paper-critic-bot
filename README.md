# Research Paper Critic

Ask fixed critique questions of a paper PDF and get answers grounded in the
paper's own text: retrieve the most relevant passages, fit them to the model's
real token budget, generate.

```bash
pip install -r requirements.txt

python critic.py data/paper.pdf                 # the four standard questions
python critic.py data/paper.pdf "your question" # ask your own instead
python critic.py --chat data/paper.pdf          # keep asking, interactively
python critic.py --demo                         # self-check
```

`--chat` prints the standard critique, then drops to a prompt. The paper is
chunked, embedded and the model loaded once, so every follow-up costs one
MiniLM pass and one generate:

```
ask (blank to quit) > What is the reduction ratio r and what value do they use?
  [2/8 retrieved chunks fitted the budget]
  > 16

ask (blank to quit) > Which ImageNet top-5 error does SE-ResNet-50 achieve?
  [3/8 retrieved chunks fitted the budget]
  > 6.62%
```

Each question is answered independently against the paper. There is no
conversation history, deliberately: the context budget is already full of
excerpts, and prior turns would displace the evidence the answer is grounded
in. It interrogates a document; it is not a chatbot.

## Output

```
56 chunks from SE Net paper.pdf
google/flan-t5-base: 1024-token input budget

- What is the main novelty or contribution of the paper?
  [3/8 retrieved chunks fitted the budget]
  > SE block comprises a lightweight gating mechanism which focuses on
    enhancing the representational power of the network

- What datasets and evaluation metrics are used?
  [3/8 retrieved chunks fitted the budget]
  > ImageNet 2012 and 50K validation images from 1000 different classes
```

## Why it was rewritten

The first version used LangChain's `RetrievalQA(chain_type="stuff")` over a
FAISS store. It ran without error and returned nonsense. Three causes:

1. **Silent context truncation.** The retriever returned four 2000-character
   chunks and pasted them into `google/flan-t5-base`, whose encoder takes 512
   tokens. About three quarters of the context was dropped before the model saw
   it. Nothing warned. Every answer was generated from a fragment.
2. **The prompt was dead code.** `prompts.py` defined a careful critique prompt
   that was never passed to the chain, so the default was used.
3. **Retrieval returned the bibliography.** Reference entries are dense with
   paper titles, so they out-scored real prose on "what is the contribution?".

Fixes, in order of how much they mattered: strip the bibliography before
chunking; measure the packed prompt against the tokenizer and report how many
chunks actually fitted; raise the budget to 1024 (T5 uses relative position
bias, so the tokenizer's 512 default is not a hard limit); add
`no_repeat_ngram_size` — beam search on a 250M model over table-like context
otherwise repeats one phrase for the whole answer budget.

LangChain and FAISS went with it. For a 56-chunk paper, retrieval is a dot
product against a 56x384 matrix, and the framework's only real contribution
here was hiding the truncation. Dependencies went from 12 to 4. The old
`requirements.txt` also listed `langchain.llms.huggingface_pipeline`, a module
path rather than a package, so `pip install -r` failed outright.

## Limits

`flan-t5-base` is a 250M-parameter model and it extracts rather than
synthesises: answers are close paraphrases of the retrieved text, and long ones
can stop mid-sentence. Definitional questions are where it is weakest -- "what
is the squeeze operation?" returns "Squeeze-and-Excitation" no matter how the
context is packed. Factual lookups are where it is strongest. That is the
model's ceiling, not the pipeline's. Point
`LLM_MODEL` at something larger to see how far the retrieval actually goes:

```bash
LLM_MODEL=google/flan-t5-large MAX_INPUT=2048 python critic.py data/paper.pdf
```

`EMBED_MODEL`, `MAX_INPUT`, `TOP_K` are environment variables too. The
`--demo` self-check asserts the embeddings are unit-norm, that a chunk
retrieves itself first, and that no packed prompt exceeds the token limit —
that last one is the regression guard for the bug above.
