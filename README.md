# ArXiv RAG

50 papers were downloaded from arxiv. 48 questions were written by hand from passage samples, then answered by qwen2.5:7b under four conditions: no context, the correct passage given directly, naive RAG, and a multi-agent loop where a critic can reject retrieved passages and send the researcher back to search again (max 2 rewrites). A second model (llama3.1:8b) graded each answer as correct or incorrect.

## How to run

Requires Ollama with `qwen2.5:7b` and `llama3.1:8b` pulled. Install Python dependencies:

```
pip install -r requirements.txt
```

## Tools

- Embedding: `intfloat/e5-base-v2` via sentence-transformers
- Vector store: Qdrant 
- Agent framework: LangGraph
- LLM (researcher + critic + judge): Ollama (qwen2.5:7b, llama3.1:8b)

## Results

### Retrieval benchmark (passage-level MRR, 48 questions, FAISS)

| model | chunks | passage@1 | passage@5 | passage@10 | MRR | paper@10 |
|---|---|---|---|---|---|---|
| bge-base | small | 0.71 | 0.85 | 0.85 | 0.763 | 0.92 |
| bge-small | small | 0.65 | 0.83 | 0.85 | 0.742 | 0.96 |
| bge-base | medium | 0.65 | 0.85 | 0.90 | 0.735 | 0.94 |
| e5-base | medium | 0.62 | 0.85 | 0.90 | 0.724 | 0.92 |
| e5-base | small | 0.60 | 0.81 | 0.94 | 0.710 | 0.96 |
| bge-small | medium | 0.60 | 0.83 | 0.85 | 0.696 | 0.94 |
| minilm | small | 0.56 | 0.77 | 0.81 | 0.652 | 0.92 |
| bge-base | large | 0.54 | 0.77 | 0.88 | 0.649 | 0.92 |
| e5-base | large | 0.54 | 0.75 | 0.88 | 0.644 | 0.94 |
| bge-small | large | 0.54 | 0.77 | 0.83 | 0.638 | 0.92 |
| minilm | medium | 0.44 | 0.69 | 0.73 | 0.558 | 0.88 |
| minilm | large | 0.44 | 0.65 | 0.69 | 0.507 | 0.88 |

### Answer quality (48 questions)

| condition | correct | incorrect | abstained | accuracy | accuracy on hard |
|---|---|---|---|---|---|
| Closed-book | 9 | 3 | 36 | 0.19 | 0.00 |
| Oracle | 47 | 0 | 1 | 0.98 | 1.00 |
| Naive RAG (Qdrant, e5-base, medium) | 39 | 3 | 6 | 0.81 | 0.82 |
| Multi-agent (researcher + critic) | 39 | 3 | 6 | 0.81 | 0.82 |

The critic loop fixed 1 question and broke 1 (sign test p = 1.0), so it made no measurable difference over naive RAG on this dataset.
