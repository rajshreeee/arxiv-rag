import json
import gc
import time
import torch
import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EVAL_FILE = "data/eval_final.json"
OUTPUT_CSV = "data/benchmark_table.csv"
OUTPUT_JSON = "data/benchmark_results.json"

K = 10
MIN_OVERLAP = 100

device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda"

BGE_QUERY = "Represent this sentence for searching relevant passages: "
MODELS = {
    "minilm":    ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    "bge-small": ("BAAI/bge-small-en-v1.5", BGE_QUERY, ""),
    "bge-base":  ("BAAI/bge-base-en-v1.5",  BGE_QUERY, ""),
    "e5-base":   ("intfloat/e5-base-v2", "query: ", "passage: "),
}

with open(EVAL_FILE) as f:
    questions = json.load(f)


def load_chunks(size):
    with open("data/chunks_" + size + ".json") as f:
        rows = json.load(f)
    return [Document(page_content=r["page_content"], metadata=r["metadata"]) for r in rows]


def overlap(start_a, end_a, start_b, end_b):
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def score_questions(vectorstore, questions):
    paper_ranks = []
    passage_ranks = []
    for q in questions:
        found = vectorstore.similarity_search(q["question"], k=K)
        paper_rank = None
        passage_rank = None
        for pos, chunk in enumerate(found, 1):
            if chunk.metadata["arxiv_id"] != q["arxiv_id"]:
                continue
            if paper_rank is None:
                paper_rank = pos
            if passage_rank is None:
                start = chunk.metadata["start_index"]
                end = start + len(chunk.page_content)
                if overlap(start, end, q["start_index"], q["end_index"]) >= MIN_OVERLAP:
                    passage_rank = pos
        paper_ranks.append(paper_rank)
        passage_ranks.append(passage_rank)
    return paper_ranks, passage_ranks


def hit_at(ranks, k):
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks):
    return sum(1 / r for r in ranks if r is not None) / len(ranks)


rows = []
all_ranks = {}

for model_key, (model_name, query_prefix, passage_prefix) in MODELS.items():
    for size in ["small", "medium", "large"]:
        print(model_key, size, "...")

        passage_kwargs = {"normalize_embeddings": True, "batch_size": 64}
        query_kwargs = {"normalize_embeddings": True}
        if passage_prefix:
            passage_kwargs["prompt"] = passage_prefix
        if query_prefix:
            query_kwargs["prompt"] = query_prefix

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs=passage_kwargs,
            query_encode_kwargs=query_kwargs,
        )

        chunks = load_chunks(size)
        t0 = time.time()
        vs = FAISS.from_documents(chunks, embeddings)
        index_sec = round(time.time() - t0)

        paper_ranks, passage_ranks = score_questions(vs, questions)

        rows.append({
            "model": model_key,
            "chunks": size,
            "n_chunks": len(chunks),
            "passage@1":  round(hit_at(passage_ranks, 1), 2),
            "passage@5":  round(hit_at(passage_ranks, 5), 2),
            "passage@10": round(hit_at(passage_ranks, 10), 2),
            "MRR":        round(mrr(passage_ranks), 3),
            "paper@10":   round(hit_at(paper_ranks, 10), 2),
            "index_sec":  index_sec,
        })
        all_ranks[model_key + "_" + size] = {
            "paper_ranks": paper_ranks,
            "passage_ranks": passage_ranks,
        }
        print("  MRR:", rows[-1]["MRR"])

        del vs, embeddings
        gc.collect()
        torch.cuda.empty_cache()

df = pd.DataFrame(rows).sort_values("MRR", ascending=False)
df.to_csv(OUTPUT_CSV, index=False)
print(df[["model", "chunks", "passage@1", "passage@5", "MRR"]].to_string(index=False))

with open(OUTPUT_JSON, "w") as f:
    json.dump(all_ranks, f)

print("Saved:", OUTPUT_CSV, OUTPUT_JSON)
