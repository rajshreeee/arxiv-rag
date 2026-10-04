import json
import gc
import torch
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EVAL_FILE = "data/eval_final.json"
ANSWERS_FILE = "data/answers.json"
HIT_FILE = "data/retrieval_hit.json"

TOP_K = 5
MIN_OVERLAP = 100

device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda"

BGE_QUERY = "Represent this sentence for searching relevant passages: "
SETUPS = [
    ("bge-base",  "BAAI/bge-base-en-v1.5",                    BGE_QUERY,  "",          "small"),
    ("e5-base",   "intfloat/e5-base-v2",                       "query: ",  "passage: ", "medium"),
    ("minilm",    "sentence-transformers/all-MiniLM-L6-v2",    "",         "",          "large"),
]

with open(EVAL_FILE) as f:
    questions = json.load(f)

try:
    with open(ANSWERS_FILE) as f:
        answers = json.load(f)
except FileNotFoundError:
    answers = {}

try:
    with open(HIT_FILE) as f:
        retrieval_hit = json.load(f)
except FileNotFoundError:
    retrieval_hit = {}

llm = ChatOllama(model="qwen2.5:7b", temperature=0, num_ctx=8192)

rag_prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context below.\n"
    "If the context does not contain the answer, reply exactly: "
    "\"The context does not contain the answer.\"\n"
    "Answer in one or two sentences.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)
closed_prompt = ChatPromptTemplate.from_template(
    "Answer the question in one or two sentences.\n"
    "If you do not know, reply exactly: \"I do not know.\"\n\n"
    "Question: {question}"
)

rag_chain = rag_prompt | llm | StrOutputParser()
closed_chain = closed_prompt | llm | StrOutputParser()


def save():
    with open(ANSWERS_FILE, "w") as f:
        json.dump(answers, f, ensure_ascii=False, indent=1)
    with open(HIT_FILE, "w") as f:
        json.dump(retrieval_hit, f)


def overlap(start_a, end_a, start_b, end_b):
    return max(0, min(end_a, end_b) - max(start_a, start_b))


for q in questions:
    key = q["qid"] + "|closed_book"
    if key not in answers:
        answers[key] = closed_chain.invoke({"question": q["question"]}).strip()

    key = q["qid"] + "|oracle"
    if key not in answers:
        answers[key] = rag_chain.invoke({"context": q["passage"], "question": q["question"]}).strip()
    save()

print("Closed-book and oracle answers done.")

for model_key, model_name, query_prefix, passage_prefix, size in SETUPS:
    condition = "rag_" + model_key + "_" + size
    print(condition, "...")

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

    with open("data/chunks_" + size + ".json") as f:
        rows = json.load(f)
    chunks = [Document(page_content=r["page_content"], metadata=r["metadata"]) for r in rows]
    vs = FAISS.from_documents(chunks, embeddings)

    hits = 0
    for q in questions:
        key = q["qid"] + "|" + condition
        found = vs.similarity_search(q["question"], k=TOP_K)

        hit = False
        for chunk in found:
            if chunk.metadata["arxiv_id"] == q["arxiv_id"]:
                start = chunk.metadata["start_index"]
                end = start + len(chunk.page_content)
                if overlap(start, end, q["start_index"], q["end_index"]) >= MIN_OVERLAP:
                    hit = True
        retrieval_hit[key] = hit
        if hit:
            hits += 1

        if key not in answers:
            context = "\n\n".join(c.page_content for c in found)
            answers[key] = rag_chain.invoke({"context": context, "question": q["question"]}).strip()
            save()

    print(" ", condition, "hit rate:", round(hits / len(questions), 2))

    del vs, embeddings
    gc.collect()
    torch.cuda.empty_cache()

save()
print("Done. Total answers:", len(answers))
