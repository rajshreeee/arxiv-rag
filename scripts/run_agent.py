import os
import json
import time
from typing import List, TypedDict

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

CHUNKS_FILE = "data/chunks_medium.json"
EVAL_FILE = "data/eval_final.json"
JUDGMENTS_FILE = "data/judgments.json"
RESULTS_FILE = "data/results_agent.json"
LABELS_FILE = "data/labels.json"

EMBED_MODEL = "intfloat/e5-base-v2"
LLM_MODEL = "qwen2.5:7b"
JUDGE_MODEL = "llama3.1:8b"
DEVICE = "cuda"
TOP_K = 5
MIN_OVERLAP = 100
MAX_REWRITES = 2
NO_ANSWER = "The context does not contain the answer."

with open(EVAL_FILE) as f:
    questions = json.load(f)
with open(CHUNKS_FILE) as f:
    chunks = json.load(f)

print("questions:", len(questions), "| chunks:", len(chunks))

embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)

texts = ["passage: " + c["page_content"] for c in chunks]
vectors = embedder.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

qdrant = QdrantClient(":memory:")
qdrant.create_collection("papers", vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE))

points = []
for i, c in enumerate(chunks):
    info = {
        "text": c["page_content"],
        "arxiv_id": c["metadata"]["arxiv_id"],
        "title": c["metadata"]["title"],
        "start_index": c["metadata"]["start_index"],
    }
    points.append(PointStruct(id=i, vector=vectors[i].tolist(), payload=info))

for i in range(0, len(points), 500):
    qdrant.upsert("papers", points=points[i:i + 500])
print("Stored", len(points), "chunks in Qdrant")


def search(query):
    vector = embedder.encode("query: " + query, normalize_embeddings=True)
    result = qdrant.query_points("papers", query=vector.tolist(), limit=TOP_K, with_payload=True)
    return [p.payload for p in result.points]


def source_hit(docs, q):
    for d in docs:
        if d["arxiv_id"] != q["arxiv_id"]:
            continue
        start = d["start_index"]
        end = start + len(d["text"])
        shared = min(end, q["end_index"]) - max(start, q["start_index"])
        if shared >= MIN_OVERLAP:
            return True
    return False


llm = ChatOllama(model=LLM_MODEL, temperature=0, num_ctx=8192)


def ask(prompt):
    return llm.invoke(prompt).content.strip()


def draft_answer(question, docs):
    context = "\n\n".join(d["text"] for d in docs)
    prompt = (
        f"Answer the question using only the context below.\n"
        f"If the context does not contain the answer, reply exactly: \"{NO_ANSWER}\"\n"
        f"Answer in one or two sentences.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    return ask(prompt)


def critic_check(question, docs):
    numbered = "\n\n".join(f"[{i + 1}] {d['text']}" for i, d in enumerate(docs))
    prompt = (
        f"You are a strict peer reviewer. You do not answer questions.\n"
        f"You only judge whether the retrieved passages contain the information needed to answer the question.\n"
        f"Passages that are only loosely on-topic, are table or figure debris, or describe a different method do NOT count.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved passages:\n{numbered}\n\n"
        f"Reply in exactly this format:\n"
        f"VERDICT: PASS or REJECT\n"
        f"FEEDBACK: if REJECT, one sentence on what is missing and which keywords to search for"
    )
    reply = ask(prompt)
    verdict = "REJECT" if "REJECT" in reply.split("\n")[0].upper() else "PASS"
    feedback = reply.split("FEEDBACK:")[1].strip() if "FEEDBACK:" in reply else ""
    return verdict, feedback


def rewrite_query(question, tried, feedback):
    prompt = (
        f"You are a research assistant searching a database of arXiv papers about retrieval-augmented generation.\n"
        f"Your last search did not find the text needed to answer the question.\n\n"
        f"Question: {question}\n"
        f"Queries already tried: {tried}\n"
        f"Reviewer feedback: {feedback}\n\n"
        f"Write ONE new search query likely to match the wording of the relevant paper section.\n"
        f"Use specific technical terms and method names from the question. Do not repeat a query already tried.\n"
        f"Reply with the query only, on one line."
    )
    new_query = ask(prompt).split("\n")[0].strip().strip('"')
    return new_query if new_query else question


class State(TypedDict):
    question: str
    query: str
    queries: List[str]
    docs: List[dict]
    first_docs: List[dict]
    verdict: str
    feedback: str
    rejections: List[str]
    rewrites: int
    answer: str


def researcher_search(state):
    docs = search(state["query"])
    first = state["first_docs"] if state["first_docs"] else docs
    return {"docs": docs, "first_docs": first}


def critic(state):
    verdict, feedback = critic_check(state["question"], state["docs"])
    return {"verdict": verdict, "feedback": feedback}


def researcher_rewrite(state):
    tried = " | ".join(state["queries"])
    new_query = rewrite_query(state["question"], tried, state["feedback"])
    return {
        "query": new_query,
        "queries": state["queries"] + [new_query],
        "rejections": state["rejections"] + [state["feedback"]],
        "rewrites": state["rewrites"] + 1,
    }


def researcher_draft(state):
    return {"answer": draft_answer(state["question"], state["docs"])}


def after_critic(state):
    if state["verdict"] == "REJECT" and state["rewrites"] < MAX_REWRITES:
        return "rewrite"
    return "draft"


graph = StateGraph(State)
graph.add_node("researcher_search", researcher_search)
graph.add_node("critic", critic)
graph.add_node("researcher_rewrite", researcher_rewrite)
graph.add_node("researcher_draft", researcher_draft)
graph.set_entry_point("researcher_search")
graph.add_edge("researcher_search", "critic")
graph.add_conditional_edges("critic", after_critic, {"rewrite": "researcher_rewrite", "draft": "researcher_draft"})
graph.add_edge("researcher_rewrite", "researcher_search")
graph.add_edge("researcher_draft", END)
app = graph.compile()


def run_multi_agent(question):
    start = {
        "question": question, "query": question, "queries": [question],
        "docs": [], "first_docs": [], "verdict": "", "feedback": "",
        "rejections": [], "rewrites": 0, "answer": "",
    }
    return app.invoke(start)


try:
    with open(RESULTS_FILE) as f:
        results = json.load(f)
except FileNotFoundError:
    results = {}

start_time = time.time()
for i, q in enumerate(questions):
    qid = q["qid"]

    if qid + "|naive" not in results:
        docs = search(q["question"])
        results[qid + "|naive"] = {
            "answer": draft_answer(q["question"], docs),
            "hit": source_hit(docs, q),
        }

    if qid + "|multi_agent" not in results:
        final = run_multi_agent(q["question"])
        results[qid + "|multi_agent"] = {
            "answer": final["answer"],
            "queries": final["queries"],
            "rejections": final["rejections"],
            "rewrites": final["rewrites"],
            "first_hit": source_hit(final["first_docs"], q),
            "hit": source_hit(final["docs"], q),
        }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=1)
    if (i + 1) % 8 == 0:
        print(i + 1, "done,", round(time.time() - start_time), "s")

print("Answers done.")

judge = ChatOllama(model=JUDGE_MODEL, temperature=0)

with open(JUDGMENTS_FILE) as f:
    old_judgments = json.load(f)

labels = {key: val["label"] for key, val in old_judgments.items()}


def is_abstention(text):
    t = text.lower()
    return "does not contain the answer" in t or "i do not know" in t


for q in questions:
    for system in ["naive", "multi_agent"]:
        key = q["qid"] + "|" + system
        if key in labels:
            continue
        answer = results[key]["answer"]
        if is_abstention(answer):
            labels[key] = "abstained"
            continue
        prompt = (
            f"You are grading an answer to a question.\n"
            f"Compare the candidate answer with the reference answer.\n\n"
            f"Reply CORRECT if the candidate gives the same key facts as the reference. "
            f"Extra detail is fine as long as it does not contradict the reference.\n"
            f"Reply INCORRECT if the candidate misses the key fact, contradicts the reference, "
            f"or talks about something else.\n\n"
            f"Reply with one word only: CORRECT or INCORRECT.\n\n"
            f"Question: {q['question']}\n"
            f"Reference answer: {q['reference_answer']}\n"
            f"Candidate answer: {answer}"
        )
        reply = judge.invoke(prompt).content.upper()
        if "INCORRECT" in reply:
            labels[key] = "incorrect"
        elif "CORRECT" in reply:
            labels[key] = "correct"
        else:
            labels[key] = "unclear"

with open(LABELS_FILE, "w") as f:
    json.dump(labels, f, indent=1)
print("Labels saved:", LABELS_FILE)
