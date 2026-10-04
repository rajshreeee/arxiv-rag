import json
import random
import time
from langchain_ollama import ChatOllama

CHUNKS_FILE = "data/chunks_medium.json"
CANDIDATES_FILE = "data/eval_candidates.json"
OUTPUT = "data/eval_drafts.json"

MODEL = "qwen2.5:7b"
MAX_CANDIDATES = 70

PROMPT = """You are helping build a test set for a search system over research papers about retrieval-augmented generation (RAG).

Below is a passage from one paper. Write ONE question that this passage answers, and a short answer based only on the passage.

Rules for the question:
- It must make sense on its own, for someone who has not seen the passage. Never write "the passage", "the text", "the paper" or "the authors".
- Do not copy phrases from the passage. Use your own words.
- Ask about a specific detail, method or finding. Do not ask something generic that any RAG paper could answer.

The answer should be 1 or 2 sentences.

Reply in exactly this format:
QUESTION: <the question>
ANSWER: <the answer>

Passage:
{passage}
"""

BAD_PHRASES = [
    "the authors", "the author", "the passage", "the text",
    "the paper", "this paper", "the study", "this study",
    "their work", "the article", "the researchers",
]


def is_good_passage(text, start_index):
    if start_index < 3000:
        return False
    letters = sum(1 for c in text if c.isalpha())
    digits = sum(1 for c in text if c.isdigit())
    non_space = sum(1 for c in text if not c.isspace())
    if non_space == 0:
        return False
    if letters / non_space < 0.85:
        return False
    if digits > 40:
        return False
    if text.count("et al") > 2:
        return False
    if len(text.split()) < 80:
        return False
    return True


def copy_rate(question, passage):
    def content_words(text):
        words = []
        for w in text.lower().split():
            w = w.strip(".,;:?!()'\"")
            if len(w) > 3:
                words.append(w)
        return words
    q_words = content_words(question)
    p_words = set(content_words(passage))
    if not q_words:
        return 0
    return sum(1 for w in q_words if w in p_words) / len(q_words)


with open(CHUNKS_FILE) as f:
    rows = json.load(f)

by_paper = {}
for row in rows:
    meta = row["metadata"]
    if is_good_passage(row["page_content"], meta["start_index"]):
        pid = meta["arxiv_id"]
        if pid not in by_paper:
            by_paper[pid] = []
        by_paper[pid].append(row)

random.seed(42)
candidates = []
for pid, paper_rows in by_paper.items():
    chosen = random.sample(paper_rows, min(2, len(paper_rows)))
    for row in chosen:
        meta = row["metadata"]
        candidates.append({
            "arxiv_id": pid,
            "title": meta["title"],
            "start_index": meta["start_index"],
            "end_index": meta["start_index"] + len(row["page_content"]),
            "passage": row["page_content"],
        })

random.shuffle(candidates)
candidates = candidates[:MAX_CANDIDATES]
for i, c in enumerate(candidates):
    c["qid"] = "q" + str(i).zfill(2)

with open(CANDIDATES_FILE, "w") as f:
    json.dump(candidates, f, ensure_ascii=False, indent=1)
print("Candidates saved:", len(candidates))

llm = ChatOllama(model=MODEL, temperature=0)

drafts = []
start = time.time()
for i, c in enumerate(candidates):
    reply = llm.invoke(PROMPT.format(passage=c["passage"])).content
    if "QUESTION:" not in reply or "ANSWER:" not in reply:
        print("parse failed:", c["qid"])
        continue
    question = reply.split("QUESTION:")[1].split("ANSWER:")[0].strip()
    answer = reply.split("ANSWER:")[1].strip()
    drafts.append({
        "qid": c["qid"],
        "arxiv_id": c["arxiv_id"],
        "title": c["title"],
        "start_index": c["start_index"],
        "end_index": c["end_index"],
        "passage": c["passage"],
        "question": question,
        "reference_answer": answer,
        "copy_rate": round(copy_rate(question, c["passage"]), 2),
    })
    if (i + 1) % 10 == 0:
        print(i + 1, "done,", round(time.time() - start), "s")

auto_rejected = {}
for d in drafts:
    q = d["question"].lower()
    for phrase in BAD_PHRASES:
        if phrase in q:
            auto_rejected[d["qid"]] = phrase
            break
    if d["qid"] not in auto_rejected and len(q.split()) < 6:
        auto_rejected[d["qid"]] = "too short"

print("Auto-rejected:", len(auto_rejected))
remaining = [d for d in drafts if d["qid"] not in auto_rejected]
print("Needs human review:", len(remaining))

with open(OUTPUT, "w") as f:
    json.dump(remaining, f, ensure_ascii=False, indent=1)
