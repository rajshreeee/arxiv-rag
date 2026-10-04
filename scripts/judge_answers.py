import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

EVAL_FILE = "data/eval_final.json"
ANSWERS_FILE = "data/answers.json"
OUTPUT = "data/judgments.json"

MODEL = "llama3.1:8b"
CONDITIONS = [
    "closed_book",
    "oracle",
    "rag_bge-base_small",
    "rag_e5-base_medium",
    "rag_minilm_large",
]

with open(EVAL_FILE) as f:
    questions = json.load(f)
with open(ANSWERS_FILE) as f:
    answers = json.load(f)

try:
    with open(OUTPUT) as f:
        judgments = json.load(f)
except FileNotFoundError:
    judgments = {}

judge_prompt = ChatPromptTemplate.from_template(
    "You are grading an answer to a question.\n"
    "Compare the candidate answer with the reference answer.\n\n"
    "Reply CORRECT if the candidate gives the same key facts as the reference. "
    "Extra detail is fine as long as it does not contradict the reference.\n"
    "Reply INCORRECT if the candidate misses the key fact, contradicts the reference, "
    "or talks about something else.\n\n"
    "Reply with one word only: CORRECT or INCORRECT.\n\n"
    "Question: {question}\n"
    "Reference answer: {reference}\n"
    "Candidate answer: {candidate}"
)

llm = ChatOllama(model=MODEL, temperature=0)
judge_chain = judge_prompt | llm | StrOutputParser()


def is_abstention(text):
    t = text.lower()
    return "does not contain the answer" in t or "i do not know" in t


def parse_label(reply):
    r = reply.upper()
    if "INCORRECT" in r:
        return "incorrect"
    if "CORRECT" in r:
        return "correct"
    return "unclear"


done = 0
for q in questions:
    for condition in CONDITIONS:
        key = q["qid"] + "|" + condition
        if key in judgments:
            continue
        candidate = answers[key]
        if is_abstention(candidate):
            judgments[key] = {"label": "abstained", "judge_reply": ""}
        else:
            reply = judge_chain.invoke({
                "question": q["question"],
                "reference": q["reference_answer"],
                "candidate": candidate,
            }).strip()
            judgments[key] = {"label": parse_label(reply), "judge_reply": reply}
        done += 1
        if done % 25 == 0:
            with open(OUTPUT, "w") as f:
                json.dump(judgments, f, ensure_ascii=False, indent=1)
            print(done, "judged")

with open(OUTPUT, "w") as f:
    json.dump(judgments, f, ensure_ascii=False, indent=1)

for condition in CONDITIONS:
    counts = {"correct": 0, "incorrect": 0, "abstained": 0, "unclear": 0}
    for q in questions:
        counts[judgments[q["qid"] + "|" + condition]["label"]] += 1
    print(condition.ljust(22), counts)
