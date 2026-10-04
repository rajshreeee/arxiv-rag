import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest

EVAL_FILE = "data/eval_final.json"
LABELS_FILE = "data/labels.json"
RESULTS_AGENT_FILE = "data/results_agent.json"
BENCHMARK_CSV = "data/benchmark_table.csv"

os.makedirs("results", exist_ok=True)

CONDITIONS = [
    "closed_book",
    "oracle",
    "naive",
    "multi_agent",
]
NICE = {
    "closed_book":  "Closed-book",
    "oracle":       "Oracle",
    "naive":        "Naive RAG (Qdrant, e5-base, medium)",
    "multi_agent":  "Multi-agent (researcher + critic)",
}

with open(EVAL_FILE) as f:
    questions = json.load(f)
with open(LABELS_FILE) as f:
    labels = json.load(f)
with open(RESULTS_AGENT_FILE) as f:
    results_agent = json.load(f)
bench = pd.read_csv(BENCHMARK_CSV)

hard_ids = [q["qid"] for q in questions
            if labels[q["qid"] + "|closed_book"] != "correct"]

rows = []
for condition in CONDITIONS:
    correct = incorrect = abstained = hard_correct = 0
    for q in questions:
        label = labels[q["qid"] + "|" + condition]
        if label == "correct":
            correct += 1
            if q["qid"] in hard_ids:
                hard_correct += 1
        elif label == "incorrect":
            incorrect += 1
        elif label == "abstained":
            abstained += 1
    answered = correct + incorrect
    rows.append({
        "condition": NICE[condition],
        "correct": correct,
        "incorrect": incorrect,
        "abstained": abstained,
        "accuracy": round(correct / len(questions), 2),
        "acc. when answered": round(correct / answered, 2) if answered else 0,
        "acc. hard subset": round(hard_correct / len(hard_ids), 2),
    })
table2 = pd.DataFrame(rows)

rag_conditions = ["naive", "multi_agent"]
rows3 = []
for condition in rag_conditions:
    found_correct = found_wrong = missed_correct = missed_abstained = missed_wrong = 0
    for q in questions:
        key = q["qid"] + "|" + condition
        label = labels[key]
        hit = results_agent[key]["hit"]
        if hit:
            if label == "correct":
                found_correct += 1
            else:
                found_wrong += 1
        else:
            if label == "correct":
                missed_correct += 1
            elif label == "abstained":
                missed_abstained += 1
            else:
                missed_wrong += 1
    rows3.append({
        "setup": NICE[condition],
        "found+correct": found_correct,
        "found,not correct": found_wrong,
        "missed+correct": missed_correct,
        "missed+abstained": missed_abstained,
        "missed+wrong": missed_wrong,
    })
table3 = pd.DataFrame(rows3)

table1 = bench.sort_values("MRR", ascending=False)[
    ["model", "chunks", "passage@1", "passage@5", "passage@10", "MRR", "paper@10"]
]

fixed = [q["qid"] for q in questions
         if labels[q["qid"] + "|multi_agent"] == "correct"
         and labels[q["qid"] + "|naive"] != "correct"]
broken = [q["qid"] for q in questions
          if labels[q["qid"] + "|naive"] == "correct"
          and labels[q["qid"] + "|multi_agent"] != "correct"]
changed = len(fixed) + len(broken)
p_value = round(binomtest(len(fixed), changed, 0.5).pvalue, 3) if changed > 0 else None

print("TABLE 1: retrieval benchmark")
print(table1.to_string(index=False))
print("\nTABLE 2: answer quality")
print(table2.to_string(index=False))
print("\nTABLE 3: retrieval vs answer")
print(table3.to_string(index=False))
print("\nNaive vs multi-agent:")
print("  fixed by critic:", len(fixed), fixed)
print("  broken by critic:", len(broken), broken)
if p_value is not None:
    print("  sign test p-value:", p_value)

pivot = bench.pivot(index="model", columns="chunks", values="MRR")[["small", "medium", "large"]]
pivot = pivot.loc[["minilm", "bge-small", "bge-base", "e5-base"]]
pivot.plot(kind="bar", figsize=(8, 5))
plt.ylabel("MRR")
plt.title("Retrieval quality by model and chunk size")
plt.xticks(rotation=0)
plt.legend(title="chunk size")
plt.savefig("results/fig1_retrieval_mrr.png", dpi=150, bbox_inches="tight")
plt.close()

outcomes = table2.set_index("condition")[["correct", "incorrect", "abstained"]].iloc[::-1]
outcomes.plot(kind="barh", stacked=True, figsize=(9, 5),
              color=["#4c9f70", "#d65f5f", "#b0b0b0"])
plt.xlabel("questions (out of 48)")
plt.title("Answer outcomes by condition")
plt.savefig("results/fig2_answer_outcomes.png", dpi=150, bbox_inches="tight")
plt.close()


def to_md(df):
    lines = ["| " + " | ".join(df.columns) + " |",
             "|" + "---|" * len(df.columns)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


with open("results/results.md", "w") as f:
    f.write("## Retrieval benchmark\n\n" + to_md(table1) + "\n\n")
    f.write("## Answer quality\n\n" + to_md(table2) + "\n\n")
    f.write("## Retrieval vs answer\n\n" + to_md(table3) + "\n")

