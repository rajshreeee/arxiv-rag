import arxiv
import pymupdf
import requests
import re
import os
import json
import time
import unicodedata
from collections import Counter

PDF_DIR = "data/pdfs"
OUTPUT = "data/papers.json"
MAX_PAPERS = 50

os.makedirs(PDF_DIR, exist_ok=True)

client = arxiv.Client(page_size=80, delay_seconds=3, num_retries=3)
search = arxiv.Search(
    query='cat:cs.CL AND abs:"retrieval augmented generation"',
    max_results=80,
    sort_by=arxiv.SortCriterion.Relevance,
)
results = [p for p in client.results(search)
           if not re.search(r"survey|review", p.title, re.I)]
print("Candidates:", len(results))


def extract_blocks(path):
    doc = pymupdf.open(path)
    blocks = []
    for page in doc:
        for b in page.get_text("blocks"):
            if b[6] != 0:
                continue
            t = unicodedata.normalize("NFKC", b[4])
            t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
            t = re.sub(r"\s+", " ", t).strip()
            if t and not re.fullmatch(r"\d{1,4}", t):
                blocks.append(t)
    return blocks, len(doc)


def drop_repeated(blocks):
    counts = Counter(b for b in blocks if len(b) < 120)
    repeated = {b for b, c in counts.items() if c >= 4 and len(b.split()) >= 2}
    return [b for b in blocks if b not in repeated]


REF_PAT = re.compile(r"^(?:\d+\.?\s*|[IVX]+\.?\s*)?(references|bibliography)$", re.I)


def cut_references(blocks):
    idx = [i for i, b in enumerate(blocks) if REF_PAT.match(b)]
    if idx and idx[-1] > 0.4 * len(blocks):
        return blocks[:idx[-1]]
    return blocks


papers = []
for p in results:
    if len(papers) >= MAX_PAPERS:
        break
    pid = p.get_short_id()
    path = PDF_DIR + "/" + pid.replace("/", "_") + ".pdf"
    try:
        if not os.path.exists(path):
            resp = requests.get(p.pdf_url, timeout=60)
            resp.raise_for_status()
            with open(path, "wb") as f:
                f.write(resp.content)
            time.sleep(3)
        blocks, n_pages = extract_blocks(path)
    except Exception as e:
        print("failed:", pid, e)
        continue
    blocks = drop_repeated(blocks)
    blocks = cut_references(blocks)
    text = "\n\n".join(blocks)
    if len(text) < 10000:
        continue
    papers.append({
        "page_content": text,
        "metadata": {
            "arxiv_id": pid,
            "title": p.title,
            "published": p.published.strftime("%Y-%m-%d"),
            "n_pages": n_pages,
            "n_paragraphs": len(blocks),
        }
    })
    print(len(papers), pid)

print("Papers saved:", len(papers))

with open(OUTPUT, "w") as f:
    json.dump(papers, f, ensure_ascii=False)
