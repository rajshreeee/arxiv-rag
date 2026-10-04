import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT = "data/papers.json"
CLEAN_OUTPUT = "data/papers_clean.json"

CONFIGS = {
    "small":  {"chunk_size": 600,  "chunk_overlap": 120},
    "medium": {"chunk_size": 1200, "chunk_overlap": 240},
    "large":  {"chunk_size": 1800, "chunk_overlap": 360},
}

with open(INPUT) as f:
    saved = json.load(f)

papers = []
for item in saved:
    blocks = item["page_content"].split("\n\n")
    kept = [b for b in blocks if len(b.split()) >= 5]
    text = "\n\n".join(kept)
    papers.append(Document(page_content=text, metadata=item["metadata"]))

print("Papers loaded:", len(papers))

with open(CLEAN_OUTPUT, "w") as f:
    json.dump([{"page_content": d.page_content, "metadata": d.metadata} for d in papers],
              f, ensure_ascii=False)


def letter_ratio(text):
    letters = sum(1 for c in text if c.isalpha())
    non_space = sum(1 for c in text if not c.isspace())
    if non_space == 0:
        return 0
    return letters / non_space


for name, settings in CONFIGS.items():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings["chunk_size"],
        chunk_overlap=settings["chunk_overlap"],
        add_start_index=True,
    )
    pieces = splitter.split_documents(papers)
    kept = []
    for i, piece in enumerate(pieces):
        if len(piece.page_content) < 100:
            continue
        if letter_ratio(piece.page_content) < 0.6:
            continue
        piece.metadata["chunk_id"] = piece.metadata["arxiv_id"] + "__" + name + "__" + str(i)
        kept.append(piece)

    rows = [{"page_content": c.page_content, "metadata": c.metadata} for c in kept]
    with open("data/chunks_" + name + ".json", "w") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(name + ":", len(kept), "chunks")
