from pypdf import PdfReader
from pathlib import Path

path = Path("data/documents/ML_u1.pdf")
reader = PdfReader(path)

total_pages = len(reader.pages)
print("no of pages:",total_pages)

chunks = []
CHUNK_SIZE = 100
chunk_id = 0
for page_no, page in enumerate(reader.pages, start = 1):
    page_text = page.extract_text()
    if page_text:
        page_text = page_text.replace("\n"," ")
        page_text = " ".join(page_text.split())

        length = len(page_text)
        i = 0
        while i + CHUNK_SIZE < length:
            j = i + CHUNK_SIZE
            while page_text[j] != " ":
                j -= 1

            text = page_text[i:j]
            chunk = {
                "id" : chunk_id,
                "text" : text,
                "page" : page_no,
                "source" : path.name
            }
            chunk_id += 1
            chunks.append(chunk)
            i = j + 1
        last_text = page_text[i:]
        chunk = {
            "id" : chunk_id,
            "text" : last_text,
            "page" : page_no,
            "source" : path.name
        }
        chunk_id += 1
        chunks.append(chunk)
