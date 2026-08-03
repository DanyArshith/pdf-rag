from pypdf import PdfReader

def preprocess_pdf(path, chunk_size):
    reader = PdfReader(path)
    chunks = []
    chunk_id = 0
    for page_no, page in enumerate(reader.pages, start = 1):
        page_text = page.extract_text()
        if page_text:
            page_text = page_text.replace("\n"," ")
            page_text = " ".join(page_text.split())

            length = len(page_text)
            i = 0
            while i + chunk_size < length:
                j = i + chunk_size
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

    return chunks