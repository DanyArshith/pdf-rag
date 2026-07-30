from pypdf import PdfReader

path = "data/documents/ML_u1.pdf"
reader = PdfReader(path)            # opens the pdf

total_pages = len(reader.pages)
print("no of pages:",total_pages)   # there are 25 page in the document

chunks = []

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        page_text = page_text.replace("\n"," ")
        page_text = " ".join(page_text.split())

        length = len(page_text)
        i = 0
        while i + 100 < length:                    #split the text into 100 characters each chunk
            j = i + 100
            while page_text[j] != " ":
                j -= 1

            chunk = page_text[i:j]
            chunks.append(chunk)
            i = j + 1
        last_chunk = page_text[i:]
        chunks.append(last_chunk)

print(chunks)