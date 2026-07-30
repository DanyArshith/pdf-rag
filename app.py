from pypdf import PdfReader

path = "data/documents/ML_u1.pdf"
reader = PdfReader(path)            # opens the pdf

total_pages = len(reader.pages)
print("no of pages:",total_pages)   # there are 25 page in the document

first_page = reader.pages[0]

# print(first_page.extract_text())    # prints the text in the first page

first_page_text = first_page.extract_text()

length = len(first_page_text)
print(length)

chunks = []
i = 0
while i + 100 < length:                     #split the text into 100 characters each chunk
    chunk = first_page_text[i:i+100]
    chunks.append(chunk)
    i += 100
last_chunk = first_page_text[i:]
chunks.append(last_chunk)

print(chunks)