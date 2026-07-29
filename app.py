from pypdf import PdfReader

path = "data/documents/ML_u1.pdf"
reader = PdfReader(path)            # opens the pdf

total_pages = len(reader.pages)
print("no of pages:",total_pages)   # there are 25 page in the document

first_page = reader.pages[0]

print(first_page.extract_text())    # prints the text in the first page
