import fitz, os, sys

# Path to the PDF (relative to project root)
pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'sample.pdf'))

if not os.path.exists(pdf_path):
    print(f"PDF not found at {pdf_path}")
    sys.exit(1)

doc = fitz.open(pdf_path)
full_text = []
for page_num, page in enumerate(doc, start=1):
    text = page.get_text()
    full_text.append(f"--- Page {page_num} ---\n{text}\n")

doc.close()
print("\n".join(full_text))
