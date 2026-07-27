from app.rag.loader import PDFLoader
from app.rag.splitter import DocumentSplitter


# Load PDFs
loader = PDFLoader()
documents = loader.load_documents()

print("=" * 60)
print(f"Total Documents : {len(documents)}")
print("=" * 60)


# Split into Chunks
splitter = DocumentSplitter()

chunks = splitter.split_documents(documents)

print(f"Total Chunks : {len(chunks)}")

print("=" * 60)


# Print First 3 Chunks
for i, chunk in enumerate(chunks[:3], start=1):

    print(f"\nChunk {i}")

    print(f"Source : {chunk['source']}")

    print(f"Characters : {len(chunk['text'])}")

    print("-" * 60)

    print(chunk["text"])

    print("=" * 60)