from app.rag.loader import PDFLoader

loader = PDFLoader()

documents = loader.load_documents()

print("\n" + "=" * 60)
print(f"Total PDFs Loaded : {len(documents)}")
print("=" * 60)

for doc in documents:

    print(f"\nSource : {doc['source']}")
    print("-" * 60)

    print(doc["text"][:500])

    print("=" * 60)