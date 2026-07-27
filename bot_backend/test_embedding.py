from app.rag.loader import PDFLoader
from app.rag.splitter import DocumentSplitter
from app.rag.embedding import EmbeddingModel


# Load PDFs
loader = PDFLoader()
documents = loader.load_documents()

print(f"\nDocuments Loaded : {len(documents)}")


# Split Documents
splitter = DocumentSplitter()

chunks = splitter.split_documents(documents)

print(f"Chunks Created : {len(chunks)}")


# Create Embeddings
embedding_model = EmbeddingModel()

texts = embedding_model.get_texts(chunks)

embeddings = embedding_model.create_embeddings(texts)

embedded_documents = embedding_model.build_documents(
    chunks,
    embeddings
)

print(f"Embeddings Created : {len(embedded_documents)}")

print("=" * 60)

print("Sample Embedded Document")

print("=" * 60)

print("Source :")
print(embedded_documents[0]["source"])

print("\nText :")
print(embedded_documents[0]["text"][:300])

print("\nEmbedding Dimension :")
print(len(embedded_documents[0]["embedding"]))

print("=" * 60)