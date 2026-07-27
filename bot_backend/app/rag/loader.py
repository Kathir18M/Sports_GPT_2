import sys
import fitz
from pathlib import Path
from app.core.config import settings

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


class PDFLoader:

    def __init__(self, data_folder=None):
        # Resolve the data folder. Prefer explicit argument, then env var, then default "data".
        if data_folder is None:
            env_path = settings.PDF_DATA_PATH
            data_folder = env_path if env_path else "data"
        path = Path(data_folder)
        if not path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            path = project_root / data_folder
        self.data_folder = path

    def load_documents(self):

        documents = []

        if not self.data_folder.exists():
            self.data_folder.mkdir(parents=True, exist_ok=True)
            print(f"Warning: Data directory '{self.data_folder}' did not exist. Created directory.")
            # Continue after creating the directory; there may still be no PDFs.


        pdf_files = list(self.data_folder.rglob("*.pdf"))

        if not pdf_files:
            print(f"Warning: No PDF files found in '{self.data_folder}'.")

        for pdf in pdf_files:

            print(f"Loading: {pdf.name}")

            try:

                document = fitz.open(pdf)

                text = ""

                for page in document:
                    text += page.get_text()

                document.close()

                documents.append({
                    "source": str(pdf),
                    "text": text
                })

            except Exception as e:
                print(f"Error reading {pdf.name}")
                print(e)

        return documents