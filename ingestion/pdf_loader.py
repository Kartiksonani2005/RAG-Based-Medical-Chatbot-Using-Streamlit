from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import tempfile
import os
import fitz

def process_uploaded_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    chunks = []

    try:
        # Check if PDF has real text
        doc = fitz.open(tmp_path)
        total_text = "".join([page.get_text() for page in doc])
        doc.close()

        if len(total_text.strip()) > 50:
            # Real text PDF — use PyMuPDF directly
            print("✅ Text PDF — using PyMuPDF")
            loader = PyMuPDFLoader(tmp_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
        else:
            # Scanned PDF — convert to image and use Groq Vision
            print("⚠️ Scanned PDF — converting to image for Groq Vision")
            doc = fitz.open(tmp_path)
            from ingestion.ocr_loader import process_image_bytes
            all_chunks = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_bytes = pix.tobytes("png")
                page_chunks = process_image_bytes(img_bytes, f"{uploaded_file.name}_page{page_num+1}")
                all_chunks.extend(page_chunks)
            doc.close()
            chunks = all_chunks

    except Exception as e:
        print(f"PDF error: {e}")

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return chunks