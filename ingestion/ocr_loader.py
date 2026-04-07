import base64
import os
import tempfile
from groq import Groq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

VISION_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview"
]

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_text_with_groq(image_data):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    for model in VISION_MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"}
                        },
                        {
                            "type": "text",
                            "text": """Extract all text from this medical report.
Preserve structure exactly as it appears.
For tables use: Test Name | Value | Unit | Reference Range
Include all values, patient info and clinical notes.
Return only extracted text."""
                        }
                    ]
                }],
                max_tokens=2000
            )
            print(f"✅ Vision model used: {model}")
            return response.choices[0].message.content
        except Exception as e:
            print(f"Model {model} failed: {e}")
            continue
    return ""

def process_uploaded_image(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        image_data = encode_image(tmp_path)
        extracted_text = extract_text_with_groq(image_data)
    except Exception as e:
        print(f"Groq Vision error: {e}")
        extracted_text = ""
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    if not extracted_text.strip():
        return []

    doc = Document(
        page_content=extracted_text,
        metadata={"source": uploaded_file.name, "type": "vision_api"}
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents([doc])

def process_image_bytes(img_bytes, filename):
    image_data = base64.b64encode(img_bytes).decode("utf-8")
    extracted_text = extract_text_with_groq(image_data)

    if not extracted_text.strip():
        return []

    doc = Document(
        page_content=extracted_text,
        metadata={"source": filename, "type": "scanned_pdf"}
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents([doc])
