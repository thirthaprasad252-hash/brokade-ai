import fitz  # PyMuPDF
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    return "".join(page.get_text() for page in doc)

def load_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())

def load_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_file(file_path: str, file_type: str) -> str:
    loaders = {"pdf": load_pdf, "docx": load_docx, "txt": load_txt}
    loader = loaders.get(file_type)
    if not loader:
        raise ValueError(f"Unsupported file type: {file_type}")
    return loader(file_path)

def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)