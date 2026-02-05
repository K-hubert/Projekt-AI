import fitz  
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class PDFPage:
    source: str   
    page: int    
    text: str

def extract_pages_from_pdf_bytes(pdf_bytes: bytes, filename: str) -> List[PDFPage]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: List[PDFPage] = []
    for i in range(doc.page_count):
        p = doc.load_page(i)
        text = (p.get_text("text") or "").strip()
        if text:
            pages.append(PDFPage(source=filename, page=i + 1, text=text))
    doc.close()
    return pages
