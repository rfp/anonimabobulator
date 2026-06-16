from __future__ import annotations

import html
import io

import pymupdf
from fastapi import HTTPException
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate

from config import MIN_TEXT_CHARS, OCR_ENABLED

_DEJAVU      = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_DEJAVU_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _ocr_pages(document: "pymupdf.Document") -> list[str]:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise HTTPException(
            422, "OCR is enabled but pytesseract / Pillow are not installed.",
        ) from exc
    texts = []
    for page in document:
        mat = pymupdf.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csGRAY)
        img = Image.frombytes("L", [pix.width, pix.height], pix.samples)
        texts.append(pytesseract.image_to_string(img))
    return texts


def extract_pages(pdf_bytes: bytes) -> tuple[list[str], list[tuple[float, float]]]:
    """Return (page_texts, page_sizes) where sizes are (width, height) in PDF points."""
    try:
        document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise HTTPException(422, "Invalid or damaged PDF.") from exc

    if document.needs_pass:
        document.close()
        raise HTTPException(422, "Password-protected PDFs are not supported.")

    if document.page_count == 0:
        document.close()
        raise HTTPException(422, "The PDF contains no pages.")

    pages: list[str] = []
    sizes: list[tuple[float, float]] = []
    try:
        for page in document:
            pages.append(page.get_text("text", sort=True).strip())
            sizes.append((page.rect.width, page.rect.height))

        if sum(ch.isalnum() for text in pages for ch in text) < MIN_TEXT_CHARS:
            if OCR_ENABLED:
                pages = _ocr_pages(document)
            else:
                raise HTTPException(
                    422,
                    "The PDF has no usable text layer. Make sure the PDF was exported using Print to PDF.",
                )
    finally:
        document.close()

    return pages, sizes


def build_pdf(
    pages: list[str],
    metadata: list[dict[str, object]],
    page_size: tuple[float, float] = A4,
) -> bytes:
    output = io.BytesIO()
    pdfmetrics.registerFont(TTFont("DejaVu",      _DEJAVU))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", _DEJAVU_BOLD))

    doc = SimpleDocTemplate(
        output, pagesize=page_size,
        rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=16 * mm,   bottomMargin=16 * mm,
        title="Anonymized Ticket — Anonimabobulator",
        author="Anonimabobulator",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TicketTitle", parent=styles["Heading1"],
        fontName="DejaVu-Bold", fontSize=15, leading=19,
        alignment=TA_LEFT, spaceAfter=8,
    )
    info_style = ParagraphStyle(
        "Info", parent=styles["Normal"],
        fontName="DejaVu", fontSize=8, leading=11,
        textColor="#555555", spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "TicketBody", parent=styles["Code"],
        fontName="DejaVu", fontSize=8.5, leading=11.5,
        leftIndent=0, rightIndent=0, wordWrap="CJK",
    )

    story = []
    for index, text in enumerate(pages):
        data = metadata[index]
        story.append(Paragraph(f"Anonimabobulator · Page {index + 1}", title_style))
        story.append(Paragraph(
            f"Detected language: {html.escape(str(data['language']))} "
            f"({float(data['language_confidence']):.1%}) · "
            f"Replacements: {int(data['entities'])}",
            info_style,
        ))
        story.append(Paragraph(html.escape(text).replace("\n", "<br/>") or " ", body_style))
        if index < len(pages) - 1:
            story.append(PageBreak())

    doc.build(story)
    return output.getvalue()


def redact_pdf(pdf_bytes: bytes, search_pairs: list[tuple[str, str]]) -> bytes:
    """Redact PII in the original PDF in-place, preserving layout.

    Applies PyMuPDF redaction annotations for every (original_text, token) pair
    on every page, then finalises them.  Longer strings are processed first so
    shorter substrings don't preempt them (e.g. "John" before "John Smith").
    """
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise HTTPException(422, "Invalid or damaged PDF.") from exc

    ordered = sorted(search_pairs, key=lambda p: len(p[0]), reverse=True)

    for page in doc:
        for original, token in ordered:
            if len(original) < 3:
                continue
            for rect in page.search_for(original):
                page.add_redact_annot(
                    rect,
                    text=token,
                    fontname="helv",
                    fontsize=7,
                    fill=(0.91, 0.94, 1.00),
                    text_color=(0.05, 0.10, 0.55),
                )
        for img in page.get_image_info():
            page.add_redact_annot(
                pymupdf.Rect(img["bbox"]),
                text="[IMAGE]",
                fontname="helv",
                fontsize=8,
                fill=(0.91, 0.94, 1.00),
                text_color=(0.05, 0.10, 0.55),
                align=1,
            )
        page.apply_redactions(images=1)  # 1 = PDF_REDACT_IMAGE_REMOVE

    buf = io.BytesIO()
    doc.save(buf, garbage=4, deflate=True)
    doc.close()
    return buf.getvalue()
