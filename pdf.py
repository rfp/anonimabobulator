from __future__ import annotations

import html
import io
import re

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


def _page_chars(page: "pymupdf.Page") -> tuple[str, list["pymupdf.Rect | None"]]:
    """Return the page text as one string plus the bbox of every character.

    Lines are joined with "\\n", whose bbox is None, so a match can span lines
    while its rectangles are still split per line.
    """
    chars: list[str] = []
    boxes: list[pymupdf.Rect | None] = []
    for block in page.get_text("rawdict", sort=True)["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                for ch in span["chars"]:
                    chars.append(ch["c"])
                    boxes.append(pymupdf.Rect(ch["bbox"]))
            chars.append("\n")
            boxes.append(None)
    return "".join(chars), boxes


def _find_whole_word_rects(
    original: str,
    text: str,
    boxes: list["pymupdf.Rect | None"],
    claimed: bytearray,
) -> list["pymupdf.Rect"]:
    """Find case-sensitive, whole-word occurrences of *original* in *text*.

    Unlike page.search_for() (case-insensitive substring search), this never
    matches inside a longer word, so a detected "Create" leaves "Created" alone.
    Characters already covered by a longer match are skipped via *claimed*.
    """
    words = original.split()
    if not words:
        return []
    pattern = r"(?<!\w)" + r"\s+".join(map(re.escape, words)) + r"(?!\w)"
    rects: list[pymupdf.Rect] = []
    for match in re.finditer(pattern, text):
        if any(claimed[match.start():match.end()]):
            continue
        claimed[match.start():match.end()] = b"\x01" * (match.end() - match.start())
        current: pymupdf.Rect | None = None
        for i in range(match.start(), match.end()):
            box = boxes[i]
            if box is None:
                if current is not None:
                    rects.append(current)
                current = None
            elif not text[i].isspace():
                current = pymupdf.Rect(box) if current is None else current | box
        if current is not None:
            rects.append(current)
    return rects


def redact_pdf(pdf_bytes: bytes, search_pairs: list[tuple[str, str]]) -> bytes:
    """Redact PII in the original PDF in-place, preserving layout.

    Applies PyMuPDF redaction annotations for every whole-word, case-sensitive
    occurrence of each (original_text, token) pair on every page, then
    finalises them.  Longer strings are processed first and claim their
    characters, so shorter ones can't preempt them (e.g. "John" inside
    "John Smith").  All embedded images are then removed in a separate pass
    that leaves text intact.
    """
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise HTTPException(422, "Invalid or damaged PDF.") from exc

    ordered = sorted(search_pairs, key=lambda p: len(p[0]), reverse=True)

    for page in doc:
        text, boxes = _page_chars(page)
        claimed = bytearray(len(text))
        for original, token in ordered:
            if len(original) < 3:
                continue
            for rect in _find_whole_word_rects(original, text, boxes, claimed):
                page.add_redact_annot(
                    rect,
                    text=token,
                    fontname="helv",
                    fontsize=7,
                    fill=(0.91, 0.94, 1.00),
                    text_color=(0.05, 0.10, 0.55),
                )
        page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE)

        # Second pass: remove every embedded image without touching text.
        # Browser "Print to PDF" output often renders panels and boxes as
        # background images under the text; redacting those with the default
        # text mode would wipe all the text on top of them.  Images with text
        # over them are removed silently (no fill, so the text stays visible);
        # only images with nothing on top get the [IMAGE] placeholder.
        for img in page.get_image_info():
            rect = pymupdf.Rect(img["bbox"])
            if page.get_text("text", clip=rect).strip():
                page.add_redact_annot(rect, fill=False)
            else:
                page.add_redact_annot(
                    rect,
                    text="[IMAGE]",
                    fontname="helv",
                    fontsize=8,
                    fill=(0.91, 0.94, 1.00),
                    text_color=(0.05, 0.10, 0.55),
                    align=1,
                )
        page.apply_redactions(
            images=pymupdf.PDF_REDACT_IMAGE_REMOVE,
            graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
            text=pymupdf.PDF_REDACT_TEXT_NONE,
        )

    buf = io.BytesIO()
    doc.save(buf, garbage=4, deflate=True)
    doc.close()
    return buf.getvalue()
