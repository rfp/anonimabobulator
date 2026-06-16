"""
Stub out the heavy ML dependencies before any project module is imported.
This keeps the test suite fast and runnable without model files on disk.
"""
import sys
from unittest.mock import MagicMock

# fasttext -------------------------------------------------------------------
_ft_stub = MagicMock()
_ft_module_stub = MagicMock()
sys.modules["fasttext"] = _ft_stub
sys.modules["fasttext.FastText"] = _ft_module_stub

# GLiNER ---------------------------------------------------------------------
sys.modules["gliner"] = MagicMock()

# pymupdf --------------------------------------------------------------------
sys.modules["pymupdf"] = MagicMock()

# reportlab ------------------------------------------------------------------
for _mod in [
    "reportlab",
    "reportlab.lib",
    "reportlab.lib.enums",
    "reportlab.lib.pagesizes",
    "reportlab.lib.styles",
    "reportlab.lib.units",
    "reportlab.pdfbase",
    "reportlab.pdfbase.pdfmetrics",
    "reportlab.pdfbase.ttfonts",
    "reportlab.platypus",
]:
    sys.modules.setdefault(_mod, MagicMock())
