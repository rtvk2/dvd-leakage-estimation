"""Build report/report.docx from the result files (so that every number in the report is read from a file in data/ or results/).

    python report/build_report.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
RES, DATA = ROOT / "results", ROOT / "data"

doc = Document()
sec = doc.sections[0]
sec.left_margin = sec.right_margin = Inches(0.9)
sec.top_margin = sec.bottom_margin = Inches(0.8)
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(10.5)
normal.paragraph_format.space_after = Pt(5)


def runs(par, text):
    """Add text to a paragraph; **bold** and `code` spans are formatted."""
    for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if part.startswith("**"):
            par.add_run(part[2:-2]).bold = True
        elif part.startswith("`"):
            r = par.add_run(part[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(9.5)
        elif part:
            par.add_run(part)


def H(text, level=1):
    doc.add_heading(text, level)


def P(text):
    runs(doc.add_paragraph(), text)


def B(items):
    for it in items:
        runs(doc.add_paragraph(style="List Bullet"), it)


def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), color)
    tcPr.append(shd)


def T(rows, header, widths=None, caption=None, size=9):
    """rows: list of lists of strings; header: list of strings."""
    if caption:
        c = doc.add_paragraph()
        runs(c, f"**{caption}**")
        c.paragraph_format.space_after = Pt(2)
        c.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        runs(cell.paragraphs[0], f"**{h}**")
        shade(cell, "D9E2F3")
    for r in rows:
        cells = t.add_row().cells
        for i, v in enumerate(r):
            cells[i].text = ""
            runs(cells[i].paragraphs[0], str(v))
    for row in t.rows:
        for i, cell in enumerate(row.cells):
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                for rn in p.runs:
                    rn.font.size = Pt(size)
            if widths:
                cell.width = Inches(widths[i])
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def F(name, caption, width=6.3):
    doc.add_picture(str(RES / name), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph()
    runs(c, caption)
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for rn in c.runs:
        rn.italic = True
        rn.font.size = Pt(9)


def have(name):
    return (RES / name).exists() or (DATA / name).exists()


def pending(text):
    P(f"**[DRAFT: {text}]**")


def pct(x, d=2):
    return f"{x:.{d}f}"


def sci(x):
    return f"{x:.2e}"


CASE = {"nmos_on": "NMOS ON", "nmos_off": "NMOS OFF", "pmos_on": "PMOS ON", "pmos_off": "PMOS OFF"}
GATE = {"inv": "INV", "nand2": "NAND2", "nor2": "NOR2", "and2": "AND2", "or2": "OR2"}

# ======================================================================================================================
title = doc.add_heading("ML-Driven PVT Variation Aware Leakage Estimation", 0)
P("Digital VLSI Design course project. Model file **16nm_HP**, nominal **VDD 0.9 V**, test circuit **3-bit carry-lookahead adder**.")

exec(open(ROOT / "report" / "sections.py", encoding="utf-8").read())
exec(open(ROOT / "report" / "sections_results.py", encoding="utf-8").read())
exec(open(ROOT / "report" / "sections_circuit.py", encoding="utf-8").read())

doc.save(ROOT / "report" / "report.docx")
print("saved report/report.docx")
