"""PDF Report Generator for NeuroFence.

Renders offline forensic analysis results as a structured, clean PDF report
using reportlab.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Union

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def render_pdf_report(results: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Generate a PDF report summarizing the scan results.

    Args:
        results: Scan results dictionary produced by ``scan_model()``.
        output_path: Path to the output PDF file.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_file),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
    )

    heading2_style = ParagraphStyle(
        "DocHeading2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )

    tbl_header_style = ParagraphStyle(
        "TblHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.white,
        alignment=1,  # Center
    )

    tbl_cell_style = ParagraphStyle(
        "TblCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1e293b"),
        alignment=1,  # Center
    )

    story = []

    # 1. Header / Title
    story.append(Paragraph("NeuroFence Backdoor Scan Report", title_style))
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph(f"Generated on {now_str} • Offline Forensics", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=12))

    # 2. Executive Summary & Verdict
    verdict = results.get("verdict", "UNKNOWN")
    score = results.get("safety_score", "—")
    prompts_tested = results.get("prompts_tested", "—")
    flagged_list: List[Dict[str, Any]] = results.get("flagged_neurons", [])
    n_flagged = len(flagged_list)

    if verdict == "CLEAN":
        verdict_color = colors.HexColor("#166534")  # Green
        verdict_bg = colors.HexColor("#dcfce7")
    elif verdict == "BACKDOOR DETECTED":
        verdict_color = colors.HexColor("#991b1b")  # Red
        verdict_bg = colors.HexColor("#fee2e2")
    else:
        verdict_color = colors.HexColor("#334155")
        verdict_bg = colors.HexColor("#f1f5f9")

    verdict_para = Paragraph(
        f"<b><font size='14' color='{verdict_color.hexval()}'>{verdict}</font></b>",
        styles["Normal"],
    )

    summary_data = [
        [
            Paragraph("<b>Overall Verdict:</b>", body_style),
            verdict_para,
            Paragraph("<b>Safety Score:</b>", body_style),
            Paragraph(f"<b>{score} / 100</b>", body_style),
        ],
        [
            Paragraph("<b>Flagged Neurons:</b>", body_style),
            Paragraph(str(n_flagged), body_style),
            Paragraph("<b>Prompts Tested:</b>", body_style),
            Paragraph(str(prompts_tested), body_style),
        ],
    ]

    summary_table = Table(summary_data, colWidths=[105, 160, 105, 170])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BACKGROUND", (1, 0), (1, 0), verdict_bg),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 14))

    # 3. Model Architecture & Parameters
    metadata = results.get("metadata") or results.get("_meta") or {}
    model_hash = results.get("model_hash") or metadata.get("hash_sha256") or metadata.get("model_hash") or "—"
    model_type = metadata.get("model_type", "gpt2")
    num_layers = metadata.get("num_layers") or metadata.get("n_layers") or "—"
    hidden_size = metadata.get("hidden_size") or metadata.get("n_neurons") or "—"
    vocab_size = metadata.get("vocab_size", "—")
    model_dir = metadata.get("model_dir", "—")
    baseline_path = metadata.get("baseline_path", "—")

    story.append(Paragraph("Model Architecture & Target", heading2_style))

    meta_data = [
        [
            Paragraph("<b>Model Hash (SHA-256):</b>", body_style),
            Paragraph(f"<font size='7'>{model_hash}</font>", body_style),
        ],
        [
            Paragraph("<b>Architecture:</b>", body_style),
            Paragraph(f"{model_type} ({num_layers} layers, {hidden_size} hidden size, vocab {vocab_size})", body_style),
        ],
        [
            Paragraph("<b>Model Path:</b>", body_style),
            Paragraph(f"<font size='7'>{model_dir}</font>", body_style),
        ],
        [
            Paragraph("<b>Baseline Reference:</b>", body_style),
            Paragraph(f"<font size='7'>{baseline_path}</font>", body_style),
        ],
    ]

    meta_table = Table(meta_data, colWidths=[140, 400])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # 4. Flagged Neurons Table
    story.append(Paragraph("Flagged Backdoor Neurons", heading2_style))

    if not flagged_list:
        no_flags_msg = Paragraph(
            "<i>No suspicious backdoor neurons were detected above the exceedance thresholds.</i>",
            body_style,
        )
        story.append(no_flags_msg)
    else:
        headers = [
            Paragraph("<b>#</b>", tbl_header_style),
            Paragraph("<b>Layer</b>", tbl_header_style),
            Paragraph("<b>Neuron</b>", tbl_header_style),
            Paragraph("<b>Word</b>", tbl_header_style),
            Paragraph("<b>Consistency</b>", tbl_header_style),
            Paragraph("<b>Median Margin</b>", tbl_header_style),
            Paragraph("<b>Normal Fire Rate</b>", tbl_header_style),
        ]
        table_rows = [headers]

        for rank, item in enumerate(flagged_list, start=1):
            consistency = item.get("consistency", 0.0)
            median_margin = item.get("median_margin", 0.0)
            normal_fire = item.get("normal_fire_rate", 0.0)

            c_text = f"{consistency * 100:.1f}%" if isinstance(consistency, (int, float)) else str(consistency)
            m_text = f"{median_margin:.2f}σ" if isinstance(median_margin, (int, float)) else str(median_margin)
            n_text = f"{normal_fire * 100:.1f}%" if isinstance(normal_fire, (int, float)) else str(normal_fire)

            row = [
                Paragraph(str(rank), tbl_cell_style),
                Paragraph(str(item.get("layer", "—")), tbl_cell_style),
                Paragraph(str(item.get("neuron", "—")), tbl_cell_style),
                Paragraph(str(item.get("word", "—")), tbl_cell_style),
                Paragraph(c_text, tbl_cell_style),
                Paragraph(m_text, tbl_cell_style),
                Paragraph(n_text, tbl_cell_style),
            ]
            table_rows.append(row)

        neuron_table = Table(
            table_rows,
            colWidths=[30, 45, 60, 105, 95, 105, 100],
        )
        neuron_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(neuron_table)

    story.append(Spacer(1, 14))

    # 5. Forensic Limitations & Disclaimers
    story.append(Paragraph("Forensic Limitations & Methodology", heading2_style))
    limitations = results.get("limitations", [])
    if not limitations:
        limitations = [
            "This report is heuristic evidence based on activation statistics relative to a clean-model baseline.",
            "Detection relies on the injected neuron being dormant in the baseline and consistently activated by candidate trigger words.",
        ]

    for lim in limitations:
        bullet_text = f"• {lim}"
        story.append(Paragraph(bullet_text, body_style))
        story.append(Spacer(1, 4))

    # Build document
    doc.build(story)
