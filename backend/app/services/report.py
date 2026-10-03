"""
PDF Referral Report Generation Service for E-Dermatologist
Generates professional 1-page clinical triage summaries using ReportLab.
Includes optional patient lesion photograph thumbnail.
"""

import io
import base64
import datetime
from typing import Dict, Any, Optional
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    HRFlowable
)
from reportlab.lib.units import inch


def generate_referral_pdf(
    result_data: Dict[str, Any],
    patient_ref: Optional[str] = "Anonymous Screening",
    notes: Optional[str] = None,
    image_base64: Optional[str] = None
) -> bytes:
    """
    Generates a high-quality 1-page Clinical Referral Summary in PDF format.
    Embeds lesion photo thumbnail if provided.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=2
    )

    sub_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#475569'),
        spaceAfter=10
    )

    section_header = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=6,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#334155')
    )

    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7,
        leading=10,
        textColor=colors.HexColor('#64748B'),
        alignment=1
    )

    elements = []

    # 1. Header Banner
    elements.append(Paragraph("E-DERMATOLOGIST CLINICAL SCREENING REPORT", title_style))
    elements.append(Paragraph(
        "Standardized Macro Photography with 3D Optical Spacer &bull; Kaushalya Hospital Collaboration",
        sub_style
    ))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563EB'), spaceAfter=8))

    # 2. Case & Capture Metadata Table
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    req_id = result_data.get("request_id", "N/A")
    modality = result_data.get("modality", "mobile_spacer")
    model_ver = result_data.get("model_version", "v2.0-mobile")

    meta_data = [
        [
            Paragraph("<b>Patient Identifier:</b> " + str(patient_ref), body_style),
            Paragraph(f"<b>Date/Time:</b> {now_str}", body_style)
        ],
        [
            Paragraph(f"<b>Screening ID:</b> {req_id}", body_style),
            Paragraph(f"<b>Modality:</b> 3D Optical Spacer ({modality})", body_style)
        ],
        [
            Paragraph(f"<b>Model Engine:</b> EfficientNet-B0 ({model_ver})", body_style),
            Paragraph(f"<b>Inference Latency:</b> {result_data.get('latency_ms', 0):.1f} ms", body_style)
        ]
    ]

    meta_table = Table(meta_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 10))

    # 3. Primary Triage Finding Card (with optional side-by-side Lesion Thumbnail)
    pred = result_data.get("prediction", {})
    advice = result_data.get("advice", {})
    level = advice.get("level", "ok")

    is_urgent = (level == "urgent" or pred.get("severity") == "urgent")
    is_uncertain = (level == "uncertain")

    if is_urgent:
        banner_bg = colors.HexColor('#FEF2F2')
        banner_border = colors.HexColor('#DC2626')
        badge_text = "URGENT SPECIALIST REFERRAL RECOMMENDED"
        badge_color = colors.HexColor('#DC2626')
    elif is_uncertain:
        banner_bg = colors.HexColor('#FFFBEB')
        banner_border = colors.HexColor('#D97706')
        badge_text = "UNCERTAIN PRESENTATION - PHYSICAL EXAM REQUIRED"
        badge_color = colors.HexColor('#D97706')
    else:
        banner_bg = colors.HexColor('#EFF6FF')
        banner_border = colors.HexColor('#2563EB')
        badge_text = "PRIMARY CARE SCREENING - ROUTINE MANAGEMENT"
        badge_color = colors.HexColor('#2563EB')

    triage_info = [
        Paragraph(f"<b><font color='{badge_color}'>{badge_text}</font></b>", ParagraphStyle('Badge', parent=body_style, fontSize=9.5, leading=12)),
        Paragraph(f"<b>Primary Finding:</b> {pred.get('label', 'N/A')} (Confidence: {pred.get('confidence', 0)*100:.1f}%)", ParagraphStyle('PredBold', parent=body_style, fontSize=11, leading=15, textColor=colors.HexColor('#0F172A'))),
        Paragraph(f"<b>Clinical Advisory:</b> {advice.get('message', '')}", body_style)
    ]

    # Process image thumbnail if available
    img_element = None
    if image_base64:
        try:
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            raw_bytes = base64.b64decode(image_base64)
            pil_img = PILImage.open(io.BytesIO(raw_bytes))
            # Resize thumbnail cleanly
            pil_img.thumbnail((160, 160))
            thumb_io = io.BytesIO()
            pil_img.save(thumb_io, format="JPEG", quality=85)
            thumb_io.seek(0)
            img_element = RLImage(thumb_io, width=1.35 * inch, height=1.35 * inch)
        except Exception as err:
            print(f"[Warning] Failed to embed image in PDF: {err}")

    if img_element:
        card_table_data = [[triage_info, img_element]]
        card_col_widths = [410, 130]
    else:
        card_table_data = [[triage_info]]
        card_col_widths = [540]

    triage_table = Table(card_table_data, colWidths=card_col_widths)
    triage_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), banner_bg),
        ('BOX', (0, 0), (-1, -1), 1.5, banner_border),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (1, 0), (1, 0), 'CENTER') if img_element else ('ALIGN', (0, 0), (0, 0), 'LEFT'),
    ]))
    elements.append(triage_table)
    elements.append(Spacer(1, 10))

    # 4. Top-3 Differential Diagnoses Table
    elements.append(Paragraph("Differential Distribution (Top-3 Model Hypotheses)", section_header))
    top3 = result_data.get("top3", [])
    diff_data = [["Rank", "Hypothesized Condition", "Severity Band", "Model Probability"]]

    for idx, item in enumerate(top3):
        cls_id = item.get("class_id", "")
        sev = "Urgent Referral" if cls_id == "suspicious_lesion" else ("Normal" if cls_id == "healthy" else "Common")
        diff_data.append([
            f"#{idx+1}",
            item.get("label", cls_id),
            sev,
            f"{item.get('probability', 0)*100:.1f}%"
        ])

    diff_table = Table(diff_data, colWidths=[45, 230, 145, 120])
    diff_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8.5),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FFFFFF')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#FFFFFF'), colors.HexColor('#F8FAFC')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('ALIGN', (3, 0), (3, -1), 'RIGHT'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
    ]))
    elements.append(diff_table)
    elements.append(Spacer(1, 10))

    # 5. Optical & Capture Quality Metrics
    elements.append(Paragraph("Optical Assessment & Normalization Telemetry", section_header))
    q = result_data.get("quality", {})
    quality_rows = [
        [
            Paragraph(f"<b>Focus Sharpness (Laplacian):</b> {q.get('blur_score', 0):.1f} (Min req: 55.0)", body_style),
            Paragraph(f"<b>Mean Illumination:</b> {q.get('brightness', 0):.1f} / 255", body_style)
        ],
        [
            Paragraph(f"<b>Specular Glare Ratio:</b> {q.get('glare_ratio', 0)*100:.1f}% (Max: 12%)", body_style),
            Paragraph(f"<b>Colour Constancy:</b> Shades-of-Gray (Minkowski p=6)", body_style)
        ],
        [
            Paragraph(f"<b>Reference Patch:</b> {'Detected' if q.get('reference_patch_found') else 'Not Detected'}", body_style),
            Paragraph(f"<b>Overall Quality Gate:</b> <b>{q.get('status', 'good').upper()}</b>", body_style)
        ]
    ]
    quality_table = Table(quality_rows, colWidths=[270, 270])
    quality_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(quality_table)

    if notes:
        elements.append(Spacer(1, 8))
        elements.append(Paragraph(f"<b>Clinician Notes:</b> {notes}", body_style))

    # 6. Medical Disclaimer Footer
    elements.append(Spacer(1, 14))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E1'), spaceAfter=6))
    elements.append(Paragraph(
        "<b>CONFIDENTIAL MEDICAL SCREENING DECISION AID:</b> This report is generated by an artificial intelligence decision-support tool "
        "designed for community health worker triage. It does NOT provide a definitive diagnosis or replace a formal examination by a dermatologist. "
        "All suspicious, advancing, or non-responsive lesions require in-person biopsy or clinical dermatological evaluation.",
        disclaimer_style
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
