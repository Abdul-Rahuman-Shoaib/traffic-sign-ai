"""
PDF Report Generator for Vision-Based Traffic Sign Recognition System
PSN Engineering College, Tirunelveli - Dept. of Computer Science & Engineering
"""
import os
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)

REPORT_DIR = os.path.join(os.path.dirname(__file__), "static", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


def generate_pdf_report(username, detections, user_full_name="Driver / Student"):
    """
    Generates a full comprehensive PDF detection report with academic project headers,
    statistical metrics, and detailed tabular logs.
    """
    filename = f"report_{username}_{int(datetime.utcnow().timestamp())}.pdf"
    filepath = os.path.join(REPORT_DIR, filename)

    doc = SimpleDocTemplate(
        filepath,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CollegeHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0f2744"),
        alignment=1
    )

    sub_style = ParagraphStyle(
        "DeptHeader",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
        alignment=1
    )

    project_title_style = ParagraphStyle(
        "ProjectTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1d4ed8"),
        alignment=1
    )

    h2_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6
    )

    cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1e293b")
    )

    cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0f172a")
    )

    story = []

    # 1. College & Project Banner Header
    story.append(Paragraph("PSN ENGINEERING COLLEGE, TIRUNELVELI", title_style))
    story.append(Paragraph("DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING", sub_style))
    story.append(Paragraph("<b>Academic Year:</b> 2026 - 2027 &nbsp;|&nbsp; <b>Final Year Project Presentation</b>", sub_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph("VISION-BASED TRAFFIC SIGN RECOGNITION AND ROAD SAFETY ASSISTANT", project_title_style))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=10))

    # 2. Project Team & Document Meta Table
    team_text = (
        "<b>Project Team:</b><br/>"
        "• Abdul Rahuman Shoaib (952323104001)<br/>"
        "• Esakkiraja (952323104009)<br/>"
        "• Jebicson Francis (952323104014)<br/>"
        "• K.G. Maharajan (952323104501)"
    )
    meta_text = (
        f"<b>Audit Session Info:</b><br/>"
        f"<b>Operator Account:</b> {username}<br/>"
        f"<b>Report ID:</b> TR-{int(datetime.utcnow().timestamp())}<br/>"
        f"<b>Generated on:</b> {datetime.utcnow().strftime('%d %b %Y, %H:%M UTC')}<br/>"
        f"<b>Total Logged Detections:</b> {len(detections)}"
    )

    meta_table_data = [
        [Paragraph(team_text, cell_style), Paragraph(meta_text, cell_style)]
    ]
    meta_table = Table(meta_table_data, colWidths=[270, 250])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # 3. Detection Summary Metrics
    if detections:
        avg_conf = sum(d.confidence for d in detections) / len(detections) * 100
        high_conf_count = sum(1 for d in detections if d.confidence >= 0.85)

        stats_data = [
            [
                Paragraph("<b>Total Scans</b>", cell_bold),
                Paragraph(f"<b>{len(detections)}</b>", title_style),
                Paragraph("<b>Avg. Confidence</b>", cell_bold),
                Paragraph(f"<b>{avg_conf:.1f}%</b>", title_style),
                Paragraph("<b>High Certainty (≥85%)</b>", cell_bold),
                Paragraph(f"<b>{high_conf_count}</b>", title_style)
            ]
        ]
        stats_table = Table(stats_data, colWidths=[85, 85, 95, 85, 95, 75])
        stats_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eff6ff")),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#93c5fd")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(Paragraph("Detection Session Performance Metrics", h2_style))
        story.append(stats_table)
        story.append(Spacer(1, 14))

    # 4. Detailed Tabular Detection Records
    story.append(Paragraph("Itemized Sign Recognition Audit Log", h2_style))

    table_data = [
        [
            Paragraph("<b>#</b>", cell_bold),
            Paragraph("<b>Traffic Sign Label</b>", cell_bold),
            Paragraph("<b>Confidence</b>", cell_bold),
            Paragraph("<b>Timestamp (UTC)</b>", cell_bold),
            Paragraph("<b>Safety Classification & Advisory</b>", cell_bold)
        ]
    ]

    for i, d in enumerate(detections, start=1):
        conf_percent = f"{d.confidence * 100:.1f}%"
        # Determine safety note based on label
        label_lower = (d.label or "").lower()
        if "stop" in label_lower:
            advisory = "Mandatory Stop - Full vehicle halt required."
        elif "speed" in label_lower:
            advisory = "Regulatory - Downshift and maintain restricted speed."
        elif "yield" in label_lower or "give way" in label_lower:
            advisory = "Priority - Slow down & yield to crossing vehicles."
        elif "pedestrian" in label_lower:
            advisory = "Caution - Yield right of way to crossing pedestrians."
        elif "school" in label_lower:
            advisory = "High Alert - Reduce speed to 25 km/h near school zone."
        elif "entry" in label_lower:
            advisory = "Prohibitory - Do not enter roadway."
        else:
            advisory = "Follow roadway regulatory instructions."

        table_data.append([
            Paragraph(str(i), cell_style),
            Paragraph(f"<b>{d.label}</b>", cell_bold),
            Paragraph(conf_percent, cell_style),
            Paragraph(d.timestamp.strftime("%Y-%m-%d %H:%M"), cell_style),
            Paragraph(advisory, cell_style)
        ])

    if len(table_data) == 1:
        table_data.append([Paragraph("1", cell_style), Paragraph("No detections recorded yet.", cell_style), "-", "-", "-"])

    records_table = Table(table_data, colWidths=[24, 130, 60, 95, 211])
    records_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(records_table)

    story.append(Spacer(1, 20))
    footer_text = (
        "<i>This report is an automated diagnostic output generated by the AI Vision-Based Traffic Sign Recognition "
        "System developed at PSN Engineering College, Tirunelveli.</i>"
    )
    story.append(Paragraph(footer_text, sub_style))

    doc.build(story)
    return filepath


def generate_single_detection_pdf(detection, username):
    """
    Generates a single inspection certificate PDF with embedded annotated image.
    """
    filename = f"inspection_{detection.id}_{int(datetime.utcnow().timestamp())}.pdf"
    filepath = os.path.join(REPORT_DIR, filename)

    doc = SimpleDocTemplate(
        filepath,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CertTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0f2744"),
        alignment=1
    )

    sub_style = ParagraphStyle(
        "CertSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#475569"),
        alignment=1
    )

    cell_style = ParagraphStyle(
        "CertCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b")
    )

    story = []
    story.append(Paragraph("PSN ENGINEERING COLLEGE, TIRUNELVELI", title_style))
    story.append(Paragraph("Department of Computer Science and Engineering - Project Lab", sub_style))
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2563eb"), spaceAfter=10))

    story.append(Paragraph("<b>OFFICIAL TRAFFIC SIGN RECOGNITION AUDIT SHEET</b>", title_style))
    story.append(Spacer(1, 10))

    base_dir = os.path.dirname(__file__)
    img_disk_path = os.path.join(base_dir, "static", detection.image_path)

    image_element = None
    if os.path.exists(img_disk_path):
        try:
            image_element = RLImage(img_disk_path, width=220, height=220)
        except Exception:
            image_element = Paragraph("[Annotated Image Preview]", cell_style)
    else:
        image_element = Paragraph("[Image not found on server]", cell_style)

    details_text = (
        f"<b>Detection Record ID:</b> #{detection.id}<br/><br/>"
        f"<b>Detected Traffic Sign:</b> {detection.label}<br/><br/>"
        f"<b>Model Confidence:</b> {detection.confidence * 100:.1f}%<br/><br/>"
        f"<b>Recorded By User:</b> {username}<br/><br/>"
        f"<b>Detection Timestamp:</b> {detection.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}<br/><br/>"
        f"<b>System Status:</b> Verified by OpenCV + HUD Geometry Engine<br/>"
    )

    layout_table = Table([[image_element, Paragraph(details_text, cell_style)]], colWidths=[240, 280])
    layout_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, 0), "CENTER"),
    ]))
    story.append(layout_table)
    story.append(Spacer(1, 16))

    doc.build(story)
    return filepath
