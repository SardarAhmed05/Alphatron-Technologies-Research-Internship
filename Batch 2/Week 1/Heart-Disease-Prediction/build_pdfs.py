import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas


def generate_flowchart_pdf():
    """
    Generates a clean, professional vector flowchart PDF based on the README architecture.
    """
    pdf_path = "flowchart_Task1_Sardar_Ahmed_Heart_disease_prediction.pdf"
    fig, ax = plt.subplots(figsize=(10, 14), dpi=300)
    ax.axis("off")

    # Title & Subtitle
    plt.text(5, 13.5, "Heart Disease Machine Learning Pipeline Flowchart",
             ha="center", va="center", fontsize=15, fontweight="bold", color="#1a252f")
    plt.text(5, 13.1, "Task 1: Clean OOP Architecture, Data Leakage Prevention & Model Tuning\nAuthor: Sardar Ahmed | Alphatron Technologies",
             ha="center", va="center", fontsize=9.5, color="#555555", style="italic")

    # Helper function to draw rounded boxes
    def draw_box(x, y, w, h, text, header, bg_color, border_color):
        box = patches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.15,rounding_size=0.15",
            facecolor=bg_color, edgecolor=border_color, linewidth=1.5, zorder=2
        )
        ax.add_patch(box)
        ax.text(x, y + h*0.22, header, ha="center", va="center", fontsize=10, fontweight="bold", color="#1a252f", zorder=3)
        ax.text(x, y - h*0.12, text, ha="center", va="center", fontsize=8.5, color="#2c3e50", zorder=3)

    # Helper function to draw arrows
    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.8, mutation_scale=14),
            zorder=1
        )
        if label:
            ax.text((x1+x2)/2 + 0.15, (y1+y2)/2, label, fontsize=8, fontweight="bold", color="#2980b9", va="center")

    # 1. Raw Dataset Node
    draw_box(5, 12.0, 7.5, 1.0, "UCI Heart Disease Cohort (Cleveland, Hungarian, Swiss, VA)\n920 Records | 16 Demographic & Clinical Attributes", "1. Raw Dataset Ingestion", "#ebf5fb", "#2980b9")

    # Arrow 1 -> 2
    draw_arrow(5, 11.5, 5, 10.7)

    # 2. DataLoader Node
    draw_box(5, 10.1, 7.5, 1.1, "• Replaces impossible 0s in [chol, trestbps] & negative oldpeak with NaN\n• Drops high-missing/index cols: ['id', 'ca' (66%), 'thal' (53%), 'slope' (34%)]\n• Separates Feature Matrix (X: 920x11) and Target Label (y: 920)", "2. DataLoader (Sanitization & Cleaning)", "#fef9e7", "#f39c12")

    # Arrow 2 -> 3
    draw_arrow(5, 9.55, 5, 8.85)

    # 3. Split Node
    draw_box(5, 8.35, 7.5, 0.9, "Stratified 80/20 Partition based on target class distribution\nTrain Set: 736 samples (80%)  |  Test Set: 184 samples (20%)", "3. Stratified Train / Test Split", "#eafaf1", "#27ae60")

    # Split arrows
    draw_arrow(3.2, 7.9, 3.2, 6.9, label="Train Set (80%)")
    draw_arrow(6.8, 7.9, 6.8, 4.0, label="Unseen Test Set (20%)")

    # 4. DataPreprocessor Pipeline Node
    draw_box(3.2, 6.2, 4.4, 1.3, "Encapsulated in scikit-learn ColumnTransformer:\n• Numeric: SimpleImputer(median) + StandardScaler\n• Categorical: SimpleImputer(most_frequent) + OneHotEncoder\n*Fitted ONLY on training data to eliminate Data Leakage*", "4. DataPreprocessor", "#f4ecf7", "#8e44ad")

    # Arrow 4 -> 5
    draw_arrow(3.2, 5.55, 3.2, 4.75)

    # 5. Model Training & CV Node
    draw_box(3.2, 4.0, 4.4, 1.4, "• Model Pipeline: Preprocessor + XGBClassifier\n• Stratified 5-Fold Cross-Validation (StratifiedKFold)\n• GridSearchCV Hyperparameter Tuning (n_estimators, max_depth,\n  learning_rate, subsample, colsample_bytree)\n• Baseline Comparison with RandomForestClassifier", "5. ModelTrainer & 5-Fold GridCV", "#fbeee6", "#d35400")

    # Merge arrow to Evaluator
    draw_arrow(3.2, 3.3, 4.2, 2.5)
    draw_arrow(6.8, 3.55, 5.8, 2.5)

    # 6. ModelEvaluator Node
    draw_box(5, 2.0, 7.5, 1.1, "• Test Accuracy, Macro F1, Weighted F1, Classification Report\n• Confusion Matrix Visualization\n• Multiclass (5-Class Severity: 58.7%) & Binary Benchmark (83.15%)", "6. ModelEvaluator (Testing & Diagnostics)", "#e8f8f5", "#16a085")

    # Arrow 6 -> 7
    draw_arrow(5, 1.45, 5, 0.95)

    # 7. Model Export Node
    draw_box(5, 0.55, 7.5, 0.7, "Pipeline serialized to disk via joblib -> 'xgboost_model.pkl'", "7. Artifact Serialization", "#eaecee", "#7f8c8d")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)

    plt.tight_layout()
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Generated updated flowchart PDF: {pdf_path}")


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and add page numbers and header to ReportLab PDF.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#7f8c8d"))

        # Running Header
        self.drawString(54, 11 * 72 - 36, "Sardar Ahmed • AI Engineering Intern • Alphatron Technologies")
        self.drawRightString(8.5 * 72 - 54, 11 * 72 - 36, "Heart Disease ML Research Report")
        self.setStrokeColor(colors.HexColor("#bdc3c7"))
        self.setLineWidth(0.5)
        self.line(54, 11 * 72 - 42, 8.5 * 72 - 54, 11 * 72 - 42)

        # Running Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 54, 36, page_text)
        self.drawString(54, 36, "Confidential - Research Internship Project")
        self.line(54, 46, 8.5 * 72 - 54, 46)
        self.restoreState()


def generate_report_pdf():
    """
    Generates a professional, comprehensive multi-page research report PDF.
    """
    pdf_path = "Report.pdf"
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1b2631"),
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#566573"),
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1f618d"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=4
    )

    story = []

    # Title Banner
    story.append(Paragraph("Heart Disease Severity Prediction & Diagnostic Screening", title_style))
    story.append(Paragraph("A Clean Object-Oriented Machine Learning Pipeline with In-Depth Exploratory Data Analysis<br/><b>Author:</b> Sardar Ahmed | <b>Affiliation:</b> Alphatron Technologies Research Internship (Week 1)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1f618d"), spaceAfter=10))

    # 1. Executive Summary & Objective
    story.append(Paragraph("1. Executive Summary & Objectives", h1_style))
    story.append(Paragraph(
        "Coronary artery disease remains the leading cause of mortality worldwide. The objective of this project was to design, implement, and validate an end-to-end Machine Learning pipeline to predict disease severity and clinical presence using the multicenter UCI Heart Disease cohort. The project emphasizes <b>rigorous exploratory data analysis (EDA)</b>, <b>data leakage elimination</b> via scikit-learn ColumnTransformers, and a clean, maintainable <b>Object-Oriented Programming (OOP)</b> architecture.",
        body_style
    ))

    # 2. Dataset Overview
    story.append(Paragraph("2. Dataset & Multicenter Clinical Cohort", h1_style))
    story.append(Paragraph(
        "The UCI Heart Disease dataset aggregates <b>920 patient records</b> across four geographic clinical centers: Cleveland Clinic (USA), Hungarian Institute of Cardiology (Hungary), University Hospital Zurich (Switzerland), and V.A. Medical Center Long Beach (USA). The dataset contains 16 features encompassing patient demographics, clinical symptoms, resting vitals, electrocardiographic (ECG) metrics, and exercise stress test parameters.",
        body_style
    ))

    # 3. Exploratory Data Analysis & Empirical Preprocessing Decisions
    story.append(Paragraph("3. Key Insights from Exploratory Data Analysis (EDA)", h1_style))
    story.append(Paragraph("Prior to modeling, a thorough visual and statistical EDA led to the following critical data decisions:", body_style))
    
    story.append(Paragraph("• <b>Handling Extreme Missingness:</b> Features <code>ca</code> (66.4% missing), <code>thal</code> (52.8% missing), and <code>slope</code> (33.6% missing) were removed because cross-center protocol discrepancies caused systematic data absence. Imputing over 50% of missing values would introduce heavy bias.", bullet_style))
    story.append(Paragraph("• <b>Biological Anomaly Correction:</b> 172 records in the Hungarian and Swiss subsets contained <code>chol = 0 mg/dl</code> and 1 record had <code>trestbps = 0 mm Hg</code> (physiologically impossible). These invalid zeroes were converted to <code>NaN</code> and imputed using median/MICE methods rather than discarding 19% of the cohort.", bullet_style))
    story.append(Paragraph("• <b>Chest Pain Paradox:</b> Over 78% of patients presenting with <i>asymptomatic chest pain</i> tested positive for coronary disease due to referral bias. Categorical variables (<code>cp</code>, <code>dataset</code>, <code>restecg</code>) were encoded using One-Hot Encoding (drop first).", bullet_style))
    story.append(Paragraph("• <b>Correlation Drivers:</b> Exercise-induced ST depression (<code>oldpeak</code>, r = +0.43) and maximum heart rate achieved (<code>thalch</code>, r = -0.39) emerged as the strongest physiological discriminators.", bullet_style))

    # 4. OOP Pipeline Architecture & Data Leakage Fix
    story.append(Paragraph("4. Object-Oriented Pipeline Architecture & Leakage Prevention", h1_style))
    story.append(Paragraph(
        "The codebase was refactored from a monolithic script into five decoupled, single-responsibility classes: "
        "<b>DataLoader</b> (raw ingestion & anomaly replacement), <b>DataPreprocessor</b> (ColumnTransformer encapsulation), "
        "<b>ModelTrainer</b> (Stratified 5-fold GridSearchCV & pipeline assembly), <b>ModelEvaluator</b> (multiclass & binary metrics), "
        "and <b>HeartDiseasePipeline</b> (high-level orchestrator).",
        body_style
    ))
    story.append(Paragraph(
        "<b>Data Leakage Elimination:</b> In standard scripts, fitting imputers or scalers globally on the dataset before splitting causes test data information to leak into training folds. Here, preprocessing is strictly wrapped inside a <code>Pipeline</code>, ensuring statistics are computed exclusively on training folds during cross-validation.",
        body_style
    ))

    # Page Break for Results & Conclusion
    story.append(PageBreak())

    # 5. Experimental Results & Benchmarks
    story.append(Paragraph("5. Experimental Results & Performance Benchmarks", h1_style))
    story.append(Paragraph(
        "The pipeline was evaluated on an 80/20 Stratified train/test split. We benchmarked a tuned <b>XGBoost Classifier</b> against a <b>Random Forest Baseline</b> across both the 5-class severity task and the standard binary clinical screening task (diameter narrowing > 50%).",
        body_style
    ))

    # Table of Results
    table_data = [
        ["Classification Task", "Model Architecture", "Test Accuracy", "Macro F1", "Weighted F1"],
        ["5-Class Severity (0-4)", "Random Forest (Baseline)", "57.61%", "0.4454", "0.6035"],
        ["5-Class Severity (0-4)", "Tuned XGBoost Pipeline", "58.70%", "0.3521", "0.5719"],
        ["Binary Screening (0 vs 1+)", "Binary XGBoost Pipeline", "83.15%", "0.8290", "0.8310"]
    ]

    t = Table(table_data, colWidths=[120, 130, 75, 65, 75])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f618d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8f9fa"), colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # Discussion on Class Imbalance
    story.append(Paragraph("6. Analysis of Class Imbalance & Clinical Implications", h1_style))
    story.append(Paragraph(
        "A key finding from this research is the pronounced impact of target class imbalance in multiclass staging. Class 4 (severe disease) constitutes only <b>3.0% (28 samples)</b> of the entire dataset, compared to <b>44.7% (411 samples)</b> in Class 0. While 5-class staging achieves ~59% accuracy due to sparse minority data, the model achieves <b>83.15% accuracy</b> on the binary screening task, confirming that the learned feature representations reliably identify cardiac risk.",
        body_style
    ))

    # 7. Conclusions & Recommendations
    story.append(Paragraph("7. Summary of Deliverables & Conclusion", h1_style))
    story.append(Paragraph(
        "This project successfully delivers an end-to-end, scientifically validated machine learning pipeline. Deliverables include:",
        body_style
    ))
    story.append(Paragraph("1. <b>Clean OOP Python Engine:</b> <code>heart_disease_prediction.py</code> with full pipeline orchestration.", bullet_style))
    story.append(Paragraph("2. <b>Visual EDA Suite:</b> 9 high-resolution charts in <code>eda_charts/</code> and an interpreted <code>EDA_Report.md</code>.", bullet_style))
    story.append(Paragraph("3. <b>Interactive Notebook:</b> <code>heart_disease_prediction.ipynb</code> for step-by-step exploration.", bullet_style))
    story.append(Paragraph("4. <b>Exported Production Artifact:</b> Serialized <code>xgboost_model.pkl</code> ready for API serving.", bullet_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated updated research report PDF: {pdf_path}")


if __name__ == "__main__":
    generate_flowchart_pdf()
    generate_report_pdf()
