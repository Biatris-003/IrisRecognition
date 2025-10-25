import sys
import os
import hashlib
import cv2
import numpy as np


from pathlib import Path
from iris.pipeline import run_full_pipeline, preprocess_iris_image  # our pipeline


from pathlib import Path
from typing import Tuple, Optional

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QTextEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QTabWidget, QGroupBox, QFormLayout,
    QSizePolicy, QScrollArea, QFrame
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QSize

#PLACEHOLDER BACKEND FUNCTIONS:

# def preprocess_iris_image(image_path: str, show_steps: bool = False):
#     """
#     Preprocess an iris image without edge detection — output a clean, enhanced grayscale iris.
#     Steps:
#       1. Convert to grayscale
#       2. Median blur (noise reduction)
#       3. CLAHE (contrast enhancement)
#       4. Reflection detection & inpainting
#       5. Light intensity normalization
#     """
#     img = cv2.imread(image_path)
#     if img is None:
#         raise FileNotFoundError(image_path)

#     # 1. Grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # 2. Reduce noise
#     blurred = cv2.medianBlur(gray, 5)

#     # 3. Contrast enhancement
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(blurred)

#     # 4. Reflection removal
#     reflection_mask = cv2.inRange(enhanced, 240, 255)
#     cleaned = cv2.inpaint(enhanced, reflection_mask, 3, cv2.INPAINT_TELEA)

#     # 5. Normalize intensity (optional)
#     normalized = cv2.normalize(cleaned, None, 0, 255, cv2.NORM_MINMAX)

#     if show_steps:
#         cv2.imshow("Original", img)
#         cv2.imshow("Enhanced (CLAHE)", enhanced)
#         cv2.imshow("Cleaned (Reflections Removed)", cleaned)
#         cv2.imshow("Normalized Output", normalized)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#     return normalized


# def compute_template(image_path: str) -> str:
#     """Replace this with your actual encoder (e.g., iris code)."""
#     edges = preprocess_iris_image(image_path, show_steps=False)
#     # Flatten binary edge map to bytes
#     flat_bytes = edges.tobytes()
#     h = hashlib.sha1(flat_bytes)  # or use a custom feature extractor later
#     return h.hexdigest()

def compare_templates(t1: str, t2: str) -> Tuple[float, float]:
    """ Replace with your real Hamming distance routine. """
    # Make equal length
    L = min(len(t1), len(t2))
    diffs = sum(1 for a, b in zip(t1[:L], t2[:L]) if a != b)
    hamming = diffs / L if L > 0 else 1.0
    similarity = max(0.0, 100.0 * (1.0 - hamming))
    return round(hamming, 4), round(similarity, 2)

def compute_template(image_path: str) -> str:
    """
    Simple placeholder 'encoder': hash the normalized image bytes so we’re consistent with segmentation.
    It runs the pipeline to guarantee normalized images even if called directly.
    """
    try:
        _, normalized_path, _ = run_full_pipeline(image_path, out_root="outputs")
        arr = cv2.imread(normalized_path, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise RuntimeError("Cannot read normalized image.")
        flat_bytes = arr.tobytes()
    except Exception:
        # fall back to preprocessing only (should be rare)
        pre = preprocess_iris_image(image_path)
        flat_bytes = pre.tobytes()

    h = hashlib.sha1(flat_bytes)
    return h.hexdigest()

def search_dataset_for_best_match(query_template: str, dataset_root: str) -> Optional[Tuple[str, str, str, float, float]]:
    """ Scans dataset_root for structure dataset/<subject>/<Left or Right>/*.png (or jpg).
    For each image found, compute template and compare; return best match info:
      (subject_name, iris_side, image_path, hamming, similarity)
    Returns None if no images found.
    NOTE: This uses placeholder compute_template + compare_templates.
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        return None
    best = None  # tuple defined above
    for subject_dir in sorted(dataset_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        subject = subject_dir.name
        # look for Left/Right or any images directly as fallback
        for side_dir in sorted(subject_dir.iterdir()):
            if side_dir.is_dir():
                iris_side = side_dir.name
                for img_path in sorted(side_dir.iterdir()):
                    if img_path.is_file() and img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                        t = compute_template(str(img_path))
                        h, s = compare_templates(query_template, t)
                        if best is None or h < best[3]:
                            best = (subject, iris_side, str(img_path), h, s)
            else:
                # potentially image directly under subject folder
                if side_dir.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                    t = compute_template(str(side_dir))
                    h, s = compare_templates(query_template, t)
                    if best is None or h < best[3]:
                        best = (subject, "Unknown", str(side_dir), h, s)
    return best

#GUI
class IrisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Iris Recognition — GUI")
        self.setMinimumSize(920, 600)
        self._dataset_root = "dataset"  # default dataset path; user can change by editing code or we can add UI later
        self._output_dir = "outputs"

        self.setStyleSheet(self._qss())
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        header = QLabel("Iris Recognition")
        header.setObjectName("headerLabel")
        header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(header)

        tabs = QTabWidget()
        tabs.addTab(self._build_verification_tab(), "Verification")
        tabs.addTab(self._build_identification_tab(), "Identification")
        tabs.setObjectName("mainTabs")
        layout.addWidget(tabs)

        footer = QLabel("")
        footer.setObjectName("footerLabel")
        layout.addWidget(footer)

        self.setLayout(layout)

#VERIFICATION TAB
    def _build_verification_tab(self):
        tab = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setSpacing(18)

        left_group = QGroupBox("Verification")
        left_layout = QFormLayout()
        left_layout.setLabelAlignment(Qt.AlignLeft)
        left_layout.setFormAlignment(Qt.AlignTop)

        self.verify_subject_input = QLineEdit()
        self.verify_subject_input.setPlaceholderText("Enter subject name exactly (e.g. subject01)")
        left_layout.addRow("Subject name:", self.verify_subject_input)

        self.verify_image_path = QLineEdit()
        self.verify_image_path.setReadOnly(True)
        left_layout.addRow("Image file:", self.verify_image_path)

        btn_row = QHBoxLayout()
        btn_upload = QPushButton("Upload Image")
        btn_upload.clicked.connect(self._on_verify_upload)
        btn_verify = QPushButton("Run Verification")
        btn_verify.clicked.connect(self._on_run_verification)
        btn_row.addWidget(btn_upload)
        btn_row.addWidget(btn_verify)
        left_layout.addRow(btn_row)


        self.verify_result_box = QTextEdit()
        self.verify_result_box.setReadOnly(True)
        self.verify_result_box.setFixedHeight(160)
        left_layout.addRow("Result:", self.verify_result_box)

        left_group.setLayout(left_layout)
        left_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        main_layout.addWidget(left_group, 1)

 
        right_group = QGroupBox("Preview & Stats")
        right_layout = QVBoxLayout()
        self.verify_image_label = QLabel()
        self.verify_image_label.setFixedSize(380, 280)
        self.verify_image_label.setFrameShape(QFrame.Box)
        self.verify_image_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.verify_image_label, alignment=Qt.AlignTop)

        stats_box = QGroupBox("Scores")
        stats_layout = QFormLayout()
        self.verify_hamming_label = QLabel("-")
        self.verify_similarity_label = QLabel("-")
        self.verify_matched_subject_label = QLabel("-")
        stats_layout.addRow("Hamming:", self.verify_hamming_label)
        stats_layout.addRow("Similarity %:", self.verify_similarity_label)
        stats_layout.addRow("Matched Subject:", self.verify_matched_subject_label)
        stats_box.setLayout(stats_layout)
        right_layout.addWidget(stats_box)
        right_group.setLayout(right_layout)
        right_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        main_layout.addWidget(right_group, 0)

        tab.setLayout(main_layout)
        return tab

#IDENTIFICATION TAB
    def _build_identification_tab(self):
        tab = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setSpacing(18)

        left_group = QGroupBox("Identification")
        left_layout = QVBoxLayout()

        control_row = QHBoxLayout()
        self.ident_image_path = QLineEdit()
        self.ident_image_path.setReadOnly(True)
        control_row.addWidget(self.ident_image_path)

        btn_upload = QPushButton("Upload Image")
        btn_upload.clicked.connect(self._on_ident_upload)
        control_row.addWidget(btn_upload)

        left_layout.addLayout(control_row)

        btn_identify = QPushButton("Run Identification")
        btn_identify.clicked.connect(self._on_run_identification)
        left_layout.addWidget(btn_identify)

        self.ident_result_box = QTextEdit()
        self.ident_result_box.setReadOnly(True)
        self.ident_result_box.setFixedHeight(160)
        left_layout.addWidget(self.ident_result_box)

     
        ds_label = QLabel(f"Dataset root: {self._dataset_root}")
        ds_label.setObjectName("mutedLabel")
        left_layout.addWidget(ds_label)

        left_group.setLayout(left_layout)
        left_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        main_layout.addWidget(left_group, 1)

        right_group = QGroupBox("Best Match")
        right_layout = QVBoxLayout()

        self.ident_query_image = QLabel()
        self.ident_query_image.setFixedSize(380, 160)
        self.ident_query_image.setFrameShape(QFrame.Box)
        self.ident_query_image.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.ident_query_image)

        self.ident_matched_image = QLabel()
        self.ident_matched_image.setFixedSize(380, 160)
        self.ident_matched_image.setFrameShape(QFrame.Box)
        self.ident_matched_image.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.ident_matched_image)

        stats_box = QGroupBox("Match Info")
        stats_layout = QFormLayout()
        self.ident_match_subject_label = QLabel("-")
        self.ident_match_side_label = QLabel("-")
        self.ident_match_hamming_label = QLabel("-")
        self.ident_match_similarity_label = QLabel("-")
        stats_layout.addRow("Subject:", self.ident_match_subject_label)
        stats_layout.addRow("Side (Left/Right):", self.ident_match_side_label)
        stats_layout.addRow("Hamming:", self.ident_match_hamming_label)
        stats_layout.addRow("Similarity %:", self.ident_match_similarity_label)
        stats_box.setLayout(stats_layout)
        right_layout.addWidget(stats_box)

        right_group.setLayout(right_layout)
        right_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        main_layout.addWidget(right_group, 0)

        tab.setLayout(main_layout)
        return tab


    # def _on_verify_upload(self):
    #     path, _ = QFileDialog.getOpenFileName(self, "Select iris image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
    #     if path:
    #         self.verify_image_path.setText(path)
    #         edges = preprocess_iris_image(path, show_steps=False)
    #         preview_path = "temp_preprocessed.png"
    #         cv2.imwrite(preview_path, edges)
    #         self._display_pixmap(self.verify_image_label, preview_path)

    def _on_verify_upload(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select iris image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.verify_image_path.setText(path)
            try:
                seg_path, norm_path, _ = run_full_pipeline(path, out_root=self._output_dir)
                # show segmented overlay in the GUI
                self._display_pixmap(self.verify_image_label, seg_path)
                # stash normalized path for later if you want (optional)
                self._verify_last_normalized = norm_path
                self.verify_result_box.setPlainText(
                    f"Saved:\n  Segmented → {seg_path}\n  Normalized → {norm_path}"
                )
            except Exception as e:
                self.verify_result_box.setPlainText(f"Segmentation/normalization failed: {e}")

    def _on_run_verification(self):
        subject = self.verify_subject_input.text().strip()
        img_path = self.verify_image_path.text().strip()
        if not subject:
            self.verify_result_box.setPlainText("Please enter a subject name to verify against.")
            return
        if not img_path or not os.path.exists(img_path):
            self.verify_result_box.setPlainText("Please upload a valid image file.")
            return

        subject_dir = Path(self._dataset_root) / subject
        match_image = None
        if subject_dir.exists():
            for p in subject_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                    match_image = str(p)
                    break

        t_query = compute_template(img_path)
        if match_image:
            t_ref = compute_template(match_image)
            hamming, similarity = compare_templates(t_query, t_ref)
            matched_subject = subject
            matched_image_path = match_image
        else:
            t_ref = None
            hamming, similarity = 1.0, 0.0
            matched_subject = "Not found in dataset"
            matched_image_path = None

        # Update UI
        self.verify_hamming_label.setText(str(hamming))
        self.verify_similarity_label.setText(f"{similarity} %")
        self.verify_matched_subject_label.setText(matched_subject)

        # Show matched image preview if available
        if matched_image_path:
            # show matched image in the right preview as small overlay? We'll show the matched file path in results
            self.verify_result_box.setPlainText(
                f"Verification against subject: {subject}\n"
                f"Matched image used from dataset: {matched_image_path}\n"
                f"Hamming distance: {hamming}\n"
                f"Similarity: {similarity} %\n"
                f"\n(PLACEHOLDER) Replace compute_template/compare_templates with your real backend."
            )
        else:
            self.verify_result_box.setPlainText(
                f"Subject '{subject}' not found under dataset root '{self._dataset_root}'.\n"
                f"Hamming distance: {hamming}\nSimilarity: {similarity} %\n"
                f"\n(PLACEHOLDER) Add subject images to dataset/{subject}/Left or Right/ to enable verification."
            )

    # def _on_ident_upload(self):
    #     path, _ = QFileDialog.getOpenFileName(self, "Select iris image to identify", "", "Images (*.png *.jpg *.jpeg *.bmp)")
    #     if path:
    #         self.ident_image_path.setText(path)
    #         self._display_pixmap(self.ident_query_image, path)

    def _on_ident_upload(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select iris image to identify", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.ident_image_path.setText(path)
            try:
                seg_path, norm_path, _ = run_full_pipeline(path, out_root=self._output_dir)
                # show segmented overlay (query image) on the left preview
                self._display_pixmap(self.ident_query_image, seg_path)
                self._ident_last_normalized = norm_path
                self.ident_result_box.setPlainText(
                    f"Saved:\n  Segmented → {seg_path}\n  Normalized → {norm_path}"
                )
            except Exception as e:
                self.ident_result_box.setPlainText(f"Segmentation/normalization failed: {e}")

    def _on_run_identification(self):
        img_path = self.ident_image_path.text().strip()
        if not img_path or not os.path.exists(img_path):
            self.ident_result_box.setPlainText("Please upload a valid image file.")
            return
        query_t = compute_template(img_path)

        best = search_dataset_for_best_match(query_t, self._dataset_root)
        if best is None:
            self.ident_result_box.setPlainText(f"No images found under dataset root '{self._dataset_root}'.")
            self._clear_ident_results()
            return

        subject, side, matched_img_path, hamming, similarity = best

        # Update UI
        self.ident_result_box.setPlainText(
            f"Best match found:\nSubject: {subject}\nSide: {side}\nMatched image: {matched_img_path}\nHamming: {hamming}\nSimilarity: {similarity} %\n\n(PLACEHOLDER) Replace search_dataset_for_best_match with your real identification logic."
        )
        self.ident_match_subject_label.setText(subject)
        self.ident_match_side_label.setText(side)
        self.ident_match_hamming_label.setText(str(hamming))
        self.ident_match_similarity_label.setText(f"{similarity} %")

        # Show matched image
        self._display_pixmap(self.ident_matched_image, matched_img_path)

    def _clear_ident_results(self):
        self.ident_query_image.clear()
        self.ident_matched_image.clear()
        self.ident_match_subject_label.setText("-")
        self.ident_match_side_label.setText("-")
        self.ident_match_hamming_label.setText("-")
        self.ident_match_similarity_label.setText("-")


    def _display_pixmap(self, label: QLabel, image_path: str):
        try:
            pix = QPixmap(image_path)
            if pix.isNull():
                label.setText("Cannot load image")
                return
            # scale pixmap while keeping aspect ratio
            w, h = label.width(), label.height()
            scaled = pix.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled)
        except Exception as e:
            label.setText("Error loading image")


    def _qss(self) -> str:
        return """
        QWidget {
            background: #f6f7fb;
            font-family: "Segoe UI", Roboto, Arial;
            color: #222;
            font-size: 11pt;
        }
        #headerLabel {
            font-size: 16pt;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 6px;
        }
        QTabWidget::pane {
            border: 0;
            background: transparent;
        }
        QTabBar::tab {
            background: #e6e9f2;
            border-radius: 8px;
            min-width: 140px;
            padding: 10px 14px;
            margin: 4px;
            font-weight: 600;
            color: #374151;
        }
        QTabBar::tab:selected {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #6366f1, stop:1 #4f46e5);
            color: white;
            box-shadow: 0 6px 18px rgba(79,70,229,0.18);
        }
        QGroupBox {
            border: 1px solid rgba(99,102,241,0.06);
            border-radius: 10px;
            padding: 12px;
            background: white;
            font-weight: 600;
        }
        QPushButton {
            background: #4f46e5;
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 8px;
            font-weight: 600;
            min-width: 90px;
        }
        QPushButton:hover {
            background: #5b4df0;
        }
        QPushButton:pressed {
            background: #4236c9;
        }
        QLineEdit, QTextEdit {
            border: 1px solid #e6e9f2;
            padding: 8px;
            border-radius: 8px;
            background: #fbfbff;
        }
        QTextEdit {
            font-family: "Consolas", monospace;
            font-size: 10pt;
        }
        QLabel#mutedLabel {
            color: #6b7280;
            font-size: 9pt;
        }
        QLabel {
            font-size: 11pt;
        }
        #footerLabel {
            color: #6b7280;
            font-size: 9pt;
            margin-top: 6px;
        }
        """

def main():
    app = QApplication(sys.argv)
    window = IrisApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
