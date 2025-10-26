import sys
import os
import hashlib
import cv2
import numpy as np
import json
import re
from scipy.spatial.distance import hamming

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
from iris.feature_extraction import encode_iris

def compute_template(image_path: str) -> np.ndarray:
    """
    Compute iris code using the existing encode_iris function.
    Returns binary iris code as numpy array.
    """
    try:
        _, normalized_path, _ = run_full_pipeline(image_path, out_root="outputs")
        norm = cv2.imread(normalized_path, cv2.IMREAD_GRAYSCALE)
        if norm is None:
            raise RuntimeError("Cannot read normalized image.")
        iris_code = encode_iris(norm)
        return iris_code
    except Exception as e:
        raise RuntimeError(f"Failed to compute iris code: {str(e)}")

def compare_templates(t1: np.ndarray, t2: np.ndarray) -> Tuple[float, float]:
    """
    Compare two iris codes using Hamming distance.
    Returns (hamming_distance, similarity_percentage).
    """
    return compute_hamming_distance(t1, t2)

def search_dataset_for_best_match(query_template, codes_file="iris_codes/iris_codes.json"):
    if not os.path.exists(codes_file):
        print("❌ No saved iris codes found. Run enrollment first.")
        return None

    with open(codes_file, "r") as f:
        saved = json.load(f)

    best_subject = None
    best_hamming = 1.0
    best_similarity = 0.0

    for subject_id, template_list in saved.items():
        stored_template = np.array(template_list, dtype=np.uint8)

        hamming, similarity = compare_templates(query_template, stored_template)

        if hamming < best_hamming:  # keep best match
            best_hamming = hamming
            best_similarity = similarity
            best_subject = subject_id

    if best_subject is None:
        return None

    #  Left/Right labels "Unknown"
    return best_subject, "Unknown", best_hamming, best_similarity


def compute_hamming_distance(code1: np.ndarray, code2: np.ndarray) -> Tuple[float, float]:
    """
    Compute Hamming distance and similarity between two iris codes.
    Handles rotation by trying small shifts.
    Returns (hamming_distance, similarity_percentage).
    """
    # if code1.shape != code2.shape:
    #     raise ValueError("Iris codes must have the same length")

    # length = len(code1)
    # min_hamming = 1.0
    # max_shift = 8  # ±8 pixels for rotation compensation

    # for shift in range(-max_shift, max_shift + 1):
    #     shifted_code1 = np.roll(code1, shift)
    #     diffs = np.sum(shifted_code1 != code2)
    #     hamming = diffs / length if length > 0 else 1.0
    #     min_hamming = min(min_hamming, hamming)
    min_hamming =  hamming(code1, code2)
    similarity = max(0.0, 100.0 * (1.0 - min_hamming))
    return round(min_hamming, 4), round(similarity, 2)


#GUI
class IrisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Iris Recognition — GUI")
        self.setMinimumSize(920, 600)
        self._dataset_root = "CASIA-Iris-Thousand"  # default dataset path 
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
        self.verify_subject_input.setPlaceholderText("Enter subject name exactly (e.g. person_001)")
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

        # self.ident_matched_image = QLabel()
        # self.ident_matched_image.setFixedSize(380, 160)
        # self.ident_matched_image.setFrameShape(QFrame.Box)
        # self.ident_matched_image.setAlignment(Qt.AlignCenter)
        # right_layout.addWidget(self.ident_matched_image)

        stats_box = QGroupBox("Match Info")
        stats_layout = QFormLayout()
        self.ident_match_subject_label = QLabel("-")
        self.ident_match_side_label = QLabel("-")
        self.ident_match_hamming_label = QLabel("-")
        self.ident_match_similarity_label = QLabel("-")
        stats_layout.addRow("Subject:", self.ident_match_subject_label)
        #stats_layout.addRow("Side (Left/Right):", self.ident_match_side_label)
        stats_layout.addRow("Hamming:", self.ident_match_hamming_label)
        stats_layout.addRow("Similarity %:", self.ident_match_similarity_label)
        stats_box.setLayout(stats_layout)
        right_layout.addWidget(stats_box)

        right_group.setLayout(right_layout)
        right_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        main_layout.addWidget(right_group, 0)

        tab.setLayout(main_layout)
        return tab

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
        self.iris_codes="iris_codes"
        if not subject:
            self.verify_result_box.setPlainText("Please enter a subject ID (e.g., person_000).")
            return
        if not img_path or not os.path.exists(img_path):
            self.verify_result_box.setPlainText("Please upload a valid .jpg image file.")
            return

        try:
            # Load iris code database
            try:
                with open(os.path.join(self.iris_codes,"iris_codes.json"), "r") as f:
                    db = json.load(f)
            except FileNotFoundError:
                self.verify_result_box.setPlainText("Iris code database not found. Please enroll subjects first.")
                return

            # Check if subject exists in database
            if subject not in db:
                self.verify_result_box.setPlainText(
                    f"Subject '{subject}' not found in iris code database '{self.iris_codes}'."
                )
                self.verify_hamming_label.setText("-")
                self.verify_similarity_label.setText("-")
                self.verify_matched_subject_label.setText("Not found")
                return

            # Compute query iris code
            t_query = compute_template(img_path)
            t_ref = np.array(db[subject], dtype=np.uint8)
            hamming, similarity = compare_templates(t_query, t_ref)

            # Threshold for verification
            threshold = 0.342  
            verified = hamming <= threshold
            matched_subject = subject if verified else "No Match"

            # Update UI
            self.verify_hamming_label.setText(f"{hamming:.4f}")
            self.verify_similarity_label.setText(f"{similarity:.2f} %")
            self.verify_matched_subject_label.setText(matched_subject)
            self.verify_result_box.setPlainText(
                f"Verification against subject: {subject}\n"
                f"Hamming distance: {hamming:.4f}\n"
                f"Similarity: {similarity:.2f} %\n"
                f"Result: {'Verified' if verified else 'Not Verified'}"
            )

            # Display matched image from dataset if verified
            if verified:
                subject_num = subject.replace("person_", "")  # Convert person_000 to 000
                subject_dir = Path(self._dataset_root) / f"S1{subject_num}"
                if subject_dir.exists():
                    for side in ("Left", "Right"):
                        side_dir = subject_dir / side
                        if side_dir.exists():
                            for p in side_dir.glob("*.jpg"):
                                self._display_pixmap(self.verify_image_label, str(p))
                                break
                            break
        except Exception as e:
            self.verify_result_box.setPlainText(f"Verification failed: {str(e)}")
            self.verify_hamming_label.setText("-")
            self.verify_similarity_label.setText("-")
            self.verify_matched_subject_label.setText("Error")


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
            self.ident_result_box.setPlainText("Please upload a valid .jpg image file.")
            return

        try:
            # Compute query iris code
            query_t = compute_template(img_path)
            # Search dataset for best match
            best = search_dataset_for_best_match(query_t)
            if best is None:
                self.ident_result_box.setPlainText("No match found in saved iris codes.")
                self._clear_ident_results()
                return

            subject, side, hamming, similarity = best
            matched_img_path = ""
            # Threshold for identification
            threshold = 0.342  
            confidence = "High" if hamming <= threshold else "Low"
            matched_subject = subject if hamming <= threshold else "Unknown"

            # Update UI
            self.ident_match_subject_label.setText(matched_subject)
            self.ident_match_side_label.setText(side)
            self.ident_match_hamming_label.setText(f"{hamming:.4f}")
            self.ident_match_similarity_label.setText(f"{similarity:.2f} %")
            #self._display_pixmap(self.ident_matched_image, matched_img_path)
            self.ident_result_box.setPlainText(
                f"Best match found:\n"
                f"Subject: {matched_subject}\n"
                # f"Side: {side}\n"
                # f"Matched image: {matched_img_path}\n"
                f"Hamming distance: {hamming:.4f}\n"
                f"Similarity: {similarity:.2f} %\n"
                f"Confidence: {confidence}"
            )
        except Exception as e:
            self.ident_result_box.setPlainText(f"Identification failed: {str(e)}")
            self._clear_ident_results()

    def _clear_ident_results(self):
        self.ident_query_image.clear()
        #self.ident_matched_image.clear()
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
