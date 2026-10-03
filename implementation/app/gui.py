# author: Petra Oravová
# file: gui.py
# description: This script implements a PyQt5-based GUI for cow identification and verification using deep learning
#              models. It provides functionality to load images, run inference or verification, visualize results,
#              and adjust settings such as the model or threshold. The GUI connects to a PostgreSQL database via
#              the import_database utility.
# used documentation and tutorials:
# - https://doc.qt.io/qtforpython/
# - https://pillow.readthedocs.io/en/stable/
# - https://learnopencv.com/annotating-images-using-opencv/#draw-rectangle
# - https://qtawesome.readthedocs.io/en/latest/icon_browser.html

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox,
    QTabWidget, QDialog, QFormLayout, QCheckBox,
    QToolButton, QSizePolicy, QDoubleSpinBox
)
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import Qt, QSize
from PIL import Image, ImageOps
import qtawesome as qta
import cv2
from db.backup import import_database
from identification.infer import infer, verify

# Application-wide stylesheet
APP_STYLE = """
QWidget { background: #f9f9f9; font-family: 'Segoe UI', sans-serif; }
QTabWidget::pane { border: none; }
QTabBar::tab { background: #e3f2fd; border-radius: 8px; padding: 12px; font-size: 18px; margin: 2px; }
QTabBar::tab:selected { background: #2196F3; color: white; }
QPushButton { background: #2196F3; color: white; border: none; border-radius: 8px; padding: 10px 20px; font-size: 16px; 
min-width: 400px;}
QPushButton:hover { background: #1976D2; }
QToolButton { background: transparent; border: none; }
QToolButton:hover { background: #e0e0e0; }
QLabel { color: #333; }
QComboBox QAbstractItemView {selection-color: black;}
"""


class SettingsDialog(QDialog):
    """
      Dialog for configuring model settings, distance display, and identification threshold.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedSize(300, 200)
        layout = QFormLayout()

        # Model selection combo box
        self.model_combo = QComboBox()
        self.model_combo.addItem("MobileNet_V3_Large", ("mobilenet", "models/mobilenet.pth"))
        self.model_combo.addItem("ResNet-50", ("resnet", "models/resnet.pth"))
        self.model_combo.addItem("EfficientNet_B1", ("efficientnet", "models/efficientnet.pth"))

        # Threshold input spinbox
        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(0.0, 2.0)
        self.threshold_input.setSingleStep(0.1)
        self.threshold_input.setDecimals(2)
        self.threshold_input.setValue(0.5)  # Default value

        # Load database
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        self.on_model_changed()

        # Distance checkbox
        self.show_distance = QCheckBox("Show Distance")

        # Add widgets to layout
        layout.addRow("Model:", self.model_combo)
        layout.addRow(self.show_distance)
        layout.addRow("Threshold:", self.threshold_input)
        self.setLayout(layout)

    def on_model_changed(self):
        """
        Called when the selected model changes. Automatically imports the relevant database.
        """
        model_key, model_path = self.model_combo.currentData()
        input_filename = f"{model_key}.dump"
        import_database(input_filename)


def _load_pixmap_from_pil(pil_image):
    """
    Converts a PIL image to QPixmap for GUI display.

    Args:
        pil_image (PIL.Image): The PIL image to convert.

    Returns:
        QPixmap: The resulting QPixmap suitable for display in Qt GUI.
    """
    img_data = pil_image.tobytes("raw", "RGB")
    qimg = QImage(img_data, pil_image.width, pil_image.height, 3 * pil_image.width, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))


def _load_pixmap(path):
    """
    Loads an image from a file and converts it to a QPixmap.

    Args:
        path (str): Path to the image file to load.

    Returns:
        QPixmap: The converted QPixmap image for GUI display.
    """
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    data = img.tobytes("raw", "RGB")
    qimg = QImage(data, img.width, img.height, 3 * img.width, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)

    return pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class CowApp(QWidget):
    """
    Main application window for cow identification and verification.
    """
    def __init__(self):
        super().__init__()
        self.image_path2 = None
        self.image_path1 = None
        self.verif_prev2 = None
        self.verif_prev1 = None
        self.verif_result = None
        self.ident_result = None
        self.ident_preview = None
        self.setWindowTitle("Cow Identification & Verification")
        self.setGeometry(300, 150, 1000, 500)
        self.setStyleSheet(APP_STYLE)

        # Settings dialog and button
        self.settings = SettingsDialog(self)
        self.settings_button = QToolButton()
        icon = qta.icon('fa5s.cog', color='grey')
        self.settings_button.setIcon(icon)
        self.settings_button.setIconSize(QSize(36, 36))
        self.settings_button.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.settings_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.settings_button.clicked.connect(self.settings.exec_)

        # Tabs: Identify and Verify
        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tabs.tabBar().setExpanding(True)
        self.tabs.tabBar().setFont(QFont('Segoe UI', 18, QFont.Bold))
        self.update_tab_widths()

        self.ident_tab = self.build_identify_tab()
        self.verif_tab = self.build_verify_tab()
        self.tabs.addTab(self.ident_tab, "Identify")
        self.tabs.addTab(self.verif_tab, "Verify")

        # Layouts
        header = QHBoxLayout()
        header.addWidget(self.tabs, 1)
        header.addWidget(self.settings_button, 0, Qt.AlignTop)

        main = QVBoxLayout(self)
        main.addLayout(header)
        self.setLayout(main)

    def resizeEvent(self, event):
        """Handles window resizing by updating tab widths."""
        super().resizeEvent(event)
        self.update_tab_widths()

    def update_tab_widths(self):
        """Dynamically resizes tabs to fill available width."""
        bar = self.tabs.tabBar()
        count = bar.count()
        if count:
            w = self.tabs.width() // count
            bar.setStyleSheet(f"QTabBar::tab {{ width: {w - 30}px; }}")

    def build_identify_tab(self):
        """
        Builds the "Identify" tab UI.

        Returns:
            QWidget: The constructed QWidget representing the Identify tab.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)

        btn_load = QPushButton("Load Image")
        btn_load.clicked.connect(self.load_image1)
        layout.addWidget(btn_load, alignment=Qt.AlignHCenter)

        self.ident_preview = QLabel()
        self.ident_preview.setAlignment(Qt.AlignCenter)
        self.ident_preview.setFixedSize(400, 400)
        self.ident_preview.setStyleSheet("border:2px solid #ccc; border-radius:8px; background:#fff;")
        layout.addWidget(self.ident_preview, alignment=Qt.AlignCenter)

        btn_ident = QPushButton("Identify")
        btn_ident.clicked.connect(self.run_identify)
        layout.addWidget(btn_ident, alignment=Qt.AlignCenter)

        self.ident_result = QLabel("No image loaded.")
        self.ident_result.setFont(QFont('Segoe UI', 32, QFont.Bold))
        self.ident_result.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.ident_result)

        return tab

    def build_verify_tab(self):
        """
        Builds the "Verify" tab UI.

        Returns:
            QWidget: The constructed QWidget representing the Verify tab.
        """
        tab = QWidget()
        layout = QVBoxLayout(tab)

        load_layout = QHBoxLayout()
        load_layout.setSpacing(20)

        btn_a = QPushButton("Load Image A")
        btn_a.clicked.connect(self.load_image1)
        btn_b = QPushButton("Load Image B")
        btn_b.clicked.connect(self.load_image2)

        load_layout.addWidget(btn_a)
        load_layout.addWidget(btn_b)
        layout.addLayout(load_layout)

        prev_layout = QHBoxLayout()
        prev_layout.setSpacing(20)

        self.verif_prev1 = QLabel()
        self.verif_prev1.setAlignment(Qt.AlignCenter)
        self.verif_prev1.setFixedSize(400, 400)
        self.verif_prev1.setStyleSheet("border:2px solid #ccc; border-radius:8px; background:#fff;")

        self.verif_prev2 = QLabel()
        self.verif_prev2.setAlignment(Qt.AlignCenter)
        self.verif_prev2.setFixedSize(400, 400)
        self.verif_prev2.setStyleSheet("border:2px solid #ccc; border-radius:8px; background:#fff;")

        prev_layout.addWidget(self.verif_prev1)
        prev_layout.addWidget(self.verif_prev2)
        layout.addLayout(prev_layout)

        btn_verify = QPushButton("Verify")
        btn_verify.clicked.connect(self.run_verify)
        layout.addWidget(btn_verify, alignment=Qt.AlignCenter)

        self.verif_result = QLabel("No images loaded.")
        self.verif_result.setFont(QFont('Segoe UI', 32, QFont.Bold))
        self.verif_result.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.verif_result)

        return tab

    def load_image1(self):
        """
        Loads the first image from disk and updates the corresponding preview.
        """
        file, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.bmp)"
        )
        if not file:
            return
        self.image_path1 = file
        pix = _load_pixmap(file)
        scaled_pix = pix.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        if self.tabs.currentIndex() == 0:
            self.ident_preview.setPixmap(scaled_pix)
            self.ident_preview.setStyleSheet("background: #f9f9f9;")
        else:
            self.verif_prev1.setPixmap(scaled_pix)
            self.verif_prev1.setStyleSheet("background: #f9f9f9;")

    def load_image2(self):
        """
        Loads the second image for verification and updates the preview.
        """
        file, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.bmp)"
        )
        if not file:
            return
        self.image_path2 = file
        pix = _load_pixmap(file)
        scaled_pix = pix.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.verif_prev2.setPixmap(scaled_pix)
        self.verif_prev2.setStyleSheet("background: #f9f9f9;")

    def run_identify(self):
        """
        Runs cow identification on the loaded image and updates the UI with results.
        """
        if not hasattr(self, 'image_path1'):
            self.ident_result.setText("No image to identify.")
        else:
            model, path = self.settings.model_combo.currentData()
            threshold = self.settings.threshold_input.value()
            result, bbox = infer(self.image_path1, model_name=model, model_path=path, threshold=threshold)

            if result == "No cow detected":
                self.ident_result.setText(result)
            else:
                cid, dist = result
                if cid is None:
                    text = f"Cow not identified\nDistance to nearest: {dist:.3f}" \
                        if self.settings.show_distance.isChecked() else f"ID: {cid}"
                else:
                    text = f"ID: {cid}\nDist: {dist:.3f}" if self.settings.show_distance.isChecked() else f"ID: {cid}"
                self.ident_result.setText(text)

            # Draw bounding box
            if bbox:
                x1, y1, x2, y2 = bbox
                img = cv2.imread(self.image_path1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 20)

                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img_qpixmap = _load_pixmap_from_pil(img_pil)

                self.ident_preview.setPixmap(img_qpixmap)

    def run_verify(self):
        """
        Runs cow verification between two loaded images and updates the UI with results.
        """
        if not hasattr(self, 'image_path1') or not hasattr(self, 'image_path2'):
            self.verif_result.setText("Load both images.")
        else:
            model, path = self.settings.model_combo.currentData()
            threshold = self.settings.threshold_input.value()

            same, dist, bbox1, bbox2 = verify(self.image_path1, self.image_path2, model_name=model, model_path=path,
                                              threshold=threshold)

            if same == "No cow detected in one or both images.":
                self.verif_result.setText(same)
            else:
                text = ("SAME COW" if same else "DIFFERENT COW") + (
                    f"\nDist: {dist:.3f}" if self.settings.show_distance.isChecked() else "")
                self.verif_result.setText(text)

                # Draw bounding boxes
                img1 = cv2.imread(self.image_path1)
                img2 = cv2.imread(self.image_path2)

                if bbox1:
                    x1, y1, x2, y2 = bbox1
                    cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 20)

                if bbox2:
                    x1, y1, x2, y2 = bbox2
                    cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 20)

                img_pil1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
                img_pil2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

                img_qpixmap1 = _load_pixmap_from_pil(img_pil1)
                img_qpixmap2 = _load_pixmap_from_pil(img_pil2)

                self.verif_prev1.setPixmap(img_qpixmap1)
                self.verif_prev2.setPixmap(img_qpixmap2)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    win = CowApp()
    win.show()
    sys.exit(app.exec_())
