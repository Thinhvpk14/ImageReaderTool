"""
ASAP-style TIFF Viewer — 3-panel synchronized viewer for pathology images.

Panels: Image | Mask | Condition
Features:
  - Synchronized pan/zoom across all panels
  - Overlay mask on image with adjustable opacity
  - Thumbnail navigation
  - Multi-resolution support (pyramidal TIFF via OpenSlide or tifffile)
  - Keyboard shortcuts: Ctrl+O (open), F (fit), R (reset), +/- (zoom)
"""

import sys
import os
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QHBoxLayout, QVBoxLayout, QWidget,
    QFileDialog, QLabel, QSlider, QToolBar, QAction, QSplitter,
    QGroupBox, QCheckBox, QStatusBar, QMessageBox, QComboBox,
    QPushButton, QFrame, QSizePolicy,
)
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QPointF, QTimer
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QWheelEvent, QIcon

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


COLORMAPS = {
    "Red":    (255, 0, 0),
    "Green":  (0, 255, 0),
    "Blue":   (0, 0, 255),
    "Yellow": (255, 255, 0),
    "Cyan":   (0, 255, 255),
    "Magenta": (255, 0, 255),
}


# ---------------------------------------------------------------------------
# Slide reader abstraction
# ---------------------------------------------------------------------------
class SlideReader:
    """Unified reader for TIFF / whole-slide images."""

    def __init__(self, path: str):
        self.path = path
        self._slide = None
        self._tiff_data = None
        self._levels = []

        if HAS_OPENSLIDE and self._try_openslide(path):
            self.backend = "openslide"
        elif HAS_TIFFFILE:
            self._load_tifffile(path)
            self.backend = "tifffile"
        else:
            raise RuntimeError(
                "No backend available. Install openslide-python or tifffile."
            )

    def _try_openslide(self, path: str) -> bool:
        try:
            self._slide = openslide.OpenSlide(path)
            self._levels = [
                self._slide.level_dimensions[i]
                for i in range(self._slide.level_count)
            ]
            return True
        except Exception:
            return False

    def _load_tifffile(self, path: str):
        tif = tifffile.TiffFile(path)
        if tif.series:
            series = tif.series[0]
            self._levels = []
            for level in series.levels:
                shape = level.shape
                h, w = shape[0], shape[1]
                self._levels.append((w, h))
            self._tiff_series = series
        else:
            page = tif.pages[0]
            data = page.asarray()
            h, w = data.shape[:2]
            self._levels = [(w, h)]
            self._tiff_data = data
        self._tif = tif

    @property
    def dimensions(self):
        return self._levels[0] if self._levels else (0, 0)

    @property
    def level_count(self):
        return len(self._levels)

    def level_dimensions(self, level: int):
        return self._levels[min(level, len(self._levels) - 1)]

    def read_region(self, location, level, size):
        """Read a region and return as RGBA numpy array."""
        level = min(level, self.level_count - 1)
        w, h = size

        if self.backend == "openslide":
            pil_img = self._slide.read_region(location, level, (w, h))
            return np.array(pil_img)

        if self._tiff_data is not None:
            data = self._tiff_data
        else:
            data = self._tiff_series.levels[level].asarray()

        ds = self._levels[0][0] / self._levels[level][0]
        x = int(location[0] / ds)
        y = int(location[1] / ds)
        ih, iw = data.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)

        region = data[y1:y2, x1:x2]

        if region.ndim == 2:
            region = np.stack([region] * 3 + [np.full(region.shape, 255, dtype=np.uint8)], axis=-1)
        elif region.shape[2] == 3:
            alpha = np.full((*region.shape[:2], 1), 255, dtype=np.uint8)
            region = np.concatenate([region, alpha], axis=-1)

        pad_h, pad_w = h - (y2 - y1), w - (x2 - x1)
        if pad_h > 0 or pad_w > 0:
            region = np.pad(
                region,
                ((0, max(0, pad_h)), (0, max(0, pad_w)), (0, 0)),
                mode="constant", constant_values=0,
            )
        return region[:h, :w]

    def get_thumbnail(self, max_size=(512, 512)):
        if self.backend == "openslide":
            thumb = self._slide.get_thumbnail(max_size)
            return np.array(thumb.convert("RGBA"))

        best = self.level_count - 1
        for i in range(self.level_count):
            w, h = self._levels[i]
            if w <= max_size[0] and h <= max_size[1]:
                best = i
                break
        data = self.read_region((0, 0), best, self._levels[best])

        from PIL import Image
        img = Image.fromarray(data)
        img.thumbnail(max_size, Image.LANCZOS)
        return np.array(img.convert("RGBA"))

    def close(self):
        if self._slide:
            self._slide.close()


def numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
    elif arr.shape[2] == 3:
        qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
    else:
        qimg = QImage(arr.data, w, h, 4 * w, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# Synchronized graphics view
# ---------------------------------------------------------------------------
class SyncGraphicsView(QGraphicsView):
    """QGraphicsView that broadcasts pan/zoom to sibling views."""

    viewChanged = pyqtSignal(QPointF, float)

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self._zoom = 1.0
        self._panning = False
        self._pan_start = QPointF()
        self._syncing = False

        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QColor(30, 30, 30))

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._zoom *= factor
        self.scale(factor, factor)
        self._broadcast()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            self._broadcast()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    def _broadcast(self):
        if self._syncing:
            return
        center = self.mapToScene(self.viewport().rect().center())
        self.viewChanged.emit(center, self._zoom)

    def sync_to(self, center: QPointF, zoom: float):
        if self._syncing:
            return
        self._syncing = True
        ratio = zoom / self._zoom
        if abs(ratio - 1.0) > 1e-6:
            self.scale(ratio, ratio)
            self._zoom = zoom
        self.centerOn(center)
        self._syncing = False

    def fit_in_view(self):
        if self.scene() and self.scene().items():
            self.fitInView(self.scene().itemsBoundingRect(), Qt.KeepAspectRatio)
            self._zoom = self.transform().m11()


# ---------------------------------------------------------------------------
# Panel widget (view + label)
# ---------------------------------------------------------------------------
class PanelWidget(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.view = SyncGraphicsView(title)
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        label = QLabel(f"  {title}")
        label.setStyleSheet(
            "color: #e0e0e0; font-size: 14px; font-weight: bold; padding: 4px;"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(label)
        layout.addWidget(self.view)

    def set_image(self, pixmap: QPixmap):
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))

    def clear(self):
        self.pixmap_item.setPixmap(QPixmap())


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class TiffViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PathoGen — TIFF Viewer")
        self.setMinimumSize(1200, 700)

        self.readers = {"image": None, "mask": None, "condition": None}
        self._overlay_on = False
        self._overlay_opacity = 0.4
        self._mask_color = COLORMAPS["Red"]

        self._build_ui()
        self._connect_sync()
        self._apply_stylesheet()

    # ---- UI ---------------------------------------------------------------
    def _build_ui(self):
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.act_open_image = QAction("Open Image", self)
        self.act_open_mask = QAction("Open Mask", self)
        self.act_open_cond = QAction("Open Condition", self)
        self.act_fit = QAction("Fit (F)", self)
        self.act_reset = QAction("Reset (R)", self)

        for act in [self.act_open_image, self.act_open_mask, self.act_open_cond]:
            toolbar.addAction(act)
        toolbar.addSeparator()
        toolbar.addAction(self.act_fit)
        toolbar.addAction(self.act_reset)
        toolbar.addSeparator()

        self.chk_overlay = QCheckBox("Overlay mask")
        toolbar.addWidget(self.chk_overlay)

        lbl_opacity = QLabel("  Opacity:")
        toolbar.addWidget(lbl_opacity)
        self.sld_opacity = QSlider(Qt.Horizontal)
        self.sld_opacity.setRange(0, 100)
        self.sld_opacity.setValue(40)
        self.sld_opacity.setFixedWidth(120)
        toolbar.addWidget(self.sld_opacity)

        lbl_color = QLabel("  Mask color:")
        toolbar.addWidget(lbl_color)
        self.cmb_color = QComboBox()
        self.cmb_color.addItems(COLORMAPS.keys())
        toolbar.addWidget(self.cmb_color)

        self.panel_image = PanelWidget("Image")
        self.panel_mask = PanelWidget("Mask")
        self.panel_cond = PanelWidget("Condition")

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.panel_image)
        splitter.addWidget(self.panel_mask)
        splitter.addWidget(self.panel_cond)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 1)

        self.setCentralWidget(splitter)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.lbl_zoom = QLabel("Zoom: 100%")
        self.lbl_info = QLabel("")
        self.status.addWidget(self.lbl_info, 1)
        self.status.addPermanentWidget(self.lbl_zoom)

        self.act_open_image.triggered.connect(lambda: self._open_file("image"))
        self.act_open_mask.triggered.connect(lambda: self._open_file("mask"))
        self.act_open_cond.triggered.connect(lambda: self._open_file("condition"))
        self.act_fit.triggered.connect(self._fit_all)
        self.act_reset.triggered.connect(self._reset_all)
        self.chk_overlay.toggled.connect(self._toggle_overlay)
        self.sld_opacity.valueChanged.connect(self._update_overlay_opacity)
        self.cmb_color.currentTextChanged.connect(self._update_mask_color)

    def _connect_sync(self):
        views = [
            self.panel_image.view,
            self.panel_mask.view,
            self.panel_cond.view,
        ]
        for v in views:
            others = [o for o in views if o is not v]
            v.viewChanged.connect(
                lambda center, zoom, _others=others: self._sync_views(center, zoom, _others)
            )

    def _sync_views(self, center, zoom, others):
        for v in others:
            v.sync_to(center, zoom)
        self.lbl_zoom.setText(f"Zoom: {zoom * 100:.0f}%")

    # ---- File operations --------------------------------------------------
    def _open_file(self, role: str):
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Open {role.capitalize()} TIFF",
            "",
            "TIFF Files (*.tiff *.tif *.svs *.ndpi *.mrxs *.scn *.bif);;All Files (*)",
        )
        if not path:
            return
        try:
            reader = SlideReader(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot open file:\n{e}")
            return

        if self.readers[role]:
            self.readers[role].close()
        self.readers[role] = reader

        self._display(role)
        self.lbl_info.setText(
            f"{role.capitalize()}: {Path(path).name}  "
            f"({reader.dimensions[0]}×{reader.dimensions[1]}, "
            f"{reader.level_count} levels, {reader.backend})"
        )

    def _display(self, role: str):
        reader = self.readers[role]
        if reader is None:
            return

        thumb = reader.get_thumbnail((2048, 2048))
        pixmap = numpy_to_qpixmap(np.ascontiguousarray(thumb))

        panel = {"image": self.panel_image, "mask": self.panel_mask, "condition": self.panel_cond}[role]
        panel.set_image(pixmap)
        QTimer.singleShot(50, panel.view.fit_in_view)

        if role in ("image", "mask") and self._overlay_on:
            self._apply_overlay()

    # ---- Overlay ----------------------------------------------------------
    def _toggle_overlay(self, on: bool):
        self._overlay_on = on
        if on:
            self._apply_overlay()
        else:
            if self.readers["image"]:
                self._display("image")

    def _update_overlay_opacity(self, value: int):
        self._overlay_opacity = value / 100.0
        if self._overlay_on:
            self._apply_overlay()

    def _update_mask_color(self, name: str):
        self._mask_color = COLORMAPS.get(name, (255, 0, 0))
        if self._overlay_on:
            self._apply_overlay()

    def _apply_overlay(self):
        img_reader = self.readers["image"]
        mask_reader = self.readers["mask"]
        if not img_reader or not mask_reader:
            return

        img_arr = img_reader.get_thumbnail((2048, 2048))
        mask_arr = mask_reader.get_thumbnail((2048, 2048))

        from PIL import Image
        img_pil = Image.fromarray(img_arr).convert("RGBA")
        mask_pil = Image.fromarray(mask_arr).convert("L")
        mask_pil = mask_pil.resize(img_pil.size, Image.NEAREST)

        mask_np = np.array(mask_pil)
        r, g, b = self._mask_color
        alpha = self._overlay_opacity

        overlay = np.array(img_pil, dtype=np.float32)
        mask_bool = mask_np > 0

        overlay[mask_bool, 0] = overlay[mask_bool, 0] * (1 - alpha) + r * alpha
        overlay[mask_bool, 1] = overlay[mask_bool, 1] * (1 - alpha) + g * alpha
        overlay[mask_bool, 2] = overlay[mask_bool, 2] * (1 - alpha) + b * alpha

        result = np.clip(overlay, 0, 255).astype(np.uint8)
        pixmap = numpy_to_qpixmap(np.ascontiguousarray(result))
        self.panel_image.set_image(pixmap)

    # ---- View actions -----------------------------------------------------
    def _fit_all(self):
        for p in [self.panel_image, self.panel_mask, self.panel_cond]:
            p.view.fit_in_view()

    def _reset_all(self):
        for p in [self.panel_image, self.panel_mask, self.panel_cond]:
            p.view.resetTransform()
            p.view._zoom = 1.0

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_F:
            self._fit_all()
        elif key == Qt.Key_R:
            self._reset_all()
        elif key == Qt.Key_Plus or key == Qt.Key_Equal:
            for p in [self.panel_image, self.panel_mask, self.panel_cond]:
                p.view.scale(1.15, 1.15)
                p.view._zoom *= 1.15
        elif key == Qt.Key_Minus:
            for p in [self.panel_image, self.panel_mask, self.panel_cond]:
                p.view.scale(1 / 1.15, 1 / 1.15)
                p.view._zoom /= 1.15
        else:
            super().keyPressEvent(event)

    # ---- Stylesheet -------------------------------------------------------
    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow { background: #1e1e1e; }
            QToolBar {
                background: #2d2d2d; border: none; padding: 4px; spacing: 6px;
            }
            QToolBar QToolButton {
                background: #3c3c3c; color: #e0e0e0; border: 1px solid #555;
                border-radius: 4px; padding: 6px 12px; font-size: 13px;
            }
            QToolBar QToolButton:hover { background: #505050; }
            QToolBar QToolButton:pressed { background: #606060; }
            QCheckBox { color: #e0e0e0; font-size: 13px; }
            QLabel { color: #c0c0c0; font-size: 13px; }
            QSlider::groove:horizontal {
                background: #555; height: 6px; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0078d4; width: 14px; height: 14px;
                margin: -4px 0; border-radius: 7px;
            }
            QComboBox {
                background: #3c3c3c; color: #e0e0e0; border: 1px solid #555;
                border-radius: 4px; padding: 4px 8px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #2d2d2d; color: #e0e0e0; selection-background-color: #0078d4;
            }
            QStatusBar { background: #007acc; color: white; font-size: 12px; }
            QSplitter::handle { background: #444; width: 2px; }
        """)

    def closeEvent(self, event):
        for r in self.readers.values():
            if r:
                r.close()
        event.accept()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PathoGen TIFF Viewer")

    window = TiffViewer()
    window.show()
    window.showMaximized()

    if len(sys.argv) >= 2:
        window.readers["image"] = SlideReader(sys.argv[1])
        window._display("image")
    if len(sys.argv) >= 3:
        window.readers["mask"] = SlideReader(sys.argv[2])
        window._display("mask")
    if len(sys.argv) >= 4:
        window.readers["condition"] = SlideReader(sys.argv[3])
        window._display("condition")

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
