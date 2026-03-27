"""
PathoGen Test Data Preparation Tool

Reads a single .tiff WSI and exports 3 JPGs for PathoGen inference:
  - image.jpg     : tissue crop (target region)
  - mask.jpg      : binary mask (white = inpaint region)
  - condition.jpg : reference/source tissue pattern

Workflow:
  1. Open TIFF → thumbnail shown on left
  2. Click on thumbnail to pick crop location → "Image" panel
  3. Draw mask on Image panel (brush / rectangle / circle) → "Mask" panel
  4. Click thumbnail again (right-click) to pick condition region → "Condition" panel
  5. Export → saves 3 JPGs to output folder

Usage:
  python prepare_test_data.py
  python prepare_test_data.py slide.tiff
  python prepare_test_data.py slide.tiff --crop-size 512 --output ./output
"""

import sys
import argparse
import numpy as np
from pathlib import Path

from PIL import Image

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QVBoxLayout, QHBoxLayout, QWidget,
    QFileDialog, QLabel, QSlider, QToolBar, QAction, QSplitter,
    QCheckBox, QStatusBar, QMessageBox, QComboBox, QSpinBox,
    QPushButton, QGroupBox, QButtonGroup, QRadioButton,
    QGraphicsRectItem, QGraphicsEllipseItem,
)
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QPointF, QTimer, QRect
from PyQt5.QtGui import (
    QPixmap, QImage, QColor, QPainter, QPen, QBrush, QCursor,
)

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False

import tifffile


# ---------------------------------------------------------------------------
# Slide reader
# ---------------------------------------------------------------------------
class SlideReader:
    def __init__(self, path: str):
        self.path = path
        self._slide = None
        self._tiff_data = None
        self._tiff_series = None
        self._levels = []

        if HAS_OPENSLIDE and self._try_openslide(path):
            self.backend = "openslide"
        else:
            self._load_tifffile(path)
            self.backend = "tifffile"

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
        if tif.series and len(tif.series[0].levels) > 0:
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

    def get_best_level_for_downsample(self, downsample: float) -> int:
        if self.backend == "openslide":
            return self._slide.get_best_level_for_downsample(downsample)
        best = 0
        for i, (w, h) in enumerate(self._levels):
            ds = self._levels[0][0] / w
            if ds <= downsample:
                best = i
        return best

    def read_region_pil(self, location, level, size) -> Image.Image:
        """Read region as PIL RGB image."""
        w, h = size
        level = min(level, self.level_count - 1)

        if self.backend == "openslide":
            pil_img = self._slide.read_region(location, level, (w, h))
            return pil_img.convert("RGB")

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
            region = np.stack([region] * 3, axis=-1)
        elif region.shape[2] == 4:
            region = region[:, :, :3]

        pad_h, pad_w = h - region.shape[0], w - region.shape[1]
        if pad_h > 0 or pad_w > 0:
            region = np.pad(
                region,
                ((0, max(0, pad_h)), (0, max(0, pad_w)), (0, 0)),
                mode="constant", constant_values=255,
            )
        return Image.fromarray(region[:h, :w])

    def get_thumbnail(self, max_size=(1024, 1024)) -> Image.Image:
        if self.backend == "openslide":
            return self._slide.get_thumbnail(max_size).convert("RGB")

        best = 0
        for i in range(self.level_count):
            w, h = self._levels[i]
            if w <= max_size[0] and h <= max_size[1]:
                best = i
                break
            best = i
        region = self.read_region_pil((0, 0), best, self._levels[best])
        region.thumbnail(max_size, Image.LANCZOS)
        return region

    def close(self):
        if self._slide:
            self._slide.close()


def pil_to_qpixmap(pil_img: Image.Image) -> QPixmap:
    if pil_img.mode == "L":
        data = pil_img.tobytes()
        qimg = QImage(data, pil_img.width, pil_img.height, pil_img.width, QImage.Format_Grayscale8)
    elif pil_img.mode == "RGB":
        data = pil_img.tobytes()
        qimg = QImage(data, pil_img.width, pil_img.height, 3 * pil_img.width, QImage.Format_RGB888)
    else:
        pil_img = pil_img.convert("RGBA")
        data = pil_img.tobytes()
        qimg = QImage(data, pil_img.width, pil_img.height, 4 * pil_img.width, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# Thumbnail view — click to pick crop locations
# ---------------------------------------------------------------------------
class ThumbnailView(QGraphicsView):
    """Left-click → pick image crop; Right-click → pick condition crop."""

    imageCropPicked = pyqtSignal(int, int)
    condCropPicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene_ = QGraphicsScene()
        self.setScene(self.scene_)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene_.addItem(self.pixmap_item)

        self._thumb_scale = 1.0
        self._crop_rect = None
        self._cond_rect = None

        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor(30, 30, 30))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    def set_thumbnail(self, pixmap: QPixmap, thumb_scale: float):
        self.pixmap_item.setPixmap(pixmap)
        self._thumb_scale = thumb_scale
        self.scene_.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.scene_.sceneRect(), Qt.KeepAspectRatio)

    def _add_rect(self, x, y, size, color, old_item_attr):
        old_item = getattr(self, old_item_attr)
        if old_item:
            self.scene_.removeItem(old_item)
        ts = self._thumb_scale
        rect = QGraphicsRectItem(x / ts, y / ts, size / ts, size / ts)
        rect.setPen(QPen(color, 2))
        rect.setBrush(QBrush(Qt.NoBrush))
        self.scene_.addItem(rect)
        setattr(self, old_item_attr, rect)

    def show_image_rect(self, x, y, size):
        self._add_rect(x, y, size, QColor(0, 200, 0), "_crop_rect")

    def show_cond_rect(self, x, y, size):
        self._add_rect(x, y, size, QColor(0, 120, 255), "_cond_rect")

    def mousePressEvent(self, event):
        pos = self.mapToScene(event.pos())
        level0_x = int(pos.x() * self._thumb_scale)
        level0_y = int(pos.y() * self._thumb_scale)

        if event.button() == Qt.LeftButton:
            self.imageCropPicked.emit(level0_x, level0_y)
        elif event.button() == Qt.RightButton:
            self.condCropPicked.emit(level0_x, level0_y)
        super().mousePressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap_item.pixmap() and not self.pixmap_item.pixmap().isNull():
            self.fitInView(self.scene_.sceneRect(), Qt.KeepAspectRatio)


# ---------------------------------------------------------------------------
# Mask drawing canvas
# ---------------------------------------------------------------------------
class MaskCanvas(QGraphicsView):
    """Draws binary mask over the image crop."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene_ = QGraphicsScene()
        self.setScene(self.scene_)
        self.bg_item = QGraphicsPixmapItem()
        self.mask_item = QGraphicsPixmapItem()
        self.scene_.addItem(self.bg_item)
        self.scene_.addItem(self.mask_item)

        self._mask_img = None
        self._drawing = False
        self._tool = "brush"
        self._brush_size = 30
        self._draw_start = None

        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor(30, 30, 30))

    def set_background(self, pixmap: QPixmap):
        self.bg_item.setPixmap(pixmap)
        self.scene_.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.scene_.sceneRect(), Qt.KeepAspectRatio)
        self._mask_img = QImage(pixmap.size(), QImage.Format_ARGB32)
        self._mask_img.fill(QColor(0, 0, 0, 0))
        self._update_mask_display()

    def set_tool(self, tool: str):
        self._tool = tool

    def set_brush_size(self, size: int):
        self._brush_size = size

    def clear_mask(self):
        if self._mask_img:
            self._mask_img.fill(QColor(0, 0, 0, 0))
            self._update_mask_display()

    def get_mask_image(self) -> Image.Image:
        """Return mask as PIL L-mode image (white=inpaint, black=keep)."""
        if self._mask_img is None:
            return None
        w, h = self._mask_img.width(), self._mask_img.height()
        ptr = self._mask_img.bits()
        ptr.setsize(h * w * 4)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4))
        alpha = arr[:, :, 3]
        binary = np.where(alpha > 0, 255, 0).astype(np.uint8)
        return Image.fromarray(binary, mode="L")

    def _update_mask_display(self):
        pm = QPixmap.fromImage(self._mask_img)
        self.mask_item.setPixmap(pm)
        self.mask_item.setOpacity(0.45)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._mask_img:
            self._drawing = True
            pos = self.mapToScene(event.pos())
            self._draw_start = pos
            if self._tool == "brush":
                self._paint_at(pos)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drawing and self._tool == "brush":
            pos = self.mapToScene(event.pos())
            self._paint_at(pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drawing:
            self._drawing = False
            pos = self.mapToScene(event.pos())
            if self._tool == "rectangle" and self._draw_start:
                self._draw_rect(self._draw_start, pos)
            elif self._tool == "circle" and self._draw_start:
                self._draw_ellipse(self._draw_start, pos)
            self._draw_start = None
        super().mouseReleaseEvent(event)

    def _paint_at(self, pos: QPointF):
        painter = QPainter(self._mask_img)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 255))
        r = self._brush_size / 2
        painter.drawEllipse(pos.toPoint(), int(r), int(r))
        painter.end()
        self._update_mask_display()

    def _draw_rect(self, p1: QPointF, p2: QPointF):
        painter = QPainter(self._mask_img)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 255))
        rect = QRectF(p1, p2).normalized()
        painter.drawRect(rect)
        painter.end()
        self._update_mask_display()

    def _draw_ellipse(self, p1: QPointF, p2: QPointF):
        painter = QPainter(self._mask_img)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 255))
        rect = QRectF(p1, p2).normalized()
        painter.drawEllipse(rect)
        painter.end()
        self._update_mask_display()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.bg_item.pixmap() and not self.bg_item.pixmap().isNull():
            self.fitInView(self.scene_.sceneRect(), Qt.KeepAspectRatio)


# ---------------------------------------------------------------------------
# Preview panel (for condition)
# ---------------------------------------------------------------------------
class PreviewPanel(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene_ = QGraphicsScene()
        self.setScene(self.scene_)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene_.addItem(self.pixmap_item)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor(30, 30, 30))

    def set_image(self, pixmap: QPixmap):
        self.pixmap_item.setPixmap(pixmap)
        self.scene_.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.scene_.sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap_item.pixmap() and not self.pixmap_item.pixmap().isNull():
            self.fitInView(self.scene_.sceneRect(), Qt.KeepAspectRatio)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class PrepareDataWindow(QMainWindow):
    def __init__(self, default_crop_size=512, output_dir="./output"):
        super().__init__()
        self.setWindowTitle("PathoGen — Prepare Test Data (image / mask / condition)")
        self.setMinimumSize(1400, 750)

        self.reader = None
        self.crop_size = default_crop_size
        self.output_dir = Path(output_dir)

        self._image_pil = None
        self._cond_pil = None
        self._image_loc = None
        self._cond_loc = None

        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        # --- Toolbar ---
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        act_open = QAction("Open TIFF", self)
        act_open.triggered.connect(self._open_tiff)
        toolbar.addAction(act_open)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel("  Crop size: "))
        self.spn_crop = QSpinBox()
        self.spn_crop.setRange(64, 4096)
        self.spn_crop.setSingleStep(64)
        self.spn_crop.setValue(self.crop_size)
        self.spn_crop.valueChanged.connect(self._on_crop_size_changed)
        toolbar.addWidget(self.spn_crop)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel("  Mask tool: "))
        self.cmb_tool = QComboBox()
        self.cmb_tool.addItems(["brush", "rectangle", "circle"])
        self.cmb_tool.currentTextChanged.connect(self._on_tool_changed)
        toolbar.addWidget(self.cmb_tool)

        toolbar.addWidget(QLabel("  Brush size: "))
        self.sld_brush = QSlider(Qt.Horizontal)
        self.sld_brush.setRange(5, 200)
        self.sld_brush.setValue(30)
        self.sld_brush.setFixedWidth(120)
        self.sld_brush.valueChanged.connect(self._on_brush_size)
        toolbar.addWidget(self.sld_brush)
        toolbar.addSeparator()

        act_clear = QAction("Clear Mask", self)
        act_clear.triggered.connect(self._clear_mask)
        toolbar.addAction(act_clear)

        act_auto = QAction("Auto Center Mask", self)
        act_auto.triggered.connect(self._auto_center_mask)
        toolbar.addAction(act_auto)
        toolbar.addSeparator()

        act_export = QAction("Export JPGs", self)
        act_export.triggered.connect(self._export)
        toolbar.addAction(act_export)

        # --- Main layout: Thumbnail | MaskCanvas | Condition ---
        splitter = QSplitter(Qt.Horizontal)

        # Left: thumbnail
        left_w = QWidget()
        left_lay = QVBoxLayout(left_w)
        left_lay.setContentsMargins(0, 0, 0, 0)
        lbl_thumb = QLabel("  Slide Thumbnail")
        lbl_thumb.setStyleSheet("color:#e0e0e0; font-weight:bold; font-size:13px; padding:4px;")
        self.lbl_hint = QLabel("  Left-click → Image crop  |  Right-click → Condition crop")
        self.lbl_hint.setStyleSheet("color:#888; font-size:11px; padding:2px;")
        self.thumb_view = ThumbnailView()
        left_lay.addWidget(lbl_thumb)
        left_lay.addWidget(self.lbl_hint)
        left_lay.addWidget(self.thumb_view)

        # Center: mask drawing canvas
        center_w = QWidget()
        center_lay = QVBoxLayout(center_w)
        center_lay.setContentsMargins(0, 0, 0, 0)
        lbl_mask = QLabel("  Image + Draw Mask")
        lbl_mask.setStyleSheet("color:#e0e0e0; font-weight:bold; font-size:13px; padding:4px;")
        self.mask_canvas = MaskCanvas()
        center_lay.addWidget(lbl_mask)
        center_lay.addWidget(self.mask_canvas)

        # Right: condition preview
        right_w = QWidget()
        right_lay = QVBoxLayout(right_w)
        right_lay.setContentsMargins(0, 0, 0, 0)
        lbl_cond = QLabel("  Condition")
        lbl_cond.setStyleSheet("color:#e0e0e0; font-weight:bold; font-size:13px; padding:4px;")
        self.cond_view = PreviewPanel()
        right_lay.addWidget(lbl_cond)
        right_lay.addWidget(self.cond_view)

        splitter.addWidget(left_w)
        splitter.addWidget(center_w)
        splitter.addWidget(right_w)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)

        self.setCentralWidget(splitter)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.lbl_status = QLabel("Open a TIFF file to begin")
        self.status.addWidget(self.lbl_status, 1)

        # Connections
        self.thumb_view.imageCropPicked.connect(self._on_image_crop_picked)
        self.thumb_view.condCropPicked.connect(self._on_cond_crop_picked)

    # ---- Actions ----------------------------------------------------------
    def _open_tiff(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open TIFF / WSI", "",
            "TIFF Files (*.tiff *.tif *.svs *.ndpi *.mrxs *.scn *.bif);;All Files (*)",
        )
        if not path:
            return
        try:
            reader = SlideReader(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot open:\n{e}")
            return

        if self.reader:
            self.reader.close()
        self.reader = reader

        thumb_pil = reader.get_thumbnail((1200, 1200))
        thumb_scale = reader.dimensions[0] / thumb_pil.width
        pixmap = pil_to_qpixmap(thumb_pil)
        self.thumb_view.set_thumbnail(pixmap, thumb_scale)

        w, h = reader.dimensions
        self.lbl_status.setText(
            f"{Path(path).name}  |  {w}×{h}  |  {reader.level_count} levels  |  {reader.backend}"
        )

    def _on_image_crop_picked(self, x, y):
        if not self.reader:
            return
        cs = self.crop_size
        w, h = self.reader.dimensions
        x = max(0, min(x - cs // 2, w - cs))
        y = max(0, min(y - cs // 2, h - cs))
        self._image_loc = (x, y)

        pil_img = self.reader.read_region_pil((x, y), 0, (cs, cs))
        self._image_pil = pil_img
        pixmap = pil_to_qpixmap(pil_img)
        self.mask_canvas.set_background(pixmap)
        self.thumb_view.show_image_rect(x, y, cs)

        self.lbl_status.setText(
            f"Image crop @ ({x}, {y}) size {cs}×{cs}  —  draw mask, then right-click thumbnail for condition"
        )

    def _on_cond_crop_picked(self, x, y):
        if not self.reader:
            return
        cs = self.crop_size
        w, h = self.reader.dimensions
        x = max(0, min(x - cs // 2, w - cs))
        y = max(0, min(y - cs // 2, h - cs))
        self._cond_loc = (x, y)

        pil_img = self.reader.read_region_pil((x, y), 0, (cs, cs))
        self._cond_pil = pil_img
        pixmap = pil_to_qpixmap(pil_img)
        self.cond_view.set_image(pixmap)
        self.thumb_view.show_cond_rect(x, y, cs)

        self.lbl_status.setText(
            f"Condition crop @ ({x}, {y}) size {cs}×{cs}  —  ready to export!"
        )

    def _on_crop_size_changed(self, val):
        self.crop_size = val

    def _on_tool_changed(self, tool):
        self.mask_canvas.set_tool(tool)

    def _on_brush_size(self, val):
        self.mask_canvas.set_brush_size(val)

    def _clear_mask(self):
        self.mask_canvas.clear_mask()

    def _auto_center_mask(self):
        """Generate a centered rectangular mask (25% of crop area)."""
        if self._image_pil is None:
            return
        cs = self.crop_size
        margin = cs // 4
        self.mask_canvas.clear_mask()
        p1 = QPointF(margin, margin)
        p2 = QPointF(cs - margin, cs - margin)
        self.mask_canvas._draw_rect(p1, p2)
        self.lbl_status.setText("Auto center mask applied (50% region)")

    # ---- Export -----------------------------------------------------------
    def _export(self):
        if self._image_pil is None:
            QMessageBox.warning(self, "Warning", "No image crop selected.\nLeft-click on the thumbnail first.")
            return

        mask_pil = self.mask_canvas.get_mask_image()
        if mask_pil is None or np.array(mask_pil).max() == 0:
            QMessageBox.warning(self, "Warning", "Mask is empty.\nDraw a mask on the image first.")
            return

        if self._cond_pil is None:
            QMessageBox.warning(self, "Warning", "No condition crop selected.\nRight-click on the thumbnail to pick condition region.")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder", str(self.output_dir))
        if not out_dir:
            return
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        img_path = out_dir / "image.jpg"
        mask_path = out_dir / "mask.jpg"
        cond_path = out_dir / "condition.jpg"

        self._image_pil.convert("RGB").save(img_path, "JPEG", quality=95)
        mask_pil.save(mask_path, "JPEG", quality=95)
        self._cond_pil.convert("RGB").save(cond_path, "JPEG", quality=95)

        self.lbl_status.setText(f"Exported → {out_dir}/  (image.jpg, mask.jpg, condition.jpg)")
        QMessageBox.information(
            self, "Export Complete",
            f"Saved 3 files to:\n{out_dir}\n\n"
            f"  • image.jpg     ({self.crop_size}×{self.crop_size})\n"
            f"  • mask.jpg      ({self.crop_size}×{self.crop_size})\n"
            f"  • condition.jpg ({self.crop_size}×{self.crop_size})\n\n"
            f"Ready for PathoGen inference!",
        )

    # ---- Style ------------------------------------------------------------
    def _apply_style(self):
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
            QLabel { color: #c0c0c0; font-size: 13px; }
            QComboBox {
                background: #3c3c3c; color: #e0e0e0; border: 1px solid #555;
                border-radius: 4px; padding: 4px 8px;
            }
            QComboBox QAbstractItemView {
                background: #2d2d2d; color: #e0e0e0; selection-background-color: #0078d4;
            }
            QSpinBox {
                background: #3c3c3c; color: #e0e0e0; border: 1px solid #555;
                border-radius: 4px; padding: 4px 8px;
            }
            QSlider::groove:horizontal {
                background: #555; height: 6px; border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0078d4; width: 14px; height: 14px;
                margin: -4px 0; border-radius: 7px;
            }
            QStatusBar { background: #007acc; color: white; font-size: 12px; }
            QSplitter::handle { background: #444; width: 2px; }
        """)

    def closeEvent(self, event):
        if self.reader:
            self.reader.close()
        event.accept()


# ---------------------------------------------------------------------------
# CLI mode: automatic extraction without GUI
# ---------------------------------------------------------------------------
def cli_extract(tiff_path, crop_size, output_dir, image_xy=None, cond_xy=None,
                mask_type="center"):
    """Extract image/mask/condition from TIFF without GUI."""
    print(f"Opening: {tiff_path}")
    reader = SlideReader(tiff_path)
    w, h = reader.dimensions
    print(f"  Dimensions: {w}×{h}, levels: {reader.level_count}, backend: {reader.backend}")

    if image_xy is None:
        cx, cy = w // 2, h // 2
        image_xy = (cx - crop_size // 2, cy - crop_size // 2)
    if cond_xy is None:
        cx, cy = image_xy
        offset = crop_size + crop_size // 2
        cond_xy = (min(cx + offset, w - crop_size), cy)

    image_xy = (max(0, min(image_xy[0], w - crop_size)),
                max(0, min(image_xy[1], h - crop_size)))
    cond_xy = (max(0, min(cond_xy[0], w - crop_size)),
               max(0, min(cond_xy[1], h - crop_size)))

    print(f"  Image crop @ {image_xy}")
    image_pil = reader.read_region_pil(image_xy, 0, (crop_size, crop_size))

    print(f"  Condition crop @ {cond_xy}")
    cond_pil = reader.read_region_pil(cond_xy, 0, (crop_size, crop_size))

    if mask_type == "center":
        m = crop_size // 4
        mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
        mask[m:crop_size - m, m:crop_size - m] = 255
    elif mask_type == "random":
        rng = np.random.default_rng(42)
        mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
        for _ in range(rng.integers(3, 8)):
            cx = rng.integers(crop_size // 4, 3 * crop_size // 4)
            cy = rng.integers(crop_size // 4, 3 * crop_size // 4)
            rx = rng.integers(crop_size // 8, crop_size // 3)
            ry = rng.integers(crop_size // 8, crop_size // 3)
            yy, xx = np.ogrid[:crop_size, :crop_size]
            ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
            mask[ellipse] = 255
    else:
        mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
        mask[crop_size // 4:3 * crop_size // 4, crop_size // 4:3 * crop_size // 4] = 255

    mask_pil = Image.fromarray(mask, mode="L")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    image_pil.save(out / "image.jpg", "JPEG", quality=95)
    mask_pil.save(out / "mask.jpg", "JPEG", quality=95)
    cond_pil.save(out / "condition.jpg", "JPEG", quality=95)

    print(f"\nExported to {out}/")
    print(f"  image.jpg     ({crop_size}×{crop_size})")
    print(f"  mask.jpg      ({crop_size}×{crop_size})")
    print(f"  condition.jpg ({crop_size}×{crop_size})")
    print("\nReady for PathoGen inference:")
    print('  image = Image.open("image.jpg")')
    print('  mask  = Image.open("mask.jpg")')
    print('  condition = Image.open("condition.jpg")')
    print("  result = model(image, mask, condition)")

    reader.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Prepare image/mask/condition JPGs from a TIFF for PathoGen"
    )
    parser.add_argument("tiff", nargs="?", help="Path to TIFF/WSI file")
    parser.add_argument("--crop-size", type=int, default=512, help="Crop size in pixels (default: 512)")
    parser.add_argument("--output", "-o", default="./output", help="Output directory (default: ./output)")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode (no GUI)")
    parser.add_argument("--image-xy", type=str, default=None, help="Image crop location 'x,y' (CLI mode)")
    parser.add_argument("--cond-xy", type=str, default=None, help="Condition crop location 'x,y' (CLI mode)")
    parser.add_argument("--mask-type", choices=["center", "random"], default="center",
                        help="Mask type for CLI mode (default: center)")

    args = parser.parse_args()

    if args.cli and args.tiff:
        image_xy = tuple(map(int, args.image_xy.split(","))) if args.image_xy else None
        cond_xy = tuple(map(int, args.cond_xy.split(","))) if args.cond_xy else None
        cli_extract(args.tiff, args.crop_size, args.output, image_xy, cond_xy, args.mask_type)
        return

    app = QApplication(sys.argv)
    window = PrepareDataWindow(default_crop_size=args.crop_size, output_dir=args.output)
    window.show()
    window.showMaximized()

    if args.tiff:
        try:
            window.reader = SlideReader(args.tiff)
            thumb_pil = window.reader.get_thumbnail((1200, 1200))
            thumb_scale = window.reader.dimensions[0] / thumb_pil.width
            pixmap = pil_to_qpixmap(thumb_pil)
            window.thumb_view.set_thumbnail(pixmap, thumb_scale)
            w, h = window.reader.dimensions
            window.lbl_status.setText(
                f"{Path(args.tiff).name}  |  {w}×{h}  |  {window.reader.level_count} levels"
            )
        except Exception as e:
            QMessageBox.critical(window, "Error", f"Cannot open:\n{e}")

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
