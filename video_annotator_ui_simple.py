import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QSettings, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


class ImageLabel(QLabel):
    bboxChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: gray;")
        self.setMinimumSize(640, 480)

        self.drawing = False
        self.moving_bbox = False
        self.moving_bbox_index = None
        self.move_offset = None
        self.start_point = None
        self.end_point = None
        self.bboxes = []
        self.current_class_id = 0
        self.fixed_box_enabled = False
        self.fixed_box_w = 120
        self.fixed_box_h = 120
        self.class_names = ["Ковш", "Сломанный ковш", "Отсутствует ковш", "Стык ленты"]
        self.class_colors = {
            0: QColor(0, 255, 0),
            1: QColor(255, 165, 0),
            2: QColor(255, 80, 80),
            3: QColor(0, 200, 255),
        }

    def get_image_coordinate(self, pos):
        if not self.pixmap():
            return None
        pix_size = self.pixmap().size()
        lbl_size = self.size()
        x_offset = (lbl_size.width() - pix_size.width()) // 2
        y_offset = (lbl_size.height() - pix_size.height()) // 2

        img_x = pos.x() - x_offset
        img_y = pos.y() - y_offset
        if 0 <= img_x < pix_size.width() and 0 <= img_y < pix_size.height():
            orig_w = self.property("orig_w") or pix_size.width()
            orig_h = self.property("orig_h") or pix_size.height()
            return int(img_x * orig_w / pix_size.width()), int(img_y * orig_h / pix_size.height())
        return None

    def get_widget_coordinate(self, orig_x, orig_y):
        if not self.pixmap():
            return None
        pix_size = self.pixmap().size()
        lbl_size = self.size()
        x_offset = (lbl_size.width() - pix_size.width()) // 2
        y_offset = (lbl_size.height() - pix_size.height()) // 2

        orig_w = self.property("orig_w") or pix_size.width()
        orig_h = self.property("orig_h") or pix_size.height()
        if orig_w == 0 or orig_h == 0:
            return None
        return int(orig_x * pix_size.width() / orig_w) + x_offset, int(orig_y * pix_size.height() / orig_h) + y_offset

    def _remove_bbox_at(self, coord):
        cx, cy = coord
        for i in reversed(range(len(self.bboxes))):
            bx, by, bw, bh = self.bboxes[i][:4]
            if bx <= cx <= bx + bw and by <= cy <= by + bh:
                self.bboxes.pop(i)
                self.bboxChanged.emit()
                self.update()
                return True
        return False

    def _find_bbox_index(self, coord):
        cx, cy = coord
        for i in reversed(range(len(self.bboxes))):
            bx, by, bw, bh = self.bboxes[i][:4]
            if bx <= cx <= bx + bw and by <= cy <= by + bh:
                return i
        return None

    def mousePressEvent(self, event):
        img_coord = self.get_image_coordinate(event.pos())
        if img_coord:
            if event.button() == Qt.MouseButton.LeftButton:
                move_idx = self._find_bbox_index(img_coord)
                if move_idx is not None:
                    bx, by, *_ = self.bboxes[move_idx]
                    cx, cy = img_coord
                    self.moving_bbox = True
                    self.moving_bbox_index = move_idx
                    self.move_offset = (cx - bx, cy - by)
                    super().mousePressEvent(event)
                    return

                if self.fixed_box_enabled:
                    # По требованию: клик задаёт левый верхний угол fixed-box
                    x, y = img_coord
                    max_w = self.property("orig_w") or 0
                    max_h = self.property("orig_h") or 0
                    bw = min(self.fixed_box_w, max_w) if max_w > 0 else self.fixed_box_w
                    bh = min(self.fixed_box_h, max_h) if max_h > 0 else self.fixed_box_h
                    x = max(0, min(x, max_w - bw if max_w > 0 else x))
                    y = max(0, min(y, max_h - bh if max_h > 0 else y))
                    self.bboxes.append((x, y, bw, bh, 1.0, self.current_class_id))
                    self.bboxChanged.emit()
                    self.update()
                else:
                    self.drawing = True
                    self.start_point = img_coord
                    self.end_point = img_coord
            elif event.button() == Qt.MouseButton.RightButton:
                self._remove_bbox_at(img_coord)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.moving_bbox and self.moving_bbox_index is not None:
            img_coord = self.get_image_coordinate(event.pos())
            if img_coord:
                cx, cy = img_coord
                off_x, off_y = self.move_offset or (0, 0)
                bx, by, bw, bh, *rest = self.bboxes[self.moving_bbox_index]
                max_w = self.property("orig_w") or 0
                max_h = self.property("orig_h") or 0
                nx = max(0, min(cx - off_x, max_w - bw if max_w > 0 else cx - off_x))
                ny = max(0, min(cy - off_y, max_h - bh if max_h > 0 else cy - off_y))
                self.bboxes[self.moving_bbox_index] = (nx, ny, bw, bh, *rest)
                self.update()
            super().mouseMoveEvent(event)
            return

        if self.drawing:
            img_coord = self.get_image_coordinate(event.pos())
            if img_coord:
                self.end_point = img_coord
                self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.moving_bbox and event.button() == Qt.MouseButton.LeftButton:
            self.moving_bbox = False
            self.moving_bbox_index = None
            self.move_offset = None
            self.bboxChanged.emit()
            self.update()
            super().mouseReleaseEvent(event)
            return

        if self.drawing and event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            img_coord = self.get_image_coordinate(event.pos())
            if img_coord:
                self.end_point = img_coord
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                x = min(x1, x2)
                y = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                if w > 3 and h > 3:
                    self.bboxes.append((x, y, w, h, 1.0, self.current_class_id))
                    self.bboxChanged.emit()
            self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.pixmap():
            return
        painter = QPainter(self)
        for bbox_data in self.bboxes:
            bx, by, bw, bh = bbox_data[:4]
            cls_id = int(bbox_data[5]) if len(bbox_data) > 5 else 0
            w_tl = self.get_widget_coordinate(bx, by)
            w_br = self.get_widget_coordinate(bx + bw, by + bh)
            if not w_tl or not w_br:
                continue
            painter.setPen(QPen(self.class_colors.get(cls_id, QColor(0, 255, 0)), 2))
            painter.drawRect(w_tl[0], w_tl[1], w_br[0] - w_tl[0], w_br[1] - w_tl[1])
            class_name = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else f"class_{cls_id}"
            painter.drawText(w_tl[0], max(0, w_tl[1] - 5), f"{cls_id}: {class_name}")

        if self.drawing and self.start_point and self.end_point:
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            w_tl = self.get_widget_coordinate(min(x1, x2), min(y1, y2))
            w_br = self.get_widget_coordinate(max(x1, x2), max(y1, y2))
            if w_tl and w_br:
                painter.setPen(QPen(QColor(0, 255, 255), 2, Qt.PenStyle.DashLine))
                painter.drawRect(w_tl[0], w_tl[1], w_br[0] - w_tl[0], w_br[1] - w_tl[1])


class SimpleAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.base_window_title = "Simple Frame Annotator"
        self.setWindowTitle(self.base_window_title)
        self.resize(1200, 800)

        self.settings = QSettings("MyOrg", "VideoAnnotatorSimple")
        self.class_names = ["Ковш", "Сломанный ковш", "Отсутствует ковш", "Стык ленты"]

        self.frames_dir = None
        self.annotations_dir = None
        self.labels_dir = None
        self.image_files = []
        self.current_frame = None
        self.current_frame_idx = 0
        self.total_frames = 0
        self.autosave_enabled = True
        self.unsaved_changes = False

        self.autosave_timer = QTimer(self)
        self.autosave_timer.setSingleShot(True)
        self.autosave_timer.setInterval(350)
        self.autosave_timer.timeout.connect(self.save_current_label)

        self.setup_ui()
        self.connect_signals()
        self.load_state()

    def setup_ui(self):
        root = QWidget(self)
        self.setCentralWidget(root)
        main = QVBoxLayout(root)

        top = QHBoxLayout()
        self.btn_load_frames = QPushButton("Load Prepared Frames")
        self.btn_set_ann_dir = QPushButton("Set Annotation Dir")
        self.btn_save = QPushButton("Save Label")
        self.btn_prev_unlabeled = QPushButton("Prev Unlabeled")
        self.btn_next_unlabeled = QPushButton("Next Unlabeled")

        self.cmb_class = QComboBox()
        self.cmb_class.addItems(self.class_names)
        self.chk_autoreset_class = QCheckBox("Auto reset class to 'Ковш'")
        self.chk_autoreset_class.setChecked(True)

        self.chk_fixed_box = QCheckBox("Fixed Box")
        self.fixed_box_w = QSpinBox()
        self.fixed_box_w.setRange(8, 2000)
        self.fixed_box_w.setValue(120)
        self.fixed_box_w.setPrefix("BW: ")
        self.fixed_box_h = QSpinBox()
        self.fixed_box_h.setRange(8, 2000)
        self.fixed_box_h.setValue(120)
        self.fixed_box_h.setPrefix("BH: ")

        self.chk_autosave = QCheckBox("Autosave")
        self.chk_autosave.setChecked(True)

        for w in [
            self.btn_load_frames,
            self.btn_set_ann_dir,
            self.btn_save,
            self.btn_prev_unlabeled,
            self.btn_next_unlabeled,
            self.cmb_class,
            self.chk_autoreset_class,
            self.chk_fixed_box,
            self.fixed_box_w,
            self.fixed_box_h,
            self.chk_autosave,
        ]:
            top.addWidget(w)
        top.addStretch()
        main.addLayout(top)

        self.image_label = ImageLabel()
        self.image_label.setProperty("orig_w", 0)
        self.image_label.setProperty("orig_h", 0)
        main.addWidget(self.image_label, stretch=1)

        bottom = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.lbl_info = QLabel("Frame: 0 / 0")
        bottom.addWidget(self.slider)
        bottom.addWidget(self.lbl_info)
        main.addLayout(bottom)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(
            "Ready. Hotkeys: ←/→, A/D, Space/Enter(next), Ctrl+S(save), Ctrl+Z(undo), C(clear), "
            "U/Delete(remove last), N/P (next/prev unlabeled), 1..4 (class)"
        )

    def connect_signals(self):
        self.btn_load_frames.clicked.connect(self.load_frames_folder)
        self.btn_set_ann_dir.clicked.connect(self.set_annotations_folder)
        self.btn_save.clicked.connect(self.save_current_label)
        self.btn_prev_unlabeled.clicked.connect(self.goto_prev_unlabeled)
        self.btn_next_unlabeled.clicked.connect(self.goto_next_unlabeled)
        self.slider.valueChanged.connect(self.goto_frame)
        self.cmb_class.currentIndexChanged.connect(self.on_class_changed)
        self.chk_autosave.stateChanged.connect(
            lambda v: setattr(self, "autosave_enabled", v == Qt.CheckState.Checked.value)
        )
        self.chk_fixed_box.stateChanged.connect(self.on_fixed_box_controls_changed)
        self.fixed_box_w.valueChanged.connect(self.on_fixed_box_controls_changed)
        self.fixed_box_h.valueChanged.connect(self.on_fixed_box_controls_changed)
        self.image_label.bboxChanged.connect(self.on_bbox_changed)

    def load_state(self):
        self.chk_fixed_box.setChecked(self.settings.value("fixed_box_enabled", False, type=bool))
        self.fixed_box_w.setValue(self.settings.value("fixed_box_w", 120, type=int))
        self.fixed_box_h.setValue(self.settings.value("fixed_box_h", 120, type=int))
        self.on_fixed_box_controls_changed()

        last_frames = self.settings.value("last_frames_dir", "")
        last_frame_idx = self.settings.value("last_frame_idx", 0, type=int)
        if last_frames and Path(last_frames).exists():
            self.load_frames_folder_internal(last_frames)
            if self.total_frames > 0:
                self.goto_frame(max(0, min(last_frame_idx, self.total_frames - 1)))

    def save_state(self):
        if self.frames_dir:
            self.settings.setValue("last_frames_dir", str(self.frames_dir))
        if self.annotations_dir:
            self.settings.setValue("last_annotations_dir", str(self.annotations_dir))
        self.settings.setValue("last_frame_idx", self.current_frame_idx)
        self.settings.setValue("fixed_box_enabled", self.chk_fixed_box.isChecked())
        self.settings.setValue("fixed_box_w", self.fixed_box_w.value())
        self.settings.setValue("fixed_box_h", self.fixed_box_h.value())

    def closeEvent(self, event):
        self.flush_changes()
        self.save_state()
        self.update_fixed_box_log()
        super().closeEvent(event)

    def read_image_robust(self, img_path):
        try:
            file_bytes = np.fromfile(str(img_path), dtype=np.uint8)
            if file_bytes.size == 0:
                return None
            return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def on_class_changed(self, class_idx):
        self.image_label.current_class_id = class_idx
        self.status_bar.showMessage(f"Active class: {self.class_names[class_idx]}")

    def on_fixed_box_controls_changed(self, _value=None):
        self.image_label.fixed_box_enabled = self.chk_fixed_box.isChecked()
        self.image_label.fixed_box_w = self.fixed_box_w.value()
        self.image_label.fixed_box_h = self.fixed_box_h.value()
        self.settings.setValue("fixed_box_enabled", self.chk_fixed_box.isChecked())
        self.settings.setValue("fixed_box_w", self.fixed_box_w.value())
        self.settings.setValue("fixed_box_h", self.fixed_box_h.value())
        self.update_fixed_box_log()

    def on_bbox_changed(self):
        self.unsaved_changes = True
        if self.chk_autoreset_class.isChecked() and self.image_label.bboxes:
            last_cls = int(self.image_label.bboxes[-1][5]) if len(self.image_label.bboxes[-1]) > 5 else 0
            if last_cls != 0:
                self.cmb_class.blockSignals(True)
                self.cmb_class.setCurrentIndex(0)
                self.cmb_class.blockSignals(False)
                self.image_label.current_class_id = 0
        if self.autosave_enabled:
            self.autosave_timer.start()

    def infer_default_annotations_dir(self, frames_dir: Path):
        explicit = self.settings.value("last_annotations_dir", "")
        candidate = frames_dir.parent / f"{frames_dir.name}_annotations"
        if candidate.exists():
            return candidate
        if explicit and Path(explicit).exists():
            return Path(explicit)
        return candidate

    def load_frames_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Prepared Frames Folder")
        if folder:
            self.load_frames_folder_internal(folder)

    def load_frames_folder_internal(self, folder):
        folder_path = Path(folder)
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = sorted([p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in valid_extensions])
        if not image_files:
            self.status_bar.showMessage(f"No images found in: {folder_path}")
            return

        self.flush_changes()
        self.frames_dir = folder_path
        self.image_files = image_files
        self.total_frames = len(image_files)
        self.current_frame_idx = 0

        if self.annotations_dir is None:
            self.annotations_dir = self.infer_default_annotations_dir(folder_path)
        self.labels_dir = self.annotations_dir / "labels"
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.write_dataset_link()

        self.slider.setEnabled(True)
        self.slider.setRange(0, self.total_frames - 1)
        self.goto_frame(0)
        self.update_fixed_box_log()

    def set_annotations_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Annotation Folder")
        if not folder:
            return
        self.flush_changes()
        self.annotations_dir = Path(folder)
        self.labels_dir = self.annotations_dir / "labels"
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.settings.setValue("last_annotations_dir", str(self.annotations_dir))
        self.write_dataset_link()
        self.goto_frame(self.current_frame_idx)
        self.update_fixed_box_log()

    def write_dataset_link(self):
        if not self.annotations_dir or not self.frames_dir:
            return
        link_path = self.annotations_dir / "dataset_link.txt"
        link_path.write_text(
            f"frames_dir={self.frames_dir}\nlabels_dir={self.labels_dir}\n",
            encoding="utf-8",
        )

    def update_window_title(self):
        if self.image_files and 0 <= self.current_frame_idx < len(self.image_files):
            self.setWindowTitle(f"{self.base_window_title} — {self.image_files[self.current_frame_idx]}")
        else:
            self.setWindowTitle(self.base_window_title)

    def label_path_for_index(self, idx):
        base_name = self.image_files[idx].stem
        return self.labels_dir / f"{base_name}.txt"

    def flush_changes(self):
        if self.unsaved_changes:
            self.save_current_label()

    def save_current_label(self):
        if self.current_frame is None or not self.labels_dir or not self.image_files:
            return

        lbl_path = self.label_path_for_index(self.current_frame_idx)
        h, w = self.current_frame.shape[:2]
        with open(lbl_path, "w", encoding="utf-8") as f:
            for bbox_data in self.image_label.bboxes:
                bx, by, bw, bh = bbox_data[:4]
                cls_id = int(bbox_data[5]) if len(bbox_data) > 5 else 0
                x_center = (bx + bw / 2.0) / w
                y_center = (by + bh / 2.0) / h
                norm_w = bw / w
                norm_h = bh / h
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        self.unsaved_changes = False
        self.status_bar.showMessage(f"Saved: {lbl_path.name}")
        self.update_fixed_box_log()

    def load_label(self, idx):
        self.image_label.bboxes = []
        lbl_path = self.label_path_for_index(idx)
        if not lbl_path.exists() or self.current_frame is None:
            return

        h, w = self.current_frame.shape[:2]
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(float(parts[0]))
                    xc, yc, bw, bh = map(float, parts[1:5])
                    bx = int((xc - bw / 2) * w)
                    by = int((yc - bh / 2) * h)
                    self.image_label.bboxes.append((bx, by, int(bw * w), int(bh * h), 1.0, cls_id))

    def goto_frame(self, idx):
        if not (0 <= idx < len(self.image_files)):
            return
        self.flush_changes()
        img = self.read_image_robust(self.image_files[idx])
        if img is None:
            self.status_bar.showMessage(f"Failed to load image: {self.image_files[idx].name}")
            return

        self.current_frame_idx = idx
        self.current_frame = img
        self.unsaved_changes = False
        self.load_label(idx)
        self.display_image()

        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)

        unlabeled_left = self.count_unlabeled()
        self.lbl_info.setText(f"Frame: {idx + 1} / {self.total_frames} | Unlabeled: {unlabeled_left}")
        self.update_window_title()
        self.update_fixed_box_log()

    def count_unlabeled(self):
        if not self.labels_dir:
            return 0
        missing = 0
        for i in range(len(self.image_files)):
            lbl = self.label_path_for_index(i)
            if not lbl.exists() or lbl.stat().st_size == 0:
                missing += 1
        return missing

    def collect_progress_stats(self):
        if not self.labels_dir:
            return 0, 0, 0
        annotated = 0
        non_empty = 0
        for i in range(len(self.image_files)):
            lbl = self.label_path_for_index(i)
            if lbl.exists():
                annotated += 1
                if lbl.stat().st_size > 0:
                    non_empty += 1
        return annotated, non_empty, len(self.image_files)

    def update_fixed_box_log(self):
        if not self.labels_dir:
            return

        annotated, non_empty, total = self.collect_progress_stats()
        fixed_enabled = self.chk_fixed_box.isChecked()
        payload = {
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "frames_dir": str(self.frames_dir) if self.frames_dir else "",
            "labels_dir": str(self.labels_dir),
            "current_frame_index": self.current_frame_idx,
            "current_frame_name": self.image_files[self.current_frame_idx].name if self.image_files else "",
            "fixed_box_enabled": fixed_enabled,
            "fixed_box_w": self.fixed_box_w.value() if fixed_enabled else "-",
            "fixed_box_h": self.fixed_box_h.value() if fixed_enabled else "-",
            "autosave_enabled": self.autosave_enabled,
            "total_frames": total,
            "annotated_label_files": annotated,
            "non_empty_label_files": non_empty,
            "unlabeled_frames": max(total - annotated, 0),
            "class_names": self.class_names,
        }
        log_path = self.labels_dir / "_annotator_session.json"
        log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def display_image(self):
        if self.current_frame is None:
            return
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setProperty("orig_w", w)
        self.image_label.setProperty("orig_h", h)
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.image_label.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.display_image()

    def find_next_unlabeled(self, start_idx=0):
        for idx in range(max(0, start_idx), len(self.image_files)):
            lbl = self.label_path_for_index(idx)
            if not lbl.exists() or lbl.stat().st_size == 0:
                return idx
        return None

    def find_prev_unlabeled(self, start_idx):
        for idx in range(min(start_idx, len(self.image_files) - 1), -1, -1):
            lbl = self.label_path_for_index(idx)
            if not lbl.exists() or lbl.stat().st_size == 0:
                return idx
        return None

    def goto_next_unlabeled(self):
        idx = self.find_next_unlabeled(self.current_frame_idx + 1)
        if idx is not None:
            self.goto_frame(idx)
        else:
            self.status_bar.showMessage("No next unlabeled frame found.")

    def goto_prev_unlabeled(self):
        idx = self.find_prev_unlabeled(self.current_frame_idx - 1)
        if idx is not None:
            self.goto_frame(idx)
        else:
            self.status_bar.showMessage("No previous unlabeled frame found.")

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()
        key_text = (event.text() or "").lower()

        # Ctrl+S / Ctrl+Ы
        if mods & Qt.KeyboardModifier.ControlModifier and (key == Qt.Key.Key_S or key_text == "ы"):
            self.save_current_label()
            return
        # Ctrl+Z / Ctrl+Я
        if mods & Qt.KeyboardModifier.ControlModifier and (key == Qt.Key.Key_Z or key_text == "я"):
            if self.image_label.bboxes:
                self.image_label.bboxes.pop()
                self.image_label.update()
                self.on_bbox_changed()
            return

        # next: Right/D/В/Space/Enter
        if key in (Qt.Key.Key_Right, Qt.Key.Key_D, Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter) or key_text == "в":
            if self.current_frame_idx < self.total_frames - 1:
                self.goto_frame(self.current_frame_idx + 1)
            return
        # prev: Left/A/Ф
        if key in (Qt.Key.Key_Left, Qt.Key.Key_A) or key_text == "ф":
            if self.current_frame_idx > 0:
                self.goto_frame(self.current_frame_idx - 1)
            return
        if key == Qt.Key.Key_Home:
            self.goto_frame(0)
            return
        if key == Qt.Key.Key_End:
            self.goto_frame(self.total_frames - 1)
            return
        # next unlabeled: N/Т
        if key == Qt.Key.Key_N or key_text == "т":
            self.goto_next_unlabeled()
            return
        # prev unlabeled: P/З
        if key == Qt.Key.Key_P or key_text == "з":
            self.goto_prev_unlabeled()
            return
        # clear all: C/С
        if key == Qt.Key.Key_C or key_text == "с":
            self.image_label.bboxes = []
            self.image_label.update()
            self.on_bbox_changed()
            return
        # remove last: U/Г/Delete/Backspace
        if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete, Qt.Key.Key_U) or key_text == "г":
            if self.image_label.bboxes:
                self.image_label.bboxes.pop()
                self.image_label.update()
                self.on_bbox_changed()
            return
        if key in (Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3, Qt.Key.Key_4):
            self.cmb_class.setCurrentIndex(int(event.text()) - 1)
            return

        super().keyPressEvent(event)


def main():
    app = QApplication(sys.argv)
    window = SimpleAnnotator()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
