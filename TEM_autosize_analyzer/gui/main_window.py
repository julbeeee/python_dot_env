from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
)

from core.analyzer import Analyzer, ImageResult, Thresholds
from core.io_utils import save_csv
from core.session import SessionState


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        ui_path = Path(__file__).with_name("main_window.ui")
        uic.loadUi(str(ui_path), self)

        icon_path = Path(__file__).resolve().parent.parent / "real_burger_icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self.analyzer = Analyzer()
        self.session = SessionState()
        self._pending_result: ImageResult | None = None
        self._pending_path: str | None = None

        self._wire_events()
        self._update_total_label()
        self._set_pending(None)

        self._init_histogram()
        self._update_stats_labels()
        self._update_image_stats(None)

    def _wire_events(self) -> None:
        self.btnAddImages.clicked.connect(self.on_add_images)
        self.btnClearList.clicked.connect(self.on_clear_list)
        self.btnAnalyze.clicked.connect(self.on_analyze_selected)
        self.btnAccept.clicked.connect(self.on_accept)
        self.btnSkip.clicked.connect(self.on_skip)
        self.btnExport.clicked.connect(self.on_export)
        self.listImages.currentItemChanged.connect(self.on_image_selected)
        self.spinHistBins.valueChanged.connect(self._on_hist_params_changed)
        self.spinHistMin.valueChanged.connect(self._on_hist_params_changed)
        self.spinHistMax.valueChanged.connect(self._on_hist_params_changed)
        self.checkHistRange.toggled.connect(self._on_hist_params_changed)

    def _on_hist_params_changed(self) -> None:
        self._update_histogram()
        self._update_stats_labels()

    def _init_histogram(self) -> None:
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
        except Exception as exc:
            self._log(f"Histogram disabled (matplotlib missing): {exc}")
            self._hist_canvas = None
            return

        fig = Figure(figsize=(4, 3))
        self._hist_canvas = FigureCanvas(fig)
        self._hist_ax = fig.add_subplot(111)
        self.histLayout.addWidget(self._hist_canvas)
        self._hist_ax.set_title("Total size distribution")
        self._hist_ax.set_xlabel("Diameter (nm)")
        self._hist_ax.set_ylabel("Counts")
        self._hist_canvas.draw()

    def _get_hist_params(self) -> tuple[int, float | None, float | None]:
        bins = int(self.spinHistBins.value())
        min_val = float(self.spinHistMin.value()) if self.checkHistRange.isChecked() else None
        max_val = float(self.spinHistMax.value()) if self.checkHistRange.isChecked() else None
        if min_val is not None and max_val is not None and max_val <= min_val:
            return bins, None, None
        return bins, min_val, max_val

    def _filter_values_by_range(
        self,
        values: list[float],
        min_val: float | None,
        max_val: float | None,
    ) -> list[float]:
        if min_val is None or max_val is None:
            return values
        return [v for v in values if min_val <= v <= max_val]

    def _update_histogram(self) -> None:
        if not getattr(self, "_hist_canvas", None):
            return
        values = self.session.total_nm
        self._hist_ax.clear()
        self._hist_ax.set_title("Total size distribution")
        self._hist_ax.set_xlabel("Diameter (nm)")
        self._hist_ax.set_ylabel("Counts")

        bins, min_val, max_val = self._get_hist_params()
        hist_range = None
        if min_val is not None and max_val is not None:
            hist_range = (min_val, max_val)

        display_values = self._filter_values_by_range(values, min_val, max_val)
        if display_values:
            self._hist_ax.hist(
                display_values,
                bins=bins,
                range=hist_range,
                color="#4c4c9d",
                edgecolor="black",
            )
        self._hist_canvas.draw()

    def _log(self, message: str) -> None:
        self.textLog.appendPlainText(message)

    def _update_total_label(self) -> None:
        self.labelTotal.setText(f"Total: {self.session.total_count()}")

    def _update_stats_labels(self) -> None:
        values = self.session.total_nm
        _, min_val, max_val = self._get_hist_params()
        values = self._filter_values_by_range(values, min_val, max_val)

        if not values:
            if min_val is not None and max_val is not None:
                self.labelMean.setText("Mean: - (filtered)")
                self.labelStd.setText("Std: - (filtered)")
            else:
                self.labelMean.setText("Mean: -")
                self.labelStd.setText("Std: -")
            return

        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        std = var ** 0.5
        suffix = " (filtered)" if min_val is not None and max_val is not None else ""
        self.labelMean.setText(f"Mean: {mean:.2f} nm{suffix}")
        self.labelStd.setText(f"Std: {std:.2f} nm{suffix}")

    def _update_image_stats(self, result: ImageResult | None) -> None:
        if result is None or result.count == 0:
            self.labelImgCount.setText("Image count: -")
            self.labelImgMean.setText("Image mean: -")
            self.labelImgStd.setText("Image std: -")
            return
        self.labelImgCount.setText(f"Image count: {result.count}")
        if result.mean_nm is None or result.std_nm is None:
            self.labelImgMean.setText("Image mean: -")
            self.labelImgStd.setText("Image std: -")
            return
        self.labelImgMean.setText(f"Image mean: {result.mean_nm:.2f} nm")
        self.labelImgStd.setText(f"Image std: {result.std_nm:.2f} nm")

    def _selected_image_path(self) -> str | None:
        item = self.listImages.currentItem()
        if item is None:
            return None
        return item.data(Qt.UserRole)

    def _set_image_list(self, paths: Iterable[str]) -> None:
        for path in paths:
            self._add_image_item(path)

    def _add_image_item(self, path: str) -> None:
        name = Path(path).name
        self.listImages.addItem(name)
        list_item = self.listImages.item(self.listImages.count() - 1)
        list_item.setData(Qt.UserRole, path)

    def _get_thresholds(self) -> Thresholds:
        return Thresholds(
            min_nm=self.spinMinNm.value(),
            max_nm=self.spinMaxNm.value(),
            ecc_max=self.spinEcc.value(),
            sol_min=self.spinSol.value(),
        )

    def _get_scale_bar_color(self) -> str:
        return "white" if self.radioWhite.isChecked() else "black"

    def _set_pending(self, result: ImageResult | None, path: str | None = None) -> None:
        self._pending_result = result
        self._pending_path = path
        if result is None:
            self.btnAccept.setEnabled(False)
            self.btnSkip.setEnabled(False)
        else:
            self.btnAccept.setEnabled(True)
            self.btnSkip.setEnabled(True)

    def _draw_overlay(self, pixmap: QPixmap, result: ImageResult) -> QPixmap:
        overlay = QPixmap(pixmap)
        painter = QPainter(overlay)
        pen = QPen(QColor(255, 40, 40))
        pen.setWidth(4)
        painter.setPen(pen)

        for cx, cy, r in result.circles:
            painter.drawEllipse(int(cx - r), int(cy - r), int(2 * r), int(2 * r))

        painter.end()
        return overlay

    def _show_preview(self, path: str, result: ImageResult | None = None) -> None:
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.labelPreview.setText("Failed to load image.")
            self.labelPreview.setPixmap(QPixmap())
            return

        if result is not None and result.circles:
            pixmap = self._draw_overlay(pixmap, result)

        scaled = pixmap.scaled(
            self.labelPreview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.labelPreview.setPixmap(scaled)
        self.labelPreview.setAlignment(Qt.AlignCenter)

    def on_add_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select images",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
        if not paths:
            return
        self._set_image_list(paths)
        self._log(f"Added {len(paths)} image(s).")

    def on_clear_list(self) -> None:
        self.listImages.clear()
        self._set_pending(None)
        self._update_image_stats(None)
        self.labelPreview.setText("Image preview will appear here.")
        self.labelPreview.setPixmap(QPixmap())
        self._log("Cleared image list.")

    def on_image_selected(self) -> None:
        path = self._selected_image_path()
        if path:
            self._show_preview(path)

    def on_analyze_selected(self) -> None:
        path = self._selected_image_path()
        if not path:
            QMessageBox.warning(self, "No selection", "Select an image first.")
            return

        default_nm = self.spinSbDefault.value()
        value, ok = QInputDialog.getDouble(
            self,
            "Scale bar length",
            f"Scale bar length for {Path(path).name} (nm)",
            default_nm,
            0.1,
            10000.0,
            1,
        )
        if not ok:
            self._log("Scale bar input canceled.")
            return

        thresholds = self._get_thresholds()
        sb_color = self._get_scale_bar_color()
        self._log(
            f"Analyze: {Path(path).name} | sb={value}nm | color={sb_color} | "
            f"min={thresholds.min_nm}, max={thresholds.max_nm}, ecc<={thresholds.ecc_max}, sol>={thresholds.sol_min}"
        )

        try:
            result = self.analyzer.analyze_image(path, value, thresholds, sb_color=sb_color)
        except Exception as exc:
            self._set_pending(None)
            self._update_image_stats(None)
            QMessageBox.critical(self, "Analyze failed", str(exc))
            self._log(f"Error: {exc}")
            return

        self._set_pending(result, path)
        self._show_preview(path, result)
        self._update_image_stats(result)
        self._log(
            f"Result: count={result.count}, mean={result.mean_nm}, std={result.std_nm}"
        )

    def on_accept(self) -> None:
        if self._pending_result is None:
            QMessageBox.information(self, "Nothing to accept", "Run analysis first.")
            return

        self.session.add_measurements(self._pending_result.values_nm)
        self._log(f"Accepted {self._pending_result.count} dots.")
        self._set_pending(None)
        self._update_total_label()
        self._update_stats_labels()
        self._update_histogram()

    def on_skip(self) -> None:
        if self._pending_result is None:
            return
        self._log("Skipped current image.")
        self._set_pending(None)
        self._update_image_stats(None)
        if self._pending_path:
            self._show_preview(self._pending_path)

    def on_export(self) -> None:
        if self.session.total_count() == 0:
            QMessageBox.information(self, "No data", "No accumulated data to export.")
            return

        _, min_val, max_val = self._get_hist_params()
        export_values = self.session.total_nm
        filtered = False
        if min_val is not None and max_val is not None:
            export_values = self._filter_values_by_range(export_values, min_val, max_val)
            filtered = True

        if not export_values:
            QMessageBox.information(self, "No data", "No data in the selected histogram range.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV",
            "QD_size_results.csv",
            "CSV Files (*.csv)"
        )
        if not path:
            return

        result = save_csv(export_values, path)
        if filtered:
            self._log(
                f"Saved CSV (filtered): {result.path} ({result.count} entries)"
            )
        else:
            self._log(f"Saved CSV: {result.path} ({result.count} entries)")
        QMessageBox.information(self, "Saved", f"Saved {result.count} rows.")
