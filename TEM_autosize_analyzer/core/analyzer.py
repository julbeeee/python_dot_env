from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import color, exposure, filters, io, measure, morphology
from scipy import ndimage as ndi


@dataclass
class Thresholds:
    min_nm: float = 3.5
    max_nm: float = 9.0
    ecc_max: float = 0.7
    sol_min: float = 0.5


@dataclass
class AnalyzerSettings:
    sb_color: str = "white"  # "white" or "black"
    sb_region_ratio: float = 0.85
    white_thr: int = 245
    black_thr: int = 10
    sb_min_area: int = 200
    crop_bottom_px: int = 150
    gamma: float = 0.7
    gauss_sigma: float = 1.0
    adapthist_clip: float = 0.02
    adapthist_tiles: tuple[int, int] = (4, 4)


@dataclass
class ImageResult:
    count: int
    mean_nm: float | None
    std_nm: float | None
    values_nm: list[float]
    circles: list[tuple[float, float, float]]  # (x, y, r) in image pixels


class Analyzer:
    def __init__(self, settings: AnalyzerSettings | None = None) -> None:
        self.settings = settings or AnalyzerSettings()

    def analyze_image(
        self,
        image_path: str,
        known_nm: float,
        thresholds: Thresholds,
        sb_color: str | None = None,
    ) -> ImageResult:
        settings = self.settings
        sb_color = (sb_color or settings.sb_color).lower()

        image = io.imread(image_path)
        if image.ndim == 3:
            gray = color.rgb2gray(image)
        else:
            gray = image.astype(np.float32) / 255.0

        h_img = gray.shape[0]
        sb_start = int(h_img * settings.sb_region_ratio)
        sb_region = gray[sb_start:, :]

        if sb_color == "white":
            thr = settings.white_thr / 255.0
            sb_bw = sb_region >= thr
        elif sb_color == "black":
            thr = settings.black_thr / 255.0
            sb_bw = sb_region <= thr
        else:
            raise ValueError("Scale bar color must be 'white' or 'black'.")

        sb_bw = morphology.remove_small_objects(sb_bw, settings.sb_min_area)
        sb_labels = measure.label(sb_bw)
        sb_props = measure.regionprops(sb_labels)
        if not sb_props:
            raise ValueError("Scale bar not found. Adjust threshold or region.")

        sb_props.sort(key=lambda p: p.area, reverse=True)
        minr, minc, maxr, maxc = sb_props[0].bbox
        bar_width_px = maxc - minc
        if bar_width_px <= 0:
            raise ValueError("Invalid scale bar width.")

        nm_per_pix = known_nm / bar_width_px

        crop_bottom = settings.crop_bottom_px
        if h_img - crop_bottom <= 1:
            raise ValueError("Crop bottom too large for this image.")
        roi = gray[: h_img - crop_bottom, :]

        roi_eq = exposure.equalize_adapthist(
            roi,
            clip_limit=settings.adapthist_clip,
            nbins=256,
        )
        roi_gamma = exposure.adjust_gamma(roi_eq, settings.gamma)
        roi_f = filters.gaussian(roi_gamma, sigma=settings.gauss_sigma)

        level = filters.threshold_otsu(roi_f)
        bw = roi_f > level
        bw = np.logical_not(bw)

        bw = morphology.remove_small_objects(bw, 30)
        bw = ndi.binary_fill_holes(bw)

        labels = measure.label(bw)
        props = measure.regionprops(labels)

        if not props:
            return ImageResult(count=0, mean_nm=None, std_nm=None, values_nm=[], circles=[])

        equiv_diam = np.array([p.equivalent_diameter for p in props], dtype=float)
        ecc = np.array([p.eccentricity for p in props], dtype=float)
        sol = np.array([p.solidity for p in props], dtype=float)

        d_nm = equiv_diam * nm_per_pix

        mask_size = (d_nm >= thresholds.min_nm) & (d_nm <= thresholds.max_nm)
        mask_shape = (ecc <= thresholds.ecc_max) & (sol >= thresholds.sol_min)
        valid = mask_size & mask_shape

        values = d_nm[valid]
        if values.size == 0:
            return ImageResult(count=0, mean_nm=None, std_nm=None, values_nm=[], circles=[])

        mean_nm = float(np.mean(values))
        std_nm = float(np.std(values))

        circles: list[tuple[float, float, float]] = []
        for prop, keep in zip(props, valid):
            if not keep:
                continue
            cy, cx = prop.centroid
            r = prop.equivalent_diameter / 2.0
            circles.append((float(cx), float(cy), float(r)))

        return ImageResult(
            count=int(values.size),
            mean_nm=mean_nm,
            std_nm=std_nm,
            values_nm=values.tolist(),
            circles=circles,
        )
