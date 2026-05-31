from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Literal, Optional
import csv
import json
import math

import numpy as np
from PIL import Image, ImageChops

try:
    import cv2  # used for flood-fill background removal and angle analysis
except Exception:  # pragma: no cover
    cv2 = None

Image.MAX_IMAGE_PIXELS = None

BackgroundMode = Literal["white", "auto_corner", "custom"]
AlignmentMode = Literal["center", "bottom-center"]

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class ProcessingSettings:
    skip_background_removal: bool = False
    background_mode: BackgroundMode = "auto_corner"
    tolerance: int = 25
    custom_bg_hex: str = "#ffffff"
    feather_edges: bool = True
    feather_radius: float = 0.6
    crop_transparent: bool = True
    crop_padding: int = 20
    scale_x: float = 1.0
    scale_y: float = 1.0
    align_to_isometric: bool = False
    iso_target_ratio: float = 0.5
    iso_max_side_diff: float = 0.1
    iso_use_virtual_bottom_corner: bool = True
    canvas_width: int = 1024
    canvas_height: int = 1024
    alignment: AlignmentMode = "bottom-center"
    bottom_margin: int = 40
    preserve_if_too_large: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> "ProcessingSettings":
        payload = json.loads(data)
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {k: v for k, v in payload.items() if k in allowed}
        return cls(**clean)


@dataclass
class ProcessResult:
    filename: str
    status: str
    original_width: int = 0
    original_height: int = 0
    processed_width: int = 0
    processed_height: int = 0
    final_width: int = 0
    final_height: int = 0
    scale_x: float = 1.0
    scale_y: float = 1.0
    detected_angle: Optional[float] = None
    suggested_scale_y: Optional[float] = None
    iso_auto_scale_applied: bool = False
    iso_auto_scale_y: Optional[float] = None
    iso_pre_right_ratio: Optional[float] = None
    iso_pre_left_ratio: Optional[float] = None
    iso_pre_avg_ratio: Optional[float] = None
    iso_pre_side_diff: Optional[float] = None
    iso_align_skipped_reason: str = ""
    iso_right_ratio: Optional[float] = None
    iso_left_ratio: Optional[float] = None
    iso_avg_ratio: Optional[float] = None
    iso_side_diff: Optional[float] = None
    iso_right_abs_ratio: Optional[float] = None
    iso_left_abs_ratio: Optional[float] = None
    iso_measurement_method: str = ""
    iso_right_point: str = ""
    iso_left_point: str = ""
    iso_bottom_point: str = ""
    output_path: str = ""
    error: str = ""


def list_images(folder: str | Path) -> list[Path]:
    path = Path(folder).expanduser()
    if not path.exists() or not path.is_dir():
        return []
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS])


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def load_image(path: str | Path) -> Image.Image:
    img = Image.open(path)
    # Handle EXIF orientation where possible
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img.convert("RGBA")


def estimate_corner_background(image: Image.Image, sample_size: int = 12) -> tuple[int, int, int]:
    img = image.convert("RGBA")
    arr = np.array(img)
    h, w = arr.shape[:2]
    s = max(1, min(sample_size, h, w))
    patches = [
        arr[0:s, 0:s, :3],
        arr[0:s, w - s : w, :3],
        arr[h - s : h, 0:s, :3],
        arr[h - s : h, w - s : w, :3],
    ]
    pixels = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0)
    rgb = np.median(pixels, axis=0)
    return tuple(int(x) for x in rgb)


def _outer_background_mask(candidate_mask: np.ndarray, connectivity: int = 4) -> np.ndarray:
    """Return only background pixels that are connected to the image border.

    Uses connected-components on the candidate mask with a padded border so that
    all edge-touching background regions are found in one pass.  Interior white
    areas that are fully surrounded by the subject are NOT included.
    """
    if cv2 is not None:
        mask_uint8 = candidate_mask.astype(np.uint8) * 255
        # One-pixel border of "background" ensures connectivity from every edge pixel.
        padded = cv2.copyMakeBorder(mask_uint8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
        _, labels = cv2.connectedComponents(padded, connectivity=connectivity)
        outer_label = int(labels[0, 0])  # always part of the padded border
        return (labels[1:-1, 1:-1] == outer_label)

    # Fallback: BFS from border pixels when cv2 is unavailable
    from collections import deque
    h, w = candidate_mask.shape
    visited = np.zeros_like(candidate_mask, dtype=bool)
    queue: deque = deque()
    for y in range(h):
        for x in (0, w - 1):
            if candidate_mask[y, x] and not visited[y, x]:
                visited[y, x] = True
                queue.append((y, x))
    for x in range(w):
        for y in (0, h - 1):
            if candidate_mask[y, x] and not visited[y, x]:
                visited[y, x] = True
                queue.append((y, x))
    while queue:
        cy, cx = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and candidate_mask[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))
    return visited


def remove_background(
    image: Image.Image,
    mode: BackgroundMode = "auto_corner",
    tolerance: int = 25,
    custom_bg_hex: str = "#ffffff",
    feather_edges: bool = True,
    feather_radius: float = 0.6,
) -> Image.Image:
    img = image.convert("RGBA")
    arr = np.array(img).astype(np.int16)

    if mode == "white":
        bg = np.array([255, 255, 255], dtype=np.int16)
    elif mode == "custom":
        bg = np.array(hex_to_rgb(custom_bg_hex), dtype=np.int16)
    else:
        bg = np.array(estimate_corner_background(img), dtype=np.int16)

    rgb = arr[..., :3].astype(np.float32)
    alpha = arr[..., 3].astype(np.uint8)
    distance = np.sqrt(np.sum((rgb - bg.astype(np.float32)) ** 2, axis=-1))

    # tolerance is user-friendly; RGB euclidean distance needs a wider internal threshold
    threshold = max(1, int(tolerance)) * 1.8

    # Only remove background pixels reachable from the image border (flood fill).
    # This prevents interior white areas of the subject from becoming transparent.
    candidate_bg = distance <= threshold
    outer_bg = _outer_background_mask(candidate_bg)
    alpha[outer_bg] = 0

    # Keep semi-transparent edge transition instead of jagged hard cut.
    # Extend the flood-fill region to cover the soft feathering zone as well.
    if feather_edges and tolerance > 0:
        soft_threshold = threshold * 1.7
        soft_zone_candidates = (distance > threshold) & (distance <= soft_threshold)
        if np.any(soft_zone_candidates):
            outer_soft = _outer_background_mask(candidate_bg | soft_zone_candidates)
            actual_soft = outer_soft & soft_zone_candidates & ~outer_bg
            if np.any(actual_soft):
                ramp = (distance[actual_soft] - threshold) / max(1, soft_threshold - threshold)
                alpha[actual_soft] = np.minimum(alpha[actual_soft], (ramp * 255).astype(np.uint8))

    out = np.array(img)
    out[..., 3] = alpha
    result = Image.fromarray(out, "RGBA")

    # Very light blur on alpha mask only; helps remove white halo around AI-generated assets
    if feather_edges and feather_radius > 0:
        from PIL import ImageFilter

        r, g, b, a = result.split()
        a = a.filter(ImageFilter.GaussianBlur(radius=float(feather_radius)))
        result = Image.merge("RGBA", (r, g, b, a))

    return result


def crop_transparent(image: Image.Image, padding: int = 20) -> Image.Image:
    img = image.convert("RGBA")
    alpha = img.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        return img
    left, upper, right, lower = bbox
    left = max(0, left - padding)
    upper = max(0, upper - padding)
    right = min(img.width, right + padding)
    lower = min(img.height, lower + padding)
    return img.crop((left, upper, right, lower))


def scale_image(image: Image.Image, scale_x: float = 1.0, scale_y: float = 1.0) -> Image.Image:
    img = image.convert("RGBA")
    new_w = max(1, int(round(img.width * float(scale_x))))
    new_h = max(1, int(round(img.height * float(scale_y))))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def fit_inside_canvas(image: Image.Image, canvas_width: int, canvas_height: int) -> Image.Image:
    if image.width <= canvas_width and image.height <= canvas_height:
        return image
    ratio = min(canvas_width / image.width, canvas_height / image.height)
    new_w = max(1, int(image.width * ratio))
    new_h = max(1, int(image.height * ratio))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def place_on_canvas(
    image: Image.Image,
    canvas_width: int = 1024,
    canvas_height: int = 1024,
    alignment: AlignmentMode = "bottom-center",
    bottom_margin: int = 40,
    preserve_if_too_large: bool = False,
) -> Image.Image:
    img = image.convert("RGBA")
    if not preserve_if_too_large:
        img = fit_inside_canvas(img, canvas_width, canvas_height)

    canvas = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
    x = (canvas_width - img.width) // 2
    if alignment == "bottom-center":
        y = canvas_height - img.height - int(bottom_margin)
    else:
        y = (canvas_height - img.height) // 2

    # If the image is larger than canvas and preserve_if_too_large=True, allow crop-like placement
    canvas.alpha_composite(img, (int(x), int(y)))
    return canvas


def process_image(image: Image.Image, settings: ProcessingSettings) -> Image.Image:
    if settings.skip_background_removal:
        out = image.convert("RGBA")
    else:
        out = remove_background(
            image,
            mode=settings.background_mode,
            tolerance=settings.tolerance,
            custom_bg_hex=settings.custom_bg_hex,
            feather_edges=settings.feather_edges,
            feather_radius=settings.feather_radius,
        )
    if settings.crop_transparent:
        out = crop_transparent(out, settings.crop_padding)
    out = scale_image(out, settings.scale_x, settings.scale_y)
    if settings.align_to_isometric:
        out, _, _, _ = apply_isometric_alignment(
            out,
            settings.iso_target_ratio,
            settings.iso_max_side_diff,
            settings.iso_use_virtual_bottom_corner,
        )
    out = place_on_canvas(
        out,
        settings.canvas_width,
        settings.canvas_height,
        settings.alignment,
        settings.bottom_margin,
        settings.preserve_if_too_large,
    )
    return out


def _median_extreme_point(xs: np.ndarray, ys: np.ndarray, axis: str, value: int) -> tuple[int, int]:
    if axis == "x":
        matches = xs == value
        return int(value), int(round(float(np.median(ys[matches]))))
    if axis == "y":
        matches = ys == value
        return int(round(float(np.median(xs[matches])))), int(value)
    raise ValueError(f"Unknown axis: {axis}")


def _line_from_points(points: np.ndarray) -> Optional[tuple[float, float]]:
    if len(points) < 8:
        return None
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    if float(x.max() - x.min()) < 8:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _fit_lower_edge(points: np.ndarray, expected_slope_sign: int) -> Optional[tuple[float, float]]:
    if len(points) < 12:
        return None

    candidates: list[tuple[float, float, float]] = []
    for percentile in (35, 45, 55, 65):
        if len(points) < 12:
            break
        cutoff = float(np.percentile(points[:, 1], percentile))
        sample = points[points[:, 1] >= cutoff]
        line = _line_from_points(sample)
        if line is None:
            continue
        slope, intercept = line
        if expected_slope_sign > 0 and slope <= 0:
            continue
        if expected_slope_sign < 0 and slope >= 0:
            continue
        residual = float(np.median(np.abs(sample[:, 1] - (slope * sample[:, 0] + intercept))))
        candidates.append((residual, slope, intercept))

    if not candidates:
        return None
    _, slope, intercept = min(candidates, key=lambda item: item[0])
    return slope, intercept


def _find_bottom_corners(
    alpha_raw: np.ndarray,
    solid_alpha_threshold: int = 200,
    y_tolerance: int = 2,
) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
    """Return (left_bottom, right_bottom) anchor points for isometric slope measurement.

    Applies solid_alpha_threshold first to exclude semi-transparent edge pixels, then
    collects all solid pixels within y_tolerance rows of the bottommost solid pixel.
    The leftmost and rightmost of those pixels become the two bottom anchors.
    """
    solid = alpha_raw >= solid_alpha_threshold
    ys, xs = np.nonzero(solid)
    if len(xs) == 0:
        return None
    max_y = int(ys.max())
    row_indices = np.arange(alpha_raw.shape[0])
    band_mask = solid & (row_indices[:, None] >= max_y - y_tolerance)
    band_ys, band_xs = np.nonzero(band_mask)
    if len(band_xs) == 0:
        return None
    left_idx = int(np.argmin(band_xs))
    right_idx = int(np.argmax(band_xs))
    return (int(band_xs[left_idx]), int(band_ys[left_idx])), (int(band_xs[right_idx]), int(band_ys[right_idx]))


def _virtual_bottom_point(alpha: np.ndarray) -> Optional[tuple[int, int]]:
    ys, xs = np.nonzero(alpha)
    if len(xs) == 0:
        return None

    left = _median_extreme_point(xs, ys, "x", int(xs.min()))
    right = _median_extreme_point(xs, ys, "x", int(xs.max()))
    physical_bottom = _median_extreme_point(xs, ys, "y", int(ys.max()))

    lower_by_x: list[tuple[int, int]] = []
    for x in np.unique(xs):
        col_ys = ys[xs == x]
        lower_by_x.append((int(x), int(col_ys.max())))
    outline = np.array(lower_by_x, dtype=np.int32)

    min_y = int(ys.min())
    max_y = int(ys.max())
    lower_cutoff = min_y + int((max_y - min_y) * 0.55)
    outline = outline[outline[:, 1] >= lower_cutoff]
    if len(outline) < 24:
        return None

    width = max(1, right[0] - left[0])
    gap = max(6, int(width * 0.04))
    left_points = outline[(outline[:, 0] < physical_bottom[0] - gap) & (outline[:, 0] > left[0] + gap)]
    right_points = outline[(outline[:, 0] > physical_bottom[0] + gap) & (outline[:, 0] < right[0] - gap)]

    left_line = _fit_lower_edge(left_points, expected_slope_sign=1)
    right_line = _fit_lower_edge(right_points, expected_slope_sign=-1)
    if left_line is None or right_line is None:
        return None

    m_left, b_left = left_line
    m_right, b_right = right_line
    denom = m_left - m_right
    if abs(denom) < 1e-6:
        return None

    x_intersection = (b_right - b_left) / denom
    y_intersection = m_left * x_intersection + b_left
    if not math.isfinite(x_intersection) or not math.isfinite(y_intersection):
        return None

    margin = width * 0.25
    if x_intersection < left[0] - margin or x_intersection > right[0] + margin:
        return None
    if y_intersection < physical_bottom[1] - (max_y - min_y) * 0.25:
        return None

    return int(round(x_intersection)), int(round(y_intersection))


def measure_isometric_ratios(
    image: Image.Image,
    alpha_threshold: int = 10,
    use_virtual_bottom_corner: bool = True,
) -> dict[str, Optional[float] | str]:
    """Measure output alpha silhouette against a 1:2 isometric y/x ratio.

    Uses separate left-bottom and right-bottom anchor points so that flat-bottomed
    isometric shapes (e.g. large sports fields) produce accurate per-side slopes.
    For pointed-tip shapes the two anchors converge to the same point, preserving
    the original behaviour.  Virtual-bottom extrapolation is only applied when the
    two anchors are nearly coincident (pointed or near-pointed tips).
    """
    alpha_raw = np.array(image.convert("RGBA").getchannel("A"))
    alpha = alpha_raw > int(alpha_threshold)
    ys, xs = np.nonzero(alpha)
    if len(xs) == 0:
        return {
            "iso_right_ratio": None,
            "iso_left_ratio": None,
            "iso_right_abs_ratio": None,
            "iso_left_abs_ratio": None,
            "iso_avg_ratio": None,
            "iso_side_diff": None,
            "iso_measurement_method": "",
            "iso_right_point": "",
            "iso_left_point": "",
            "iso_bottom_point": "",
        }

    left = _median_extreme_point(xs, ys, "x", int(xs.min()))
    right = _median_extreme_point(xs, ys, "x", int(xs.max()))

    # Determine per-side bottom anchors using solid-pixel filtering.
    corners = _find_bottom_corners(alpha_raw, solid_alpha_threshold=200, y_tolerance=2)
    if corners is not None:
        left_bottom, right_bottom = corners
    else:
        single = _median_extreme_point(xs, ys, "y", int(ys.max()))
        left_bottom = right_bottom = single

    # Decide if the bottom is genuinely flat (wide gap between the two anchors).
    width = max(1, int(xs.max()) - int(xs.min()))
    flat_bottom = abs(right_bottom[0] - left_bottom[0]) > max(6, int(width * 0.03))

    method = "dual-bottom" if flat_bottom else "physical-bottom"

    # Virtual-bottom extrapolation is only meaningful for non-flat (pointed/chopped) tips.
    if not flat_bottom and use_virtual_bottom_corner:
        virtual = _virtual_bottom_point(alpha)
        if virtual is not None:
            left_bottom = right_bottom = virtual
            method = "virtual-bottom"

    def ratio(corner: tuple[int, int], bottom: tuple[int, int]) -> Optional[float]:
        dx = corner[0] - bottom[0]
        if dx == 0:
            return None
        return round((corner[1] - bottom[1]) / dx, 4)

    right_ratio = ratio(right, right_bottom)
    left_ratio = ratio(left, left_bottom)
    right_abs = round(abs(right_ratio), 4) if right_ratio is not None else None
    left_abs = round(abs(left_ratio), 4) if left_ratio is not None else None
    avg_ratio = round((right_abs + left_abs) / 2, 4) if right_abs is not None and left_abs is not None else None
    side_diff = round(abs(right_abs - left_abs), 4) if right_abs is not None and left_abs is not None else None

    if flat_bottom:
        bottom_str = f"L({left_bottom[0]},{left_bottom[1]})|R({right_bottom[0]},{right_bottom[1]})"
    else:
        bottom_str = f"({left_bottom[0]},{left_bottom[1]})"

    return {
        "iso_right_ratio": right_ratio,
        "iso_left_ratio": left_ratio,
        "iso_right_abs_ratio": right_abs,
        "iso_left_abs_ratio": left_abs,
        "iso_avg_ratio": avg_ratio,
        "iso_side_diff": side_diff,
        "iso_measurement_method": method,
        "iso_right_point": f"({right[0]},{right[1]})",
        "iso_left_point": f"({left[0]},{left[1]})",
        "iso_bottom_point": bottom_str,
    }


def isometric_y_scale_factor(
    measurements: dict[str, Optional[float] | str],
    target_ratio: float = 0.5,
    max_side_diff: float = 0.1,
) -> tuple[Optional[float], str]:
    avg_ratio = measurements.get("iso_avg_ratio")
    side_diff = measurements.get("iso_side_diff")
    if not isinstance(avg_ratio, (int, float)) or avg_ratio <= 0:
        return None, "cannot measure average ratio"
    if not isinstance(side_diff, (int, float)):
        return None, "cannot measure side difference"
    if side_diff > max_side_diff:
        return None, f"left/right difference {side_diff:.4f} exceeds {max_side_diff:.4f}"
    return round(float(target_ratio) / float(avg_ratio), 4), ""


def apply_isometric_alignment(
    image: Image.Image,
    target_ratio: float = 0.5,
    max_side_diff: float = 0.1,
    use_virtual_bottom_corner: bool = True,
) -> tuple[Image.Image, dict[str, Optional[float] | str], Optional[float], str]:
    measurements = measure_isometric_ratios(image, use_virtual_bottom_corner=use_virtual_bottom_corner)
    factor, reason = isometric_y_scale_factor(measurements, target_ratio, max_side_diff)
    if factor is None:
        return image, measurements, None, reason
    return scale_image(image, 1.0, factor), measurements, factor, ""


def make_before_after(before: Image.Image, after: Image.Image, checker: bool = True) -> Image.Image:
    before = before.convert("RGBA")
    after = after.convert("RGBA")
    target_h = 360

    def resize_keep(img: Image.Image) -> Image.Image:
        ratio = target_h / img.height
        return img.resize((max(1, int(img.width * ratio)), target_h), Image.Resampling.LANCZOS)

    b = resize_keep(before)
    a = resize_keep(after)
    gap = 20
    canvas = Image.new("RGBA", (b.width + a.width + gap, target_h), (245, 245, 245, 255))
    canvas.alpha_composite(b, (0, 0))
    bg = checkerboard(a.size) if checker else Image.new("RGBA", a.size, (245, 245, 245, 255))
    bg.alpha_composite(a, (0, 0))
    canvas.alpha_composite(bg, (b.width + gap, 0))
    return canvas


def checkerboard(size: tuple[int, int], block: int = 16) -> Image.Image:
    w, h = size
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            v = 220 if ((x // block + y // block) % 2 == 0) else 250
            arr[y, x] = [v, v, v, 255]
    return Image.fromarray(arr, "RGBA")


def detect_base_angle(image: Image.Image) -> tuple[Optional[float], Optional[float]]:
    """Best-effort angle detection. Returns average angle and suggested scale_y.

    It looks at the lower 40% of the alpha mask and searches for slanted lines.
    This is intentionally conservative. Bad AI shadows / missing base corners may fail.
    """
    if cv2 is None:
        return None, None

    img = image.convert("RGBA")
    alpha = np.array(img.getchannel("A"))
    if alpha.max() == 0:
        return None, None

    h, w = alpha.shape
    roi = alpha[int(h * 0.55) :, :]
    _, binary = cv2.threshold(roi, 10, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    min_len = max(20, int(w * 0.08))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35, minLineLength=min_len, maxLineGap=12)
    if lines is None:
        return None, None

    angles: list[float] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, line)
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 2:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx)))
        if angle > 90:
            angle = 180 - angle
        # Keep plausible isometric-ish slants only
        if 12 <= angle <= 45:
            angles.append(angle)

    if not angles:
        return None, None

    current = float(np.median(angles))
    target = 26.565051
    if current <= 0:
        return current, None
    suggested_scale_y = math.tan(math.radians(target)) / math.tan(math.radians(current))
    return round(current, 2), round(suggested_scale_y, 4)


def process_file(path: Path, output_dir: Path, settings: ProcessingSettings, analyze_angle: bool = False) -> ProcessResult:
    result = ProcessResult(filename=path.name, status="failed", scale_x=settings.scale_x, scale_y=settings.scale_y)
    try:
        img = load_image(path)
        result.original_width = img.width
        result.original_height = img.height

        if settings.skip_background_removal:
            intermediate = img.convert("RGBA")
        else:
            intermediate = remove_background(
                img,
                settings.background_mode,
                settings.tolerance,
                settings.custom_bg_hex,
                settings.feather_edges,
                settings.feather_radius,
            )
        if settings.crop_transparent:
            intermediate = crop_transparent(intermediate, settings.crop_padding)
        intermediate = scale_image(intermediate, settings.scale_x, settings.scale_y)

        if settings.align_to_isometric:
            intermediate, pre_measurements, factor, reason = apply_isometric_alignment(
                intermediate,
                settings.iso_target_ratio,
                settings.iso_max_side_diff,
                settings.iso_use_virtual_bottom_corner,
            )
            result.iso_pre_right_ratio = pre_measurements["iso_right_abs_ratio"]  # type: ignore[assignment]
            result.iso_pre_left_ratio = pre_measurements["iso_left_abs_ratio"]  # type: ignore[assignment]
            result.iso_pre_avg_ratio = pre_measurements["iso_avg_ratio"]  # type: ignore[assignment]
            result.iso_pre_side_diff = pre_measurements["iso_side_diff"]  # type: ignore[assignment]
            if factor is None:
                result.iso_align_skipped_reason = reason
            else:
                result.iso_auto_scale_applied = True
                result.iso_auto_scale_y = factor

        result.processed_width = intermediate.width
        result.processed_height = intermediate.height

        if analyze_angle:
            angle, suggested = detect_base_angle(intermediate)
            result.detected_angle = angle
            result.suggested_scale_y = suggested

        final = place_on_canvas(
            intermediate,
            settings.canvas_width,
            settings.canvas_height,
            settings.alignment,
            settings.bottom_margin,
            settings.preserve_if_too_large,
        )
        iso_measurements = measure_isometric_ratios(
            final,
            use_virtual_bottom_corner=settings.iso_use_virtual_bottom_corner,
        )
        result.iso_right_ratio = iso_measurements["iso_right_ratio"]  # type: ignore[assignment]
        result.iso_left_ratio = iso_measurements["iso_left_ratio"]  # type: ignore[assignment]
        result.iso_right_abs_ratio = iso_measurements["iso_right_abs_ratio"]  # type: ignore[assignment]
        result.iso_left_abs_ratio = iso_measurements["iso_left_abs_ratio"]  # type: ignore[assignment]
        result.iso_avg_ratio = iso_measurements["iso_avg_ratio"]  # type: ignore[assignment]
        result.iso_side_diff = iso_measurements["iso_side_diff"]  # type: ignore[assignment]
        result.iso_measurement_method = str(iso_measurements["iso_measurement_method"])
        result.iso_right_point = str(iso_measurements["iso_right_point"])
        result.iso_left_point = str(iso_measurements["iso_left_point"])
        result.iso_bottom_point = str(iso_measurements["iso_bottom_point"])
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{path.stem}_fixed.png"
        final.save(out_path)
        result.final_width = final.width
        result.final_height = final.height
        result.output_path = str(out_path)
        result.status = "success"
        return result
    except Exception as exc:
        result.error = str(exc)
        return result


def process_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    settings: ProcessingSettings,
    analyze_angle: bool = False,
) -> list[ProcessResult]:
    paths = list_images(input_dir)
    out_dir = Path(output_dir).expanduser()
    results = [process_file(p, out_dir, settings, analyze_angle=analyze_angle) for p in paths]
    write_log(results, out_dir / "process_log.csv")
    (out_dir / "settings.json").write_text(settings.to_json(), encoding="utf-8")
    return results


def write_log(results: Iterable[ProcessResult], csv_path: str | Path) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in results]
    if not rows:
        rows = []
    fields = list(ProcessResult.__dataclass_fields__.keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
