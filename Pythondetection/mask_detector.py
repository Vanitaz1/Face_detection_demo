# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
WINDOW_NAME = "OpenCV Mask Detector"
FONT_PATHS = [
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/simsun.ttc"),
]
FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}
THEMES = {
    "dark": {
        "name": "\u6df1\u8272",
        "panel": (18, 22, 28),
        "panel_border": (70, 78, 88),
        "text": (245, 248, 252),
        "muted": (175, 182, 192),
        "camera_off": (24, 28, 34),
    },
    "light": {
        "name": "\u6d45\u8272",
        "panel": (236, 240, 245),
        "panel_border": (180, 188, 198),
        "text": (26, 32, 40),
        "muted": (82, 92, 105),
        "camera_off": (226, 231, 238),
    },
    "contrast": {
        "name": "\u9ad8\u5bf9\u6bd4",
        "panel": (0, 0, 0),
        "panel_border": (0, 255, 255),
        "text": (255, 255, 255),
        "muted": (0, 255, 255),
        "camera_off": (0, 0, 0),
    },
}
THEME_ORDER = ["dark", "light", "contrast"]
CURRENT_THEME = "dark"


@dataclass
class DetectionResult:
    label: str
    color: tuple[int, int, int]
    skin_ratio: float
    mouth_found: bool
    confidence: float = 0.0
    method: str = "rule"
    obstruction_found: bool = False


@dataclass
class FaceTrack:
    track_id: int
    box: tuple[int, int, int, int]
    missed_frames: int = 0


@dataclass
class FaceTracker:
    tracks: dict[int, FaceTrack] = field(default_factory=dict)
    next_id: int = 1
    max_missed_frames: int = 10
    max_distance_ratio: float = 0.45

    def update(self, boxes: list[tuple[int, int, int, int]]) -> dict[tuple[int, int, int, int], int]:
        assignments: dict[tuple[int, int, int, int], int] = {}
        used_tracks: set[int] = set()

        for box in boxes:
            best_track_id = None
            best_distance = float("inf")
            bx, by, bw, bh = box
            center = (bx + bw / 2.0, by + bh / 2.0)
            max_distance = max(bw, bh) * self.max_distance_ratio

            for track_id, track in self.tracks.items():
                if track_id in used_tracks:
                    continue
                tx, ty, tw, th = track.box
                track_center = (tx + tw / 2.0, ty + th / 2.0)
                distance = ((center[0] - track_center[0]) ** 2 + (center[1] - track_center[1]) ** 2) ** 0.5
                if distance < best_distance and distance <= max_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is None:
                best_track_id = self.next_id
                self.next_id += 1

            self.tracks[best_track_id] = FaceTrack(best_track_id, box)
            used_tracks.add(best_track_id)
            assignments[box] = best_track_id

        for track_id in list(self.tracks):
            if track_id not in used_tracks:
                self.tracks[track_id].missed_frames += 1
                if self.tracks[track_id].missed_frames > self.max_missed_frames:
                    del self.tracks[track_id]

        return assignments


@dataclass
class AppState:
    camera_enabled: bool = True
    detection_enabled: bool = True
    alarm_enabled: bool = True
    show_help: bool = True
    paused: bool = False
    fullscreen: bool = False
    theme_index: int = 0
    brightness: int = 0
    contrast: float = 1.0
    sensitivity: int = 55
    frozen_frame: np.ndarray | None = None

    @property
    def theme(self) -> str:
        return THEME_ORDER[self.theme_index % len(THEME_ORDER)]


class DeepLearningMaskClassifier:
    def __init__(
        self,
        model_path: str,
        labels: list[str],
        input_size: int = 224,
        confidence_threshold: float = 0.65,
    ) -> None:
        self.labels = labels
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.net = cv2.dnn.readNetFromONNX(model_path)

    def predict(self, face_bgr: np.ndarray) -> DetectionResult | None:
        if face_bgr.size == 0:
            return None

        blob = cv2.dnn.blobFromImage(
            face_bgr,
            scalefactor=1.0 / 255.0,
            size=(self.input_size, self.input_size),
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        scores = self.net.forward().reshape(-1).astype(np.float32)
        if scores.size < 2:
            return None

        probabilities = softmax(scores)
        class_id = int(np.argmax(probabilities))
        confidence = float(probabilities[class_id])
        if confidence < self.confidence_threshold:
            return None

        label = self.labels[class_id] if class_id < len(self.labels) else str(class_id)
        label = normalize_label(label)
        color = (0, 185, 95) if label == "MASK" else (0, 0, 255)
        return DetectionResult(label, color, 0.0, False, confidence, "deep-learning")


class BeepAlarm:
    def __init__(self, enabled: bool = True, cooldown: float = 2.0) -> None:
        self.enabled = enabled
        self.cooldown = cooldown
        self.last_alarm_time = 0.0

    def trigger(self) -> None:
        now = time.time()
        if not self.enabled or now - self.last_alarm_time < self.cooldown:
            return
        print("\a", end="", flush=True)
        self.last_alarm_time = now


class ConsecutiveAlarmGate:
    def __init__(self, required_frames: int = 3) -> None:
        self.required_frames = max(1, required_frames)
        self.current_frames = 0

    def update(self, has_no_mask: bool) -> bool:
        if has_no_mask:
            self.current_frames += 1
        else:
            self.current_frames = 0
        return self.current_frames >= self.required_frames


def get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if size in FONT_CACHE:
        return FONT_CACHE[size]

    for path in FONT_PATHS:
        if path.exists():
            FONT_CACHE[size] = ImageFont.truetype(str(path), size)
            return FONT_CACHE[size]

    FONT_CACHE[size] = ImageFont.load_default()
    return FONT_CACHE[size]


def text_size(text: str, size: int) -> tuple[int, int]:
    font = get_font(size)
    box = font.getbbox(text)
    return box[2] - box[0], box[3] - box[1]


def draw_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    size: int,
    color: tuple[int, int, int],
) -> None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.text(position, text, font=get_font(size), fill=(color[2], color[1], color[0]))
    frame[:, :] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def display_label(label: str) -> str:
    labels = {
        "MASK": "\u5df2\u6234\u53e3\u7f69",
        "NO MASK": "\u672a\u6234\u53e3\u7f69",
        "BLOCKED": "\u906e\u6321\u5f02\u5e38",
        "UNCERTAIN": "\u4e0d\u786e\u5b9a",
    }
    return labels.get(label, label)


def theme_value(key: str):
    return THEMES[CURRENT_THEME][key]


def set_theme(theme: str) -> None:
    global CURRENT_THEME
    CURRENT_THEME = theme if theme in THEMES else "dark"


def no_op(_: int) -> None:
    return None


def create_controls(args: argparse.Namespace) -> None:
    cv2.createTrackbar("Brightness", WINDOW_NAME, 100, 200, no_op)
    cv2.createTrackbar("Contrast", WINDOW_NAME, 100, 300, no_op)
    cv2.createTrackbar("Sensitivity", WINDOW_NAME, int(args.no_mask_threshold * 100), 100, no_op)


def read_controls(args: argparse.Namespace, dl_classifier: DeepLearningMaskClassifier | None, state: AppState) -> None:
    state.brightness = cv2.getTrackbarPos("Brightness", WINDOW_NAME) - 100
    state.contrast = max(0.1, cv2.getTrackbarPos("Contrast", WINDOW_NAME) / 100.0)
    state.sensitivity = cv2.getTrackbarPos("Sensitivity", WINDOW_NAME)

    sensitivity = state.sensitivity / 100.0
    args.no_mask_threshold = max(0.10, min(0.90, sensitivity))
    args.skin_threshold = max(0.05, min(0.65, sensitivity * 0.62))
    args.obstruction_threshold = max(0.20, min(0.90, sensitivity))
    args.dl_threshold = max(0.10, min(0.99, sensitivity))
    if dl_classifier is not None:
        dl_classifier.confidence_threshold = args.dl_threshold


def apply_visual_adjustments(frame: np.ndarray, brightness: int, contrast: float) -> np.ndarray:
    if brightness == 0 and abs(contrast - 1.0) < 0.01:
        return frame
    return cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)


def face_quality_tips(
    frame: np.ndarray,
    faces: list[tuple[int, int, int, int]],
) -> list[str]:
    tips: list[str] = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    if brightness < 55:
        tips.append("\u5149\u7ebf\u4e0d\u8db3")
    elif brightness > 215:
        tips.append("\u753b\u9762\u8fc7\u4eae")

    if not faces:
        return tips

    frame_area = frame.shape[0] * frame.shape[1]
    largest_ratio = max((w * h) / frame_area for _, _, w, h in faces)
    if largest_ratio < 0.025:
        tips.append("\u8bf7\u9760\u8fd1\u4e00\u4e9b")
    elif largest_ratio > 0.38:
        tips.append("\u8bf7\u540e\u9000\u4e00\u4e9b")

    for _, _, w, h in faces:
        ratio = w / max(h, 1)
        if ratio < 0.68 or ratio > 1.32:
            tips.append("\u8bf7\u5c3d\u91cf\u6b63\u5bf9\u6444\u50cf\u5934")
            break
    return tips[:3]


def save_training_frame(frame: np.ndarray, folder_name: str) -> Path:
    target_dir = BASE_DIR / folder_name
    target_dir.mkdir(exist_ok=True)
    path = target_dir / f"{folder_name}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(path), frame)
    return path


def load_cascade(filename: str) -> cv2.CascadeClassifier:
    path = cv2.data.haarcascades + filename
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError(f"Could not load OpenCV cascade: {path}")
    return cascade


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    script_relative = BASE_DIR / path
    return script_relative if script_relative.exists() else path


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    total = np.sum(exp_scores)
    if total <= 0:
        return np.zeros_like(scores)
    return exp_scores / total


def normalize_label(label: str) -> str:
    normalized = label.strip().upper().replace("_", " ").replace("-", " ")
    if normalized in {"NO MASK", "WITHOUT MASK", "UNMASKED", "0 NO MASK"}:
        return "NO MASK"
    if normalized in {"MASK", "WITH MASK", "MASKED", "0 MASK"}:
        return "MASK"
    return normalized


def clamp_box(
    x: int,
    y: int,
    w: int,
    h: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_width, x + w)
    y2 = min(frame_height, y + h)
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def expand_box(
    box: tuple[int, int, int, int],
    padding: float,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    return clamp_box(x - pad_x, y - pad_y, w + pad_x * 2, h + pad_y * 2, frame_width, frame_height)


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = aw * ah + bw * bh - intersection
    return 0.0 if union <= 0 else intersection / union


def non_max_suppression(
    boxes: list[tuple[int, int, int, int]],
    threshold: float = 0.35,
) -> list[tuple[int, int, int, int]]:
    kept: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        if all(iou(box, kept_box) < threshold for kept_box in kept):
            kept.append(box)
    return kept


def detect_faces(
    gray: np.ndarray,
    cascades: list[cv2.CascadeClassifier],
    min_face_size: int,
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for cascade in cascades:
        detected = cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(min_face_size, min_face_size),
        )
        boxes.extend((int(x), int(y), int(w), int(h)) for x, y, w, h in detected)
    return non_max_suppression(boxes)


def load_deep_learning_classifier(args: argparse.Namespace) -> DeepLearningMaskClassifier | None:
    if not args.model:
        return None

    model_path = resolve_path(args.model)
    if not model_path.exists():
        print(f"Deep learning model not found: {model_path}. Using rule-based fallback.")
        return None

    labels = [normalize_label(label) for label in args.labels.split(",")]
    if len(labels) < 2:
        raise ValueError("--labels requires at least two labels, for example MASK,NO MASK")

    try:
        return DeepLearningMaskClassifier(
            model_path=str(model_path),
            labels=labels,
            input_size=args.model_input_size,
            confidence_threshold=args.dl_threshold,
        )
    except cv2.error as exc:
        raise RuntimeError(f"Could not load deep learning model: {args.model}") from exc


def skin_pixel_ratio(bgr_roi: np.ndarray) -> float:
    if bgr_roi.size == 0:
        return 0.0

    ycrcb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2YCrCb)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    ycrcb_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    lower_skin_hsv = np.array([0, 20, 45], dtype=np.uint8)
    upper_skin_hsv = np.array([25, 255, 255], dtype=np.uint8)
    hsv_mask = cv2.inRange(hsv, lower_skin_hsv, upper_skin_hsv)

    skin_mask = cv2.bitwise_and(ycrcb_mask, hsv_mask)
    kernel = np.ones((3, 3), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    return cv2.countNonZero(skin_mask) / float(skin_mask.size)


def lower_face_regions(face_bgr: np.ndarray, face_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = face_gray.shape[:2]
    lower_y1 = int(height * 0.52)
    lower_y2 = int(height * 0.92)
    lower_x1 = int(width * 0.12)
    lower_x2 = int(width * 0.88)
    return (
        face_bgr[lower_y1:lower_y2, lower_x1:lower_x2],
        face_gray[lower_y1:lower_y2, lower_x1:lower_x2],
    )


def mouth_visible(lower_face_gray: np.ndarray, mouth_cascade: cv2.CascadeClassifier) -> bool:
    mouths = mouth_cascade.detectMultiScale(
        lower_face_gray,
        scaleFactor=1.7,
        minNeighbors=12,
        minSize=(25, 15),
    )
    return len(mouths) > 0


def obstruction_score(lower_face_bgr: np.ndarray, skin_ratio: float) -> float:
    if lower_face_bgr.size == 0:
        return 0.0

    hsv = cv2.cvtColor(lower_face_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(lower_face_bgr, cv2.COLOR_BGR2GRAY)
    saturation = float(np.mean(hsv[:, :, 1]))
    brightness = float(np.mean(hsv[:, :, 2]))
    color_std = float(np.mean(np.std(lower_face_bgr.reshape(-1, 3), axis=0)))
    edges = cv2.Canny(gray, 60, 140)
    edge_density = cv2.countNonZero(edges) / float(edges.size)

    score = 0.0
    if skin_ratio > 0.18:
        score += 0.35
    if edge_density > 0.16:
        score += 0.25
    if color_std > 42:
        score += 0.2
    if saturation > 95:
        score += 0.1
    if brightness < 35 or brightness > 235:
        score += 0.1
    return min(score, 1.0)


def detect_non_mask_obstruction(
    lower_face_bgr: np.ndarray,
    skin_ratio: float,
    mouth_found: bool,
    skin_threshold: float,
    obstruction_threshold: float,
) -> tuple[bool, float]:
    if mouth_found:
        return False, 0.0

    covered_like_mask = skin_ratio <= skin_threshold
    hand_like_cover = 0.18 < skin_ratio < 0.38
    if not covered_like_mask and not hand_like_cover:
        return False, 0.0

    score = obstruction_score(lower_face_bgr, skin_ratio)
    return score >= obstruction_threshold, score


def classify_mask(
    face_bgr: np.ndarray,
    face_gray: np.ndarray,
    mouth_cascade: cv2.CascadeClassifier,
    skin_threshold: float,
    no_mask_threshold: float,
    obstruction_threshold: float,
) -> DetectionResult:
    lower_face_bgr, lower_face_gray = lower_face_regions(face_bgr, face_gray)
    ratio = skin_pixel_ratio(lower_face_bgr)
    mouth_found = mouth_visible(lower_face_gray, mouth_cascade)
    obstruction_found, obstruction_confidence = detect_non_mask_obstruction(
        lower_face_bgr,
        ratio,
        mouth_found,
        skin_threshold,
        obstruction_threshold,
    )

    if obstruction_found:
        return DetectionResult("BLOCKED", (0, 105, 255), ratio, mouth_found, obstruction_confidence, obstruction_found=True)

    if mouth_found or ratio >= no_mask_threshold:
        confidence = max(0.55, min(0.95, ratio / max(no_mask_threshold, 0.01)))
        return DetectionResult("NO MASK", (0, 0, 255), ratio, mouth_found, confidence)

    if ratio <= skin_threshold:
        confidence = max(0.55, min(0.95, 1.0 - ratio / max(skin_threshold, 0.01) * 0.4))
        return DetectionResult("MASK", (0, 185, 95), ratio, mouth_found, confidence)

    return DetectionResult("UNCERTAIN", (0, 190, 255), ratio, mouth_found, 0.5)


def classify_mask_with_deep_learning(
    face_bgr: np.ndarray,
    face_gray: np.ndarray,
    mouth_cascade: cv2.CascadeClassifier,
    skin_threshold: float,
    no_mask_threshold: float,
    obstruction_threshold: float,
    classifier: DeepLearningMaskClassifier | None,
) -> DetectionResult:
    if classifier is not None:
        result = classifier.predict(face_bgr)
        if result is not None:
            if result.label == "MASK":
                lower_face_bgr, lower_face_gray = lower_face_regions(face_bgr, face_gray)
                ratio = skin_pixel_ratio(lower_face_bgr)
                mouth_found = mouth_visible(lower_face_gray, mouth_cascade)
                obstruction_found, obstruction_confidence = detect_non_mask_obstruction(
                    lower_face_bgr,
                    ratio,
                    mouth_found,
                    skin_threshold,
                    obstruction_threshold,
                )
                if obstruction_found:
                    return DetectionResult(
                        "BLOCKED",
                        (0, 105, 255),
                        ratio,
                        mouth_found,
                        obstruction_confidence,
                        method="deep-learning-check",
                        obstruction_found=True,
                    )
            return result

    result = classify_mask(face_bgr, face_gray, mouth_cascade, skin_threshold, no_mask_threshold, obstruction_threshold)
    result.method = "rule-fallback" if classifier is not None else "rule"
    return result


def draw_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    text_width, text_height = text_size(text, 18)
    x1 = max(8, x)
    y1 = max(8, y - text_height - 18)
    x2 = min(frame.shape[1] - 8, x1 + text_width + 24)
    y2 = min(frame.shape[0] - 8, y1 + text_height + 18)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), theme_value("panel"), -1)
    cv2.rectangle(overlay, (x1, y1), (x1 + 6, y2), color, -1)
    cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    draw_text(frame, text, (x1 + 14, y1 + 8), 18, (255, 255, 255))


def draw_status(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (min(frame.shape[1] - 12, 540), 56), theme_value("panel"), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.circle(frame, (34, 34), 7, color, -1, cv2.LINE_AA)
    draw_text(frame, text, (52, 22), 22, theme_value("text"))


def draw_corner_box(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color: tuple[int, int, int],
) -> None:
    length = max(18, int(min(w, h) * 0.24))
    thickness = 3
    shadow = (12, 16, 22)

    for dx, dy, col, thick in ((2, 2, shadow, thickness + 2), (0, 0, color, thickness)):
        xx = x + dx
        yy = y + dy
        cv2.line(frame, (xx, yy), (xx + length, yy), col, thick, cv2.LINE_AA)
        cv2.line(frame, (xx, yy), (xx, yy + length), col, thick, cv2.LINE_AA)
        cv2.line(frame, (xx + w, yy), (xx + w - length, yy), col, thick, cv2.LINE_AA)
        cv2.line(frame, (xx + w, yy), (xx + w, yy + length), col, thick, cv2.LINE_AA)
        cv2.line(frame, (xx, yy + h), (xx + length, yy + h), col, thick, cv2.LINE_AA)
        cv2.line(frame, (xx, yy + h), (xx, yy + h - length), col, thick, cv2.LINE_AA)
        cv2.line(frame, (xx + w, yy + h), (xx + w - length, yy + h), col, thick, cv2.LINE_AA)
        cv2.line(frame, (xx + w, yy + h), (xx + w, yy + h - length), col, thick, cv2.LINE_AA)


def draw_dashboard(
    frame: np.ndarray,
    status_text: str,
    status_color: tuple[int, int, int],
    face_count: int,
    no_mask_count: int,
    fps: float,
    has_model: bool,
) -> None:
    draw_status(frame, status_text, status_color)

    panel_w = 245
    x1 = max(12, frame.shape[1] - panel_w - 14)
    y1 = 12
    x2 = frame.shape[1] - 14
    y2 = 150

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), theme_value("panel"), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), theme_value("panel_border"), 1, cv2.LINE_AA)
    draw_text(frame, "\u5b9e\u65f6\u68c0\u6d4b\u9762\u677f", (x1 + 16, y1 + 14), 19, theme_value("text"))

    rows = [
        ("\u4eba\u8138", str(face_count), (120, 210, 255)),
        ("\u63d0\u9192", str(no_mask_count), (0, 0, 255) if no_mask_count else (0, 190, 0)),
        ("\u5e27\u7387", f"{fps:4.1f}" if fps > 0 else "--", theme_value("text")),
        ("\u6a21\u5f0f", "\u6a21\u578b" if has_model else "\u89c4\u5219", theme_value("text")),
    ]
    for index, (name, value, value_color) in enumerate(rows):
        row_y = y1 + 56 + index * 22
        draw_text(frame, name, (x1 + 16, row_y - 14), 16, theme_value("muted"))
        draw_text(frame, value, (x2 - 92, row_y - 14), 16, value_color)


def draw_alarm_frame(frame: np.ndarray) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 18)
    cv2.addWeighted(overlay, 0.48, frame, 0.52, 0, frame)
    cv2.rectangle(frame, (8, 8), (frame.shape[1] - 9, frame.shape[0] - 9), (0, 0, 255), 2, cv2.LINE_AA)


def draw_quality_tips(frame: np.ndarray, tips: list[str]) -> None:
    if not tips:
        return

    text = "\u63d0\u793a\uff1a" + " / ".join(tips)
    width, height = text_size(text, 19)
    x1 = 14
    y1 = 66
    x2 = min(frame.shape[1] - 14, x1 + width + 28)
    y2 = y1 + height + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), theme_value("panel"), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 1, cv2.LINE_AA)
    draw_text(frame, text, (x1 + 14, y1 + 9), 19, theme_value("text"))


def draw_control_bar(
    frame: np.ndarray,
    camera_enabled: bool,
    detection_enabled: bool,
    alarm_enabled: bool,
    show_help: bool,
) -> None:
    if not show_help:
        return

    on_text = "\u5f00"
    off_text = "\u5173"
    line1 = (
        f"C \u6444\u50cf\u5934:{on_text if camera_enabled else off_text}    "
        f"D \u68c0\u6d4b:{on_text if detection_enabled else off_text}    "
        f"A \u62a5\u8b66:{on_text if alarm_enabled else off_text}    "
        "P \u6682\u505c    F \u5168\u5c4f    T \u4e3b\u9898"
    )
    line2 = (
        "+/- \u4eae\u5ea6    ,/. \u5bf9\u6bd4\u5ea6    S \u622a\u56fe    "
        "X \u5047\u9633\u6027    V \u5047\u9634\u6027    H \u63d0\u793a    Q/Esc \u9000\u51fa"
    )
    y1 = frame.shape[0] - 72
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, y1), (frame.shape[1] - 12, frame.shape[0] - 12), theme_value("panel"), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    draw_text(frame, line1, (28, y1 + 9), 18, theme_value("text"))
    draw_text(frame, line2, (28, y1 + 37), 16, theme_value("muted"))


def draw_camera_off_screen(frame: np.ndarray, show_help: bool, alarm_enabled: bool) -> None:
    frame[:, :] = theme_value("camera_off")
    cx = frame.shape[1] // 2
    cy = frame.shape[0] // 2
    cv2.circle(frame, (cx, cy - 35), 42, (68, 76, 88), 2, cv2.LINE_AA)
    cv2.line(frame, (cx - 25, cy - 10), (cx + 25, cy - 60), (68, 76, 88), 3, cv2.LINE_AA)
    draw_text(frame, "\u6444\u50cf\u5934\u5df2\u5173\u95ed", (cx - 82, cy + 8), 28, theme_value("text"))
    draw_text(frame, "\u6309 C \u91cd\u65b0\u5f00\u542f\u6444\u50cf\u5934", (cx - 112, cy + 48), 20, theme_value("muted"))
    draw_dashboard(frame, "\u6444\u50cf\u5934\u5df2\u5173\u95ed", (180, 180, 180), 0, 0, 0.0, False)
    draw_control_bar(frame, False, False, alarm_enabled, show_help)


def save_snapshot(frame: np.ndarray) -> Path:
    snapshot_dir = BASE_DIR / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    path = snapshot_dir / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(path), frame)
    return path


def preprocess_gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def enhance_frame(frame: np.ndarray, sharpen: float, denoise: bool) -> np.ndarray:
    if denoise:
        frame = cv2.bilateralFilter(frame, d=5, sigmaColor=45, sigmaSpace=45)

    if sharpen <= 0:
        return frame

    blurred = cv2.GaussianBlur(frame, (0, 0), 1.2)
    return cv2.addWeighted(frame, 1.0 + sharpen, blurred, -sharpen, 0)


def process_frame_detailed(
    frame: np.ndarray,
    face_cascades: list[cv2.CascadeClassifier],
    mouth_cascade: cv2.CascadeClassifier,
    dl_classifier: DeepLearningMaskClassifier | None,
    args: argparse.Namespace,
    tracker: FaceTracker | None = None,
) -> tuple[np.ndarray, int, int, list[str]]:
    if args.scale != 1.0:
        frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale)

    frame = enhance_frame(frame, args.sharpen, args.denoise)
    gray = preprocess_gray(frame)
    faces = detect_faces(gray, face_cascades, args.min_face_size)
    quality_tips = face_quality_tips(frame, faces)
    expanded_faces = [
        expand_box((x, y, w, h), args.face_padding, frame.shape[1], frame.shape[0])
        for x, y, w, h in faces
    ]
    track_ids = tracker.update(expanded_faces) if tracker is not None else {}
    no_mask_count = 0

    for ex, ey, ew, eh in expanded_faces:
        face_bgr = frame[ey : ey + eh, ex : ex + ew]
        face_gray = gray[ey : ey + eh, ex : ex + ew]
        result = classify_mask_with_deep_learning(
            face_bgr,
            face_gray,
            mouth_cascade,
            args.skin_threshold,
            args.no_mask_threshold,
            args.obstruction_threshold,
            dl_classifier,
        )

        if result.label in {"NO MASK", "BLOCKED"}:
            no_mask_count += 1

        draw_corner_box(frame, ex, ey, ew, eh, result.color)
        if result.method == "deep-learning":
            label = f"{display_label(result.label)} \u6a21\u578b={result.confidence:.2f}"
        elif result.label == "BLOCKED":
            label = f"{display_label(result.label)} \u906e\u6321={result.confidence:.2f}"
        else:
            label = f"{display_label(result.label)} \u89c4\u5219={result.confidence:.2f} \u80a4\u8272={result.skin_ratio:.2f}"
        track_id = track_ids.get((ex, ey, ew, eh))
        if track_id is not None:
            label = f"ID {track_id}  {label}"
        draw_label(frame, label, ex, max(ey - 8, 28), result.color)

    return frame, len(faces), no_mask_count, quality_tips


def process_frame(
    frame: np.ndarray,
    face_cascades: list[cv2.CascadeClassifier],
    mouth_cascade: cv2.CascadeClassifier,
    dl_classifier: DeepLearningMaskClassifier | None,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int, int]:
    output, face_count, no_mask_count, _ = process_frame_detailed(
        frame,
        face_cascades,
        mouth_cascade,
        dl_classifier,
        args,
    )
    return output, face_count, no_mask_count


def configure_capture(cap: cv2.VideoCapture, args: argparse.Namespace) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)


def open_capture(source: int | str, args: argparse.Namespace) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if cap.isOpened() and not args.video:
        configure_capture(cap, args)
    return cap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple OpenCV mask detector")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--image", type=str, default="", help="detect one image and show result")
    parser.add_argument("--video", type=str, default="", help="detect a video file instead of camera")
    parser.add_argument("--scale", type=float, default=1.0, help="frame resize scale")
    parser.add_argument("--width", type=int, default=1280, help="requested camera width")
    parser.add_argument("--height", type=int, default=720, help="requested camera height")
    parser.add_argument("--fps", type=int, default=30, help="requested camera fps")
    parser.add_argument("--sharpen", type=float, default=0.35, help="display sharpening amount")
    parser.add_argument("--denoise", action="store_true", help="enable light denoise before detection")
    parser.add_argument("--min-face-size", type=int, default=70, help="minimum detected face size")
    parser.add_argument("--face-padding", type=float, default=0.12, help="padding around face box")
    parser.add_argument("--model", type=str, default="models/mask_detector.onnx", help="path to ONNX model")
    parser.add_argument("--model-input-size", type=int, default=224, help="deep learning model input size")
    parser.add_argument("--labels", type=str, default="MASK,NO MASK", help="comma separated class names")
    parser.add_argument("--dl-threshold", type=float, default=0.65, help="minimum deep learning confidence")
    parser.add_argument("--skin-threshold", type=float, default=0.28, help="lower value means more likely mask")
    parser.add_argument("--no-mask-threshold", type=float, default=0.45, help="higher value means more likely no mask")
    parser.add_argument("--obstruction-threshold", type=float, default=0.55, help="non-mask obstruction threshold")
    parser.add_argument("--alarm-cooldown", type=float, default=2.0, help="seconds between alarm beeps")
    parser.add_argument("--alarm-frames", type=int, default=3, help="consecutive no-mask frames before alarm")
    parser.add_argument("--no-alarm", action="store_true", help="show visual warning only")
    return parser.parse_args()


def draw_overall_status(
    frame: np.ndarray,
    face_count: int,
    no_mask_count: int,
    should_alarm: bool,
    fps: float,
    has_model: bool,
) -> None:
    if face_count == 0:
        status_text = "\u672a\u68c0\u6d4b\u5230\u4eba\u8138"
        status_color = (180, 180, 180)
    elif should_alarm:
        status_text = "\u8b66\u544a\uff1a\u672a\u6234\u53e3\u7f69\u6216\u906e\u6321\u5f02\u5e38"
        status_color = (0, 0, 255)
    elif no_mask_count > 0:
        status_text = "\u68c0\u67e5\u4e2d\uff1a\u53ef\u80fd\u672a\u6234\u53e3\u7f69"
        status_color = (0, 180, 255)
    else:
        status_text = "\u53e3\u7f69\u68c0\u6d4b\u8fd0\u884c\u4e2d"
        status_color = (0, 180, 0)

    if should_alarm:
        draw_alarm_frame(frame)

    draw_dashboard(frame, status_text, status_color, face_count, no_mask_count, fps, has_model)


def main() -> None:
    args = parse_args()
    face_cascades = [
        load_cascade("haarcascade_frontalface_default.xml"),
        load_cascade("haarcascade_frontalface_alt2.xml"),
    ]
    mouth_cascade = load_cascade("haarcascade_smile.xml")
    dl_classifier = load_deep_learning_classifier(args)

    alarm = BeepAlarm(enabled=not args.no_alarm, cooldown=args.alarm_cooldown)
    alarm_gate = ConsecutiveAlarmGate(required_frames=args.alarm_frames)
    tracker = FaceTracker()

    if args.image:
        image_path = resolve_path(args.image)
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        frame, face_count, no_mask_count, quality_tips = process_frame_detailed(
            frame,
            face_cascades,
            mouth_cascade,
            dl_classifier,
            args,
            tracker,
        )
        draw_overall_status(frame, face_count, no_mask_count, no_mask_count > 0, 0.0, dl_classifier is not None)
        draw_quality_tips(frame, quality_tips)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    source: int | str = str(resolve_path(args.video)) if args.video else args.camera
    cap = open_capture(source, args)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    if not args.video:
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera stream: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.width, args.height)
    create_controls(args)
    print("Keys: C camera, D detection, A alarm, P pause, F fullscreen, T theme, S snapshot, X/V correction, Q/Esc quit.")

    state = AppState(alarm_enabled=not args.no_alarm)
    set_theme(state.theme)
    alarm.enabled = state.alarm_enabled
    last_frame_time = time.perf_counter()
    smoothed_fps = 0.0
    last_display_frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    video_finished = False

    while True:
        read_controls(args, dl_classifier, state)
        set_theme(state.theme)

        if state.paused and state.frozen_frame is not None:
            frame = state.frozen_frame.copy()
            draw_dashboard(frame, "\u753b\u9762\u5df2\u6682\u505c", (0, 180, 255), 0, 0, smoothed_fps, dl_classifier is not None)
            draw_control_bar(frame, state.camera_enabled, state.detection_enabled, state.alarm_enabled, state.show_help)
        elif state.camera_enabled:
            ok, frame = cap.read()
            if not ok:
                state.camera_enabled = False
                cap.release()
                frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                if args.video:
                    video_finished = True
                    frame[:, :] = theme_value("camera_off")
                    draw_text(frame, "\u89c6\u9891\u56de\u653e\u7ed3\u675f", (frame.shape[1] // 2 - 110, frame.shape[0] // 2), 30, theme_value("text"))
                    draw_text(frame, "\u6309 C \u91cd\u65b0\u64ad\u653e\uff0c\u6309 Q \u9000\u51fa", (frame.shape[1] // 2 - 135, frame.shape[0] // 2 + 44), 20, theme_value("muted"))
                    draw_control_bar(frame, False, state.detection_enabled, state.alarm_enabled, state.show_help)
                else:
                    draw_camera_off_screen(frame, state.show_help, state.alarm_enabled)
            else:
                frame = apply_visual_adjustments(frame, state.brightness, state.contrast)
                if state.detection_enabled:
                    frame, face_count, no_mask_count, quality_tips = process_frame_detailed(frame, face_cascades, mouth_cascade, dl_classifier, args, tracker)
                    should_alarm = alarm_gate.update(no_mask_count > 0)
                    if should_alarm:
                        alarm.trigger()
                    now = time.perf_counter()
                    elapsed = max(now - last_frame_time, 1e-6)
                    last_frame_time = now
                    instant_fps = 1.0 / elapsed
                    smoothed_fps = instant_fps if smoothed_fps <= 0 else smoothed_fps * 0.88 + instant_fps * 0.12
                    draw_overall_status(frame, face_count, no_mask_count, should_alarm, smoothed_fps, dl_classifier is not None)
                    draw_quality_tips(frame, quality_tips)
                else:
                    frame = enhance_frame(frame, args.sharpen, args.denoise)
                    alarm_gate.update(False)
                    draw_dashboard(frame, "\u68c0\u6d4b\u5df2\u6682\u505c", (0, 180, 255), 0, 0, smoothed_fps, dl_classifier is not None)

                draw_control_bar(frame, state.camera_enabled, state.detection_enabled, state.alarm_enabled, state.show_help)
        else:
            frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            if video_finished:
                frame[:, :] = theme_value("camera_off")
                draw_text(frame, "\u89c6\u9891\u56de\u653e\u7ed3\u675f", (frame.shape[1] // 2 - 110, frame.shape[0] // 2), 30, theme_value("text"))
                draw_text(frame, "\u6309 C \u91cd\u65b0\u64ad\u653e\uff0c\u6309 Q \u9000\u51fa", (frame.shape[1] // 2 - 135, frame.shape[0] // 2 + 44), 20, theme_value("muted"))
                draw_control_bar(frame, False, state.detection_enabled, state.alarm_enabled, state.show_help)
            else:
                draw_camera_off_screen(frame, state.show_help, state.alarm_enabled)

        last_display_frame = frame.copy()
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("c"), ord("C")):
            if state.camera_enabled:
                cap.release()
                state.camera_enabled = False
                state.paused = False
                alarm_gate.update(False)
            else:
                cap = open_capture(source, args)
                state.camera_enabled = cap.isOpened()
                state.paused = False
                video_finished = False
                tracker.tracks.clear()
                last_frame_time = time.perf_counter()
                smoothed_fps = 0.0
        elif key in (ord("d"), ord("D")):
            state.detection_enabled = not state.detection_enabled
            alarm_gate.update(False)
        elif key in (ord("a"), ord("A")):
            state.alarm_enabled = not state.alarm_enabled
            alarm.enabled = state.alarm_enabled
        elif key in (ord("p"), ord("P")):
            if state.camera_enabled:
                state.paused = not state.paused
                state.frozen_frame = last_display_frame.copy() if state.paused else None
        elif key in (ord("f"), ord("F")):
            state.fullscreen = not state.fullscreen
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if state.fullscreen else cv2.WINDOW_NORMAL)
        elif key in (ord("t"), ord("T")):
            state.theme_index = (state.theme_index + 1) % len(THEME_ORDER)
            set_theme(state.theme)
        elif key in (ord("h"), ord("H")):
            state.show_help = not state.show_help
        elif key in (ord("s"), ord("S")):
            snapshot_path = save_snapshot(last_display_frame)
            print(f"Snapshot saved: {snapshot_path}")
        elif key in (ord("x"), ord("X")):
            correction_path = save_training_frame(last_display_frame, "false_positive")
            print(f"False positive frame saved: {correction_path}")
        elif key in (ord("v"), ord("V")):
            correction_path = save_training_frame(last_display_frame, "false_negative")
            print(f"False negative frame saved: {correction_path}")
        elif key in (ord("+"), ord("=")):
            cv2.setTrackbarPos("Brightness", WINDOW_NAME, min(200, cv2.getTrackbarPos("Brightness", WINDOW_NAME) + 5))
        elif key in (ord("-"), ord("_")):
            cv2.setTrackbarPos("Brightness", WINDOW_NAME, max(0, cv2.getTrackbarPos("Brightness", WINDOW_NAME) - 5))
        elif key == ord("."):
            cv2.setTrackbarPos("Contrast", WINDOW_NAME, min(300, cv2.getTrackbarPos("Contrast", WINDOW_NAME) + 5))
        elif key == ord(","):
            cv2.setTrackbarPos("Contrast", WINDOW_NAME, max(10, cv2.getTrackbarPos("Contrast", WINDOW_NAME) - 5))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
