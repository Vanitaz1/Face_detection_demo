# -*- coding: utf-8 -*-
"""Streamlit web app for the mask detector.

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

import mask_detector as detector


TEXT = {
    "zh": {
        "page_title": "口罩检测",
        "title": "口罩佩戴检测系统",
        "detection": "检测设置",
        "camera_index": "摄像头编号",
        "model_path": "ONNX 模型路径",
        "labels": "类别顺序",
        "dl_confidence": "深度学习置信度",
        "skin_threshold": "肤色阈值",
        "no_mask_threshold": "未戴口罩阈值",
        "obstruction_threshold": "非口罩遮挡阈值",
        "image": "画面设置",
        "camera_width": "摄像头宽度",
        "camera_height": "摄像头高度",
        "camera_fps": "摄像头帧率",
        "sharpen": "锐化强度",
        "denoise": "降噪",
        "min_face_size": "最小人脸尺寸",
        "face_padding": "人脸框扩边",
        "model_input_size": "模型输入尺寸",
        "live_tab": "实时摄像头",
        "snapshot_tab": "浏览器拍照",
        "upload_tab": "上传图片",
        "live_duration": "实时检测时长",
        "close_other_camera": "当前使用 OpenCV 本机摄像头模式。开始前请关闭其他摄像头窗口。",
        "start_live": "开始实时检测",
        "stop_live": "关闭实时检测",
        "live_waiting": "实时检测未启动。",
        "take_photo": "拍照检测",
        "choose_image": "选择图片",
        "faces": "人脸",
        "no_mask": "未戴口罩",
        "unsafe": "需提醒",
        "camera_open_failed": "无法打开摄像头编号",
        "frame_failed": "读取摄像头画面失败。",
        "live_done": "实时检测结束，已处理 {frame_count} 帧。",
        "info": "当前 Web 版支持 OpenCV 本机摄像头实时检测、浏览器拍照和图片上传。实时检测不依赖浏览器摄像头权限。",
    },
    "en": {
        "page_title": "Mask Detector",
        "title": "Mask Wearing Detection System",
        "detection": "Detection",
        "camera_index": "Camera index",
        "model_path": "ONNX model path",
        "labels": "Labels",
        "dl_confidence": "DL confidence",
        "skin_threshold": "Skin threshold",
        "no_mask_threshold": "No-mask threshold",
        "obstruction_threshold": "Non-mask obstruction threshold",
        "image": "Image",
        "camera_width": "Camera width",
        "camera_height": "Camera height",
        "camera_fps": "Camera FPS",
        "sharpen": "Sharpen",
        "denoise": "Denoise",
        "min_face_size": "Min face size",
        "face_padding": "Face padding",
        "model_input_size": "Model input size",
        "live_tab": "Live camera",
        "snapshot_tab": "Camera snapshot",
        "upload_tab": "Upload image",
        "live_duration": "Live detection duration",
        "close_other_camera": "OpenCV local-camera mode is active. Close any other camera window before starting live detection.",
        "start_live": "Start live detection",
        "stop_live": "Stop live detection",
        "live_waiting": "Live detection is not running.",
        "take_photo": "Take a photo",
        "choose_image": "Choose an image",
        "faces": "Faces",
        "no_mask": "No mask",
        "unsafe": "Alerts",
        "camera_open_failed": "Could not open camera index",
        "frame_failed": "Camera frame read failed.",
        "live_done": "Live detection finished. Processed {frame_count} frames.",
        "info": "This web version supports OpenCV local live camera detection, browser snapshots, and image upload. Live detection does not depend on browser camera permission.",
    },
}


def get_text() -> dict[str, str]:
    return TEXT[st.session_state.get("language", "zh")]


def build_args() -> argparse.Namespace:
    """Create detector settings for the web UI."""

    return argparse.Namespace(
        camera=0,
        image="",
        video="",
        scale=1.0,
        width=1280,
        height=720,
        fps=30,
        sharpen=st.session_state.get("sharpen", 0.35),
        denoise=st.session_state.get("denoise", False),
        min_face_size=st.session_state.get("min_face_size", 70),
        face_padding=st.session_state.get("face_padding", 0.12),
        model=st.session_state.get("model_path", "models/mask_detector.onnx"),
        model_input_size=st.session_state.get("model_input_size", 224),
        labels=st.session_state.get("labels", "MASK,NO MASK"),
        dl_threshold=st.session_state.get("dl_threshold", 0.65),
        skin_threshold=st.session_state.get("skin_threshold", 0.28),
        no_mask_threshold=st.session_state.get("no_mask_threshold", 0.45),
        obstruction_threshold=st.session_state.get("obstruction_threshold", 0.55),
        alarm_cooldown=2.0,
        alarm_frames=3,
        no_alarm=True,
    )


@st.cache_resource
def load_resources(model_path: str, labels: str, input_size: int, dl_threshold: float):
    """Load cascades and optional deep learning model once."""

    args = argparse.Namespace(
        model=model_path,
        labels=labels,
        model_input_size=input_size,
        dl_threshold=dl_threshold,
    )
    face_cascades = [
        detector.load_cascade("haarcascade_frontalface_default.xml"),
        detector.load_cascade("haarcascade_frontalface_alt2.xml"),
    ]
    mouth_cascade = detector.load_cascade("haarcascade_smile.xml")
    dl_classifier = detector.load_deep_learning_classifier(args)
    return face_cascades, mouth_cascade, dl_classifier


def decode_image(data: bytes) -> np.ndarray | None:
    file_bytes = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


@st.cache_resource
def get_local_capture(camera: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    args = argparse.Namespace(width=width, height=height, fps=fps)
    cap = cv2.VideoCapture(camera)
    if cap.isOpened():
        detector.configure_capture(cap, args)
    return cap


def release_local_capture() -> None:
    get_local_capture.clear()


def render_detection(frame_bgr: np.ndarray, args: argparse.Namespace) -> None:
    text = get_text()
    face_cascades, mouth_cascade, dl_classifier = load_resources(
        args.model,
        args.labels,
        args.model_input_size,
        args.dl_threshold,
    )
    output_bgr, face_count, no_mask_count = detector.process_frame(
        frame_bgr,
        face_cascades,
        mouth_cascade,
        dl_classifier,
        args,
    )

    if face_count == 0:
        detector.draw_status(output_bgr, "No face/person detected", (180, 180, 180))
    elif no_mask_count > 0:
        detector.draw_status(output_bgr, "ALARM: person without mask", (0, 0, 255))
    else:
        detector.draw_status(output_bgr, "Mask check complete", (0, 180, 0))

    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    st.image(output_rgb, channels="RGB", use_container_width=True)
    st.caption(f"{text['faces']}: {face_count} | {text['unsafe']}: {no_mask_count}")


def draw_detection_status(output_bgr: np.ndarray, face_count: int, no_mask_count: int) -> None:
    if face_count == 0:
        detector.draw_status(output_bgr, "No face/person detected", (180, 180, 180))
    elif no_mask_count > 0:
        detector.draw_status(output_bgr, "ALARM: no mask or blocked", (0, 0, 255))
    else:
        detector.draw_status(output_bgr, "Mask check running", (0, 180, 0))


def style_live_button(running: bool) -> None:
    color = "#dc2626" if running else "#16a34a"
    hover = "#b91c1c" if running else "#15803d"
    st.markdown(
        f"""
        <style>
        iframe[title="streamlit-webrtc component"] {{
            border-radius: 8px;
        }}
        div[data-testid="stButton"] > button[kind="primary"] {{
            background-color: {color};
            border-color: {color};
            color: white;
        }}
        div[data-testid="stButton"] > button[kind="primary"]:hover {{
            background-color: {hover};
            border-color: {hover};
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_local_camera_frame(
    args: argparse.Namespace,
    image_slot=None,
    caption_slot=None,
    message_slot=None,
) -> None:
    text = get_text()
    face_cascades, mouth_cascade, dl_classifier = load_resources(
        args.model,
        args.labels,
        args.model_input_size,
        args.dl_threshold,
    )

    cap = get_local_capture(args.camera, args.width, args.height, args.fps)
    if not cap.isOpened():
        target = message_slot if message_slot is not None else st
        target.error(f"{text['camera_open_failed']}: {args.camera}")
        return

    ok, frame = cap.read()
    if not ok:
        target = message_slot if message_slot is not None else st
        target.warning(text["frame_failed"])
        return

    output_bgr, face_count, no_mask_count = detector.process_frame(
        frame,
        face_cascades,
        mouth_cascade,
        dl_classifier,
        args,
    )
    draw_detection_status(output_bgr, face_count, no_mask_count)
    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    image_target = image_slot if image_slot is not None else st
    caption_target = caption_slot if caption_slot is not None else st
    image_target.image(output_rgb, channels="RGB", use_container_width=True)
    caption_target.caption(f"{text['faces']}: {face_count} | {text['unsafe']}: {no_mask_count}")


@st.fragment(run_every=0.35)
def live_camera_panel(args: argparse.Namespace) -> None:
    text = get_text()
    running = st.session_state.get("live_running", False)
    style_live_button(running)

    button_label = text["stop_live"] if running else text["start_live"]
    if st.button(button_label, type="primary", key="live_toggle_safe"):
        if running:
            release_local_capture()
        st.session_state["live_running"] = not running
        running = not running

    if not running:
        st.info(text["live_waiting"])
        return

    message_slot = st.empty()
    image_slot = st.empty()
    caption_slot = st.empty()
    render_local_camera_frame(args, image_slot, caption_slot, message_slot)


def main() -> None:
    st.set_page_config(page_title="Mask Detector", layout="wide")
    with st.sidebar:
        st.selectbox("语言 / Language", ["中文", "English"], key="language_label")
        st.session_state["language"] = "en" if st.session_state["language_label"] == "English" else "zh"

    text = get_text()
    st.title(text["title"])

    with st.sidebar:
        st.header(text["detection"])
        st.number_input(text["camera_index"], min_value=0, max_value=10, value=0, step=1, key="camera")
        st.text_input(text["model_path"], "models/mask_detector.onnx", key="model_path")
        st.text_input(text["labels"], "MASK,NO MASK", key="labels")
        st.slider(text["dl_confidence"], 0.1, 0.99, 0.65, 0.01, key="dl_threshold")
        st.slider(text["skin_threshold"], 0.01, 0.8, 0.28, 0.01, key="skin_threshold")
        st.slider(text["no_mask_threshold"], 0.05, 0.95, 0.45, 0.01, key="no_mask_threshold")
        st.slider(text["obstruction_threshold"], 0.1, 0.95, 0.55, 0.01, key="obstruction_threshold")

        st.header(text["image"])
        st.number_input(text["camera_width"], min_value=320, max_value=3840, value=1280, step=160, key="width")
        st.number_input(text["camera_height"], min_value=240, max_value=2160, value=720, step=120, key="height")
        st.number_input(text["camera_fps"], min_value=1, max_value=60, value=30, step=1, key="fps")
        st.slider(text["sharpen"], 0.0, 1.5, 0.35, 0.05, key="sharpen")
        st.checkbox(text["denoise"], value=False, key="denoise")
        st.slider(text["min_face_size"], 30, 180, 70, 5, key="min_face_size")
        st.slider(text["face_padding"], 0.0, 0.4, 0.12, 0.01, key="face_padding")
        st.number_input(text["model_input_size"], min_value=64, max_value=640, value=224, step=16, key="model_input_size")

    args = build_args()
    args.camera = st.session_state.get("camera", 0)
    args.width = st.session_state.get("width", 1280)
    args.height = st.session_state.get("height", 720)
    args.fps = st.session_state.get("fps", 30)

    tab_live, tab_camera, tab_upload = st.tabs(
        [
            text["live_tab"],
            text["snapshot_tab"],
            text["upload_tab"],
        ]
    )

    with tab_live:
        st.caption(text["close_other_camera"])
        live_camera_panel(args)

    with tab_camera:
        camera_image = st.camera_input(text["take_photo"])
        if camera_image is not None:
            frame = decode_image(camera_image.getvalue())
            if frame is not None:
                render_detection(frame, args)

    with tab_upload:
        uploaded = st.file_uploader(text["choose_image"], type=["jpg", "jpeg", "png", "bmp"])
        if uploaded is not None:
            frame = decode_image(uploaded.getvalue())
            if frame is not None:
                render_detection(frame, args)

    st.info(text["info"])


if __name__ == "__main__":
    main()
