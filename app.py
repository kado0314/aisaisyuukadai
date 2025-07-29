import streamlit as st
import random, os, math
import numpy as np
import cv2
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# カメラ入力
def camera_input_widget(label, key=None):
    return st.camera_input(label, key=key)

def calc_angle(a, b, c):
    ...

def mediapipe_reset():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=True,
                           max_num_hands=2,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)
    return mp_hands, mp_drawing, hands

def mediapipe_process(uploaded_file, mp_hands, mp_drawing, hands):
    if uploaded_file is None:
        return 0
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    total = 0
    annotated = img.copy()
    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(annotated, lm, mp_hands.HAND_CONNECTIONS)
            # 指の数を数える
            # （calc_angle やランドマーク解析を適宜呼び出す）
            # total += ...
    return total

@st.cache_resource
def load_diff_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "./dreamshaper-6-safetensors",
        torch_dtype=torch.float32,
        use_safetensors=True,
        safety_checker=None
    )
    return pipe.to("cpu")

@st.cache_resource
def load_yolo_model():
    import torch
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    return model

def detect_objects(model, img_path):
    results = model(img_path)
    return len(results.pandas().xyxy[0])

# Streamlit UI
st.title("🍎 果物数当てゲーム")
pipe = load_diff_model()
yolo = load_yolo_model()

if st.button("画像生成"):
    num = random.randint(1, 10)
    prompt = f"Exactly {num} red apples on a white background..."
    with st.spinner("生成中..."):
        img = pipe(prompt, height=512, width=512, num_inference_steps=30).images[0]
    save_path = "generated.png"
    img.save(save_path)
    st.image(img, caption=f"{num} 個の画像", use_container_width=True)
    detected = detect_objects(yolo, save_path)
    st.write(f"YOLO 検出数: {detected}")

count = None
uploaded = camera_input_widget("指の本数を撮影")
if uploaded:
    mp_hands, mp_drawing, hands = mediapipe_reset()
    count = mediapipe_process(uploaded, mp_hands, mp_drawing, hands)
    st.write(f"検出された指の本数: {count}")

if st.button("答え合わせ") and count is not None:
    st.write("…答え合わせ処理をここに")
