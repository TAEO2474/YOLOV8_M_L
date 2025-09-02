import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import gdown




# =========================
# 1️⃣ Google Drive에서 모델 다운로드
# =========================
FILE_ID = "1CW3aLADi1X1ve5qkM_whe8V1MDnl0Dai"  # 공유 링크 파일 ID
MODEL_PATH = "yolo_best.pt"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    st.info("모델 다운로드 중...")
    gdown.download(url, MODEL_PATH, quiet=False)
    st.success("모델 다운로드 완료!")

# =========================
# 2️⃣ 모델 로딩
# =========================
@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(MODEL_PATH)
names = model.names

# =========================
# 3️⃣ Streamlit UI
# =========================
st.title("YOLOv8 Object Detection 🚀")
uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    results = model(image, conf=0.25, iou=0.7)
    r = results[0]

    # 클래스별 탐지 개수
    class_ids = r.boxes.cls.cpu().numpy().astype(int)
    class_counts = {names[i]: (class_ids == i).sum() for i in np.unique(class_ids)}
    st.subheader("클래스별 탐지 개수")
    st.write(class_counts)

    # 표시할 클래스 선택
    selected_classes = st.multiselect(
        "표시할 클래스 선택",
        options=list(names.values()),
        default=list(names.values())
    )

    # 이미지에 박스 표시
    img_np = np.array(image)
    for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        cls_id = int(cls_id.item())
        cls_name = names[cls_id]
        if cls_name not in selected_classes:
            continue
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_np, f"{cls_name} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    st.subheader("탐지 결과")

    st.image(img_np, caption="Detection Result", use_column_width=True)
