# app.py
# =========================================
# YOLOv8 + Streamlit 시각화 앱 (전체 수정본)
# - Google Drive에서 가중치 자동 다운로드 (gdown)
# - 모델 로딩/파일 캐시
# - Confidence/IoU/imgsz 슬라이더
# - 클래스별 카운트/필터
# - 내장 플로팅 결과 + 선택 클래스만 수동 드로잉
# - 결과를 CSV로 다운로드
# - OpenCV 임포트 체크 (headless 권장)
# =========================================

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import gdown

# -------------------------
# 0) 환경 체크
# -------------------------
try:
    import cv2  # 반드시 opencv-python-headless 권장
except Exception as e:
    st.error(
        "OpenCV 임포트 실패: 서버/클라우드에서는 'opencv-python-headless<5'를 requirements.txt에 사용하세요.\n"
        f"에러: {e}"
    )
    st.stop()


# -------------------------
# 1) 기본 설정
# -------------------------
st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")
st.title("YOLOv8 Object Detection 🚀")

# Google Drive 모델 파일 설정
DEFAULT_FILE_ID = "1CW3aLADi1X1ve5qkM_whe8V1MDnl0Dai"  # 사용자의 공유 링크 파일 ID
DEFAULT_MODEL_PATH = "yolo_best.pt"


# -------------------------
# 2) 유틸 함수
# -------------------------
@st.cache_data(show_spinner=False)
def ensure_model_file(file_id: str, out_path: str) -> str:
    """Google Drive에서 모델 파일이 없으면 다운로드하고 로컬 경로 반환"""
    if not os.path.exists(out_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, out_path, quiet=False)
    return out_path


@st.cache_resource(show_spinner=False)
def load_model(path: str) -> YOLO:
    """YOLO 모델 로딩 (리소스 캐시)"""
    return YOLO(path)


def results_to_dataframe(r) -> pd.DataFrame:
    """Ultralytics Results(단일 프레임) -> DataFrame (xyxy, conf, cls, name)"""
    if r.boxes is None or len(r.boxes) == 0:
        return pd.DataFrame(columns=["x1", "y1", "x2", "y2", "conf", "cls", "name"])

    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    names = r.names if hasattr(r, "names") and r.names else {}

    rows = []
    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        rows.append(
            {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "conf": float(c),
                "cls": int(k),
                "name": names.get(int(k), str(int(k))),
            }
        )
    return pd.DataFrame(rows)


def draw_selected_classes(image_rgb: np.ndarray, r, selected: set) -> np.ndarray:
    """선택된 클래스만 녹색 박스로 수동 드로잉하여 반환 (RGB 입력/출력)"""
    img = image_rgb.copy()
    if r.boxes is None or len(r.boxes) == 0:
        return img

    for box, cls_id, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
        k = int(cls_id.item())
        name = r.names.get(k, str(k))
        if name not in selected:
            continue
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # RGB에 그려도 표시 문제 없음
        cv2.putText(
            img,
            f"{name} {float(conf):.2f}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return img


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# -------------------------
# 3) 사이드바 / 입력 UI
# -------------------------
with st.sidebar:
    st.header("Settings")

    # 모델 소스 선택: (1) Google Drive ID, (2) 로컬 경로 직접 입력
    model_source = st.radio("모델 소스", ["Google Drive (file_id)", "Local path"], index=0)

    if model_source == "Google Drive (file_id)":
        file_id = st.text_input("Google Drive file_id", value=DEFAULT_FILE_ID)
        model_path = DEFAULT_MODEL_PATH  # 다운로드 목적지
        with st.spinner("모델 파일 확인/다운로드 중..."):
            model_file = ensure_model_file(file_id, model_path)
    else:
        model_path = st.text_input("모델 로컬 경로(.pt)", value=DEFAULT_MODEL_PATH)
        if not os.path.exists(model_path):
            st.warning("입력한 로컬 경로에 모델 파일이 없습니다. 올바른 경로를 입력하세요.")
        model_file = model_path

    conf_th = st.slider("Confidence", 0.05, 0.95, 0.25, 0.05)
    iou_th = st.slider("IoU", 0.30, 0.95, 0.70, 0.05)
    imgsz = st.selectbox("Image size", [480, 640, 800, 1024, 1280], index=1)

    st.caption("Tip: 서버/클라우드에서는 opencv-python-headless 사용을 권장합니다.")


# -------------------------
# 4) 모델 로딩
# -------------------------
with st.spinner("모델 로딩 중..."):
    try:
        model = load_model(model_file)
    except Exception as e:
        st.error(f"모델 로딩 실패: {e}")
        st.stop()

names = model.names if hasattr(model, "names") else None
if not isinstance(names, dict):
    # 일부 케이스에서 names가 dict가 아닐 수 있음
    names = {i: str(i) for i in range(1000)}  # 안전 장치


# -------------------------
# 5) 파일 업로드 & 추론
# -------------------------
uploaded_file = st.file_uploader("이미지 업로드 (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("좌측 Settings에서 모델을 준비한 뒤, 이미지를 업로드하세요.")
    st.stop()

# 원본 이미지 표시
image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="입력 이미지", use_column_width=True)

# 추론
with st.spinner("추론 중..."):
    results = model.predict(
        source=np.array(image),
        conf=conf_th,
        iou=iou_th,
        imgsz=imgsz,
        verbose=False,
    )

if not results or len(results) == 0:
    st.warning("결과 객체가 비어 있습니다.")
    st.stop()

r = results[0]  # 단일 이미지 가정
r.names = names  # 안전하게 names 보장

# -------------------------
# 6) 결과 요약(데이터프레임/카운트)
# -------------------------
df = results_to_dataframe(r)

if df.empty:
    st.warning("검출된 객체가 없습니다.")
    st.stop()

# 클래스별 카운트
counts = df["name"].value_counts().to_dict()

st.subheader("클래스별 탐지 개수")
st.write(counts)

with st.expander("Detections (DataFrame 보기)"):
    st.dataframe(df, use_container_width=True)
    st.download_button(
        label="결과 CSV 다운로드",
        data=df_to_csv_bytes(df),
        file_name="detections.csv",
        mime="text/csv",
    )

# -------------------------
# 7) 시각화
# -------------------------
# (A) 내장 플로팅 전체 보기
plotted_bgr = r.plot()  # BGR ndarray
st.subheader("검출 결과 (전체)")
st.image(plotted_bgr[:, :, ::-1], use_column_width=True)

# (B) 선택 클래스만 그리기
all_class_names = sorted(df["name"].unique().tolist())
selected = st.multiselect(
    "표시할 클래스 선택",
    options=all_class_names,
    default=all_class_names,
)

if set(selected) != set(all_class_names):
    drawn = draw_selected_classes(np.array(image), r, set(selected))
    st.subheader("검출 결과 (선택 클래스만)")
    st.image(drawn, use_column_width=True)

st.success("완료 ✅")
