import streamlit as st
import os
import cv2
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
import tempfile
import seaborn as sns
import gdown

# === Class ID to Character Mapping ===
class_map = {
    0: '-', 1: '.', 2: '0', 3: '1', 4: '2', 5: '3',
    6: '4', 7: '5', 8: '6', 9: '7', 10: '8', 11: '9'
}
# === Removing Outliers===
def remove_outliers_iqr(df, y_col):
    # Compute Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df[y_col].quantile(0.15)
    Q3 = df[y_col].quantile(0.85)
    IQR = Q3 - Q1
    
    # Define lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter data within bounds
    df_clean = df[(df[y_col] >= lower_bound) & (df[y_col] <= upper_bound)]

    return df_clean
# === Frame Extractor with Progress Bar ===
def extract_frames(video_path, frame_interval=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = int(fps * frame_interval)
    frame_count = 0
    frame_index = 0
    frames = []

    extract_progress = st.progress(0, text="Extracting frames...")
    total_steps = total_frames if total_frames > 0 else 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_step == 0:
            frames.append((frame_index, frame))
            frame_index += 1
        frame_count += 1
        extract_progress.progress(min(frame_count / total_steps, 1.0), text="Extracting frames...")

    cap.release()
    extract_progress.empty()
    return frames

# === Digit Detection with Streamlit Progress Bar ===
def predict_digits(model, frames):
    results = []
    progress = st.progress(0, text="Predicting digits...")
    total = len(frames)

    for i, (index, frame) in enumerate(frames):
        h, w, _ = frame.shape

        preds = model.predict(
            source=frame,
            conf=0.15,
            iou=0.1,
            show_conf=False
        )[0]

        characters = []
        for box in preds.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            label = int(cls)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            if 0 <= center_x <= w and 0 <= center_y <= h and label in class_map:
                characters.append((center_x, class_map[label]))

        characters.sort(key=lambda x: x[0])
        number_str = ''.join([c[1] for c in characters])
        has_decimal = '.' in number_str

        results.append({
            'frame_number': f"frame_{index:04d}",
            'prediction': number_str,
            'has_decimal': has_decimal
        })

        progress.progress((i + 1) / total, text=f"Predicting digits... ({i + 1}/{total})")

    progress.empty()
    return results

# === Streamlit App ===
st.title("Digit Detection from Video")

# Let user set frame interval
frame_interval = st.number_input(
    "Frame interval (in seconds)", 
    min_value=0.1, 
    max_value=10.0, 
    value=0.5, 
    step=0.1
)

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file:
    with st.spinner("Loading video..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name

    # Download the YOLO model from Google Drive
    model_url = "https://drive.google.com/file/d/1sIZedSrlG63U2ixjzqAMP7eMXnKbk3i7/"  # Google Drive ID
    model_path = "best_model.pt"
    gdown.download(model_url, model_path, quiet=False)

    # Load the model
    model = YOLO(model_path)

    frames = extract_frames(video_path, frame_interval=frame_interval)
    results = predict_digits(model, frames)

    df = pd.DataFrame(results)
    df = df[df["has_decimal"] == True]
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df = df.dropna(subset=["prediction"])  # Optional but recommended
    df = remove_outliers_iqr(df, "prediction")
    df = df.drop(columns="has_decimal")
    sns.histplot(df["prediction"], kde=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = f"digit_predictions_{timestamp}.xlsx"
    df.to_excel(excel_path, index=False)

    st.success("✅ Done!")
    st.dataframe(df)

    with open(excel_path, "rb") as f:
        st.download_button("📥 Download Excel", f, file_name=excel_path)
