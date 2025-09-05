import streamlit as st
import cv2
import tempfile
import time
from ultralytics import YOLO
from PIL import Image
import numpy as np

model = YOLO("yolov8_detrac.pt")
st.title("Vehicle Detection")
st.subheader("IMAGE")
up_img = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
if up_img is not None:
    image = Image.open(up_img)
    st.image(image, caption="FI", use_container_width=True)
    ans = model.predict(np.array(image), imgsz=640, device="cpu")
    bbox = ans[0].plot()
    st.image(bbox, caption="Answer", use_container_width=True)

st.subheader("VIDEO")
up_vid = st.file_uploader("Choose a video", type=["mp4"])
if up_vid is not None:
    file = tempfile.NamedTemporaryFile(delete=False)
    file.write(up_vid.read())
    cap = cv2.VideoCapture(file.name)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        ans = model.predict(frame, imgsz=640, device="cpu", verbose=False)
        bbox_frame = ans[0].plot()
        stframe.image(bbox_frame, channels="BGR", use_container_width=True)
        
        time.sleep(0.03)
    cap.release()
