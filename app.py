import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(page_title="Chat-JI: ALPD", page_icon="🚗")

st.title("🚗 Automatic License Plate Detection (ALPD)")
st.write("Apne **Chat-JI** project ke liye Image ya Video upload karein.")

# 1. Model Load karein
@st.cache_resource # Takay model bar bar load na ho
def load_model():
    return YOLO('best.pt')

model = load_model()

# Sidebar for options
st.sidebar.title("Settings")
mode = st.sidebar.radio("Select Mode", ["Image", "Video"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25)

# --- IMAGE MODE ---
if mode == "Image":
    uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Detect Plate'):
            results = model.predict(source=image, conf=conf_threshold)
            res_plotted = results[0].plot()
            st.image(res_plotted, caption='Detection Result', use_column_width=True)

# --- VIDEO MODE ---
elif mode == "Video":
    uploaded_file = st.file_uploader("Upload a video...", type=['mp4', 'mov', 'avi'])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        st.video(tfile.name) # Original Video
        
        if st.button('Process Video'):
            st.write("Processing frames... Please wait.")
            cap = cv2.VideoCapture(tfile.name)
            
            # Placeholder for processing
            frame_placeholder = st.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Inference
                results = model.predict(frame, conf=conf_threshold, verbose=False)
                res_plotted = results[0].plot()
                
                # Convert BGR to RGB for Streamlit
                res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(res_plotted_rgb, channels="RGB")
            
            cap.release()
            st.success("Processing Complete!")