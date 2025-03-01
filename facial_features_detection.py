import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os
import urllib.request

# Create application title and file uploader widget
st.title("Facial Features Detection")
video_file_buffer = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])

# Function to download cascade files
def download_cascade_files():
    base_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
    cascade_files = {
        'face': 'haarcascade_frontalface_default.xml',
        'eye': 'haarcascade_eye.xml',
        'mouth': 'haarcascade_smile.xml'
    }
    
    if not os.path.exists('cascades'):
        os.makedirs('cascades')
    
    for name, filename in cascade_files.items():
        filepath = os.path.join('cascades', filename)
        if not os.path.exists(filepath):
            try:
                urllib.request.urlretrieve(base_url + filename, filepath)
            except Exception as e:
                st.error(f"Error downloading {filename}: {str(e)}")
                return None
    return cascade_files

# Load cascades
@st.cache_resource
def load_cascades():
    cascade_files = download_cascade_files()
    if not cascade_files:
        return None, None, None
    
    face_cascade = cv2.CascadeClassifier(os.path.join('cascades', cascade_files['face']))
    eye_cascade = cv2.CascadeClassifier(os.path.join('cascades', cascade_files['eye']))
    mouth_cascade = cv2.CascadeClassifier(os.path.join('cascades', cascade_files['mouth']))
    
    if face_cascade.empty() or eye_cascade.empty() or mouth_cascade.empty():
        st.error("Error: One or more cascade classifiers failed to load.")
        return None, None, None
    
    return face_cascade, eye_cascade, mouth_cascade

face_cascade, eye_cascade, mouth_cascade = load_cascades()
if None in (face_cascade, eye_cascade, mouth_cascade):
    st.stop()

# Function to detect facial features
def detect_facial_features(frame, scaleFactor=1.3, minNeighbors=5):
    if frame is None:
        return frame
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
        mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=11)
        for (mx, my, mw, mh) in mouths:
            cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 255, 255), 2)
    
    return frame

# Sidebar settings
st.sidebar.header("Detection Parameters")
scale_factor = st.sidebar.slider("Scale Factor", 1.1, 2.0, 1.3, 0.1)
min_neighbors = st.sidebar.slider("Minimum Neighbors", 1, 10, 5)
source = st.radio("Select Input Source", ["Upload Video", "Webcam"])

if source == "Upload Video" and video_file_buffer:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file_buffer.read())
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = detect_facial_features(frame, scale_factor, min_neighbors)
        stframe.image(processed_frame, channels="BGR")
    
    cap.release()
    os.unlink(tfile.name)

elif source == "Webcam":
    cap = None
    for i in range(3):  # Try multiple camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            break
    
    if cap is None or not cap.isOpened():
        st.error("Error: Could not access webcam. Make sure it is not being used by another application.")
    else:
        stframe = st.empty()
        stop_button = st.button("Stop Webcam")
        
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video from webcam.")
                break
            processed_frame = detect_facial_features(frame, scale_factor, min_neighbors)
            stframe.image(processed_frame, channels="BGR")
        
        cap.release()

st.sidebar.markdown("""
### Color Legend
- **Blue**: Face detection
- **Green**: Eye detection
- **Yellow**: Mouth detection
""")
