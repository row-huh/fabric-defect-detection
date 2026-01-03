import streamlit as st
from PIL import Image
import cv2
import numpy as np

st.set_page_config(
    page_title="Car Fabric Defect Detection",
    layout="wide"
)

def detect_car_fabric_defects(image):    
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
        
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        vis_image = img_array.copy()
    else:
        gray = img_array
        vis_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3,3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)
    

    thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

   
    defect_count = 0
    
    contours_edges, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_edges:
        area = cv2.contourArea(cnt)
        if area > 50: 
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(vis_image, (x,y), (x+w, y+h), (255, 0, 0), 2)
            defect_count += 1

    contours_thresh, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_thresh:
        area = cv2.contourArea(cnt)
        if area > 30: 
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(vis_image, (x,y), (x+w, y+h), (255, 0, 0), 2) 
            defect_count += 1

    is_flawed = defect_count > 0

    return {
        'is_flawed': is_flawed,
        'defect_count': defect_count,
        'visualization': vis_image,
        'blurred': blurred,
        'edges': edges_dilated,
        'threshold': thresh_clean
    }

st.title("Car Fabric Defect Detection")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)


st.sidebar.header("Detection Settings")
show_debug = st.sidebar.checkbox("Show Debug Layers", value=False)


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.success("Image uploaded successfully!")
    
    with st.spinner("Analyzing car fabric..."):
        results = detect_car_fabric_defects(image)
    
    st.markdown("---")
    
    if results['is_flawed']:
        st.error(f"**DEFECTS FOUND**: {results['defect_count']} issues detected")
    else:
        st.success("**PASS**: Fabric appears clean")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Defect Visualization")
        st.image(results['visualization'], use_container_width=True, caption="Detected Defects (Red Boxes)")
        
    if show_debug:
        st.markdown("---")
        st.subheader("Debug Layers (Basic CV Operations)")
        
        d1, d2, d3 = st.columns(3)
        
        with d1:
            st.image(results['blurred'], caption="Gaussian Blur", use_container_width=True)
        with d2:
            st.image(results['edges'], caption="Canny Edges (Dilated)", use_container_width=True)
        with d3:
            st.image(results['threshold'], caption="Threshold (Inverted)", use_container_width=True)

else:
    st.info("Please upload a car door fabric image to begin")