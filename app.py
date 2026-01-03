import streamlit as st
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="Image Upload App",
    page_icon="🖼️",
    layout="wide"
)

# App title
st.title("Image Upload Application")
st.write("Upload an image to get started")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "bmp", "gif"]
)

# Display uploaded image
if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    
    # Display image information
    st.success("Image uploaded successfully!")
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("Image Details")
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**Image Size:** {image.size[0]} x {image.size[1]} pixels")
        st.write(f"**Image Mode:** {image.mode}")
        st.write(f"**File Size:** {uploaded_file.size / 1024:.2f} KB")
        
        # Optional: Add image format info
        if hasattr(image, 'format'):
            st.write(f"**Format:** {image.format}")

else:
    st.info("👆 Please upload an image file to begin")
