# app.py
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import RRDBNet_arch as arch

# ----------------------
# Model setup
# ----------------------
@st.cache_resource  # caches the model to avoid reloading on every interaction
def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)
    return model, device

model_path = 'models/RRDB_ESRGAN_x4.pth'
model, device = load_model(model_path)

# ----------------------
# Streamlit UI
# ----------------------
st.title("ESRGAN Super-Resolution Demo")
st.write("Upload an image to enhance it with ESRGAN.")

uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Original Image', use_column_width=True)
    
    # Preprocess image
    img_input = img.astype(np.float32) / 255.0
    img_input = torch.from_numpy(np.transpose(img_input[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_input = img_input.unsqueeze(0).to(device)

    # Super-resolution
    with torch.no_grad():
        output = model(img_input).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)

    st.image(output, caption='Super-Resolution Image', use_column_width=True)

    # Option to download
    output_pil = Image.fromarray(output)
    st.download_button(
        label="Download Image",
        data=output_pil.tobytes(),
        file_name="sr_image.png",
        mime="image/png"
    )
