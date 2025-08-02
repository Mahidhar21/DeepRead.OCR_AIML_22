
# --- Imports ---
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import io
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

st.set_page_config(page_title="Drawing Canvas OCR", layout="wide")
st.title("📝 Real-Time Handwriting OCR Canvas")

st.sidebar.header("Canvas Controls")

# Sidebar controls
stroke_width = st.sidebar.slider("Stroke width", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color", "#000000")
background_color = st.sidebar.color_picker("Background color", "#FFFFFF")

# Tool selection
mode = st.sidebar.selectbox(
    "Tool",
    ("freedraw", "line", "rect", "circle", "transform", "polygon", "point", "eraser")
)

fill_color = st.sidebar.color_picker("Fill color (for shapes)", "#FFFFFF")

canvas_width = st.sidebar.number_input("Canvas width", 200, 1200, 600)
canvas_height = st.sidebar.number_input("Canvas height", 200, 1200, 400)


@st.cache_resource
def load_trocr_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

processor, model, device = load_trocr_model()

# --- OCR Function ---
def perform_ocr_from_npimg(np_img, processor, model, device):
    pil_img = Image.fromarray(np_img.astype(np.uint8))
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=64)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Draw here:")
    canvas_result = st_canvas(
        fill_color=fill_color+"77",  
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=background_color,
        background_image=None,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode=mode,
        key="canvas",
    )
    st.markdown("""
    - Use the sidebar to select pen/eraser, color, and size.<br>
    - Draw freehand, lines, rectangles, circles, polygons, or erase.<br>
    - The OCR result will update every 4 seconds automatically.<br>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("🧠 OCR Output (updates every 4 seconds):")
    if 'last_ocr' not in st.session_state:
        st.session_state.last_ocr = ""
    if 'last_ocr_time' not in st.session_state:
        st.session_state.last_ocr_time = 0

    import time
    now = time.time()
    if canvas_result.image_data is not None:
        img_bytes = canvas_result.image_data.tobytes()
        img_hash = hash(img_bytes)
        if 'last_img_hash' not in st.session_state or st.session_state.last_img_hash != img_hash or now - st.session_state.last_ocr_time > 4:
            with st.spinner("Recognizing handwriting..."):
                try:
                    ocr_text = perform_ocr_from_npimg(canvas_result.image_data, processor, model, device)
                except Exception as e:
                    ocr_text = f"[OCR error: {e}]"
            st.session_state.last_ocr = ocr_text
            st.session_state.last_ocr_time = now
            st.session_state.last_img_hash = img_hash
        else:
            ocr_text = st.session_state.last_ocr
        st.markdown(f"<div style='background:#181a20;padding:20px;border-radius:12px;min-height:120px;color:#ededed;font-family:monospace;'>{ocr_text}</div>", unsafe_allow_html=True)
        # Auto-refresh every 4 seconds
        st.experimental_rerun()
    else:
        st.info("Draw something on the canvas to see OCR output here.")
