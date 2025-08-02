import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# --- 1. Define int_to_char mapping and decode_prediction function ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "captcha_recognition_model_char.keras")
INT_TO_CHAR_PATH = os.path.join(CURRENT_DIR, "int_to_char.json")

try:
    with open(INT_TO_CHAR_PATH, "r") as f:
        str_int_to_char_mapping = json.load(f)
        int_to_char = {int(k): v for k, v in str_int_to_char_mapping.items()}
    print(f"int_to_char mapping loaded successfully.")
except Exception as e:
    print(f"Error loading int_to_char.json: {e}")
    int_to_char = {i: chr(i + ord('A')) for i in range(26)}
    int_to_char.update({26 + i: str(i) for i in range(10)})
    int_to_char.update({36 + i: chr(i + ord('a')) for i in range(26)})
    int_to_char[0] = '<pad>'
    print("Using fallback int_to_char.")

fixed_solution_length = 5

def decode_prediction(prediction_output, int_to_char_mapping):
    predicted_indices = np.argmax(prediction_output, axis=-1)[0]
    predicted_chars = [int_to_char_mapping.get(idx, '') for idx in predicted_indices]
    return "".join([char for char in predicted_chars if char != '<pad>'])

def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded.")
        return model
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None

model = load_model()

# --- 2. Prediction function exposed to Gradio ---
def predict_captcha(image: Image.Image) -> str:
    if model is None:
        return "Error: Model not loaded."

    img = image.resize((200, 50))
    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)
    return decode_prediction(prediction, int_to_char)

# --- 3. Create and launch Gradio interface ---
iface = gr.Interface(
    fn=predict_captcha,
    inputs=gr.Image(type="pil", label="Upload Captcha Image"),
    outputs=gr.Textbox(label="Predicted Captcha"),
    title="Captcha Recognition",
    description="Upload a captcha image (200x50 pixels expected).",
    allow_flagging="never"
)

# Only required locally; not needed on Hugging Face Spaces.
if __name__ == "__main__":
    iface.launch()
