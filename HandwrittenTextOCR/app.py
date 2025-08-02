import gradio as gr
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np
import cv2
from paddleocr import TextDetection
from spaces import GPU  

MODEL_HUB_ID = "imperiusrex/Handwritten_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🔄 Loading models...")

processor = TrOCRProcessor.from_pretrained(MODEL_HUB_ID)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_HUB_ID)
model.to(device)
model.eval()

ocr_det_model = TextDetection(model_name="PP-OCRv5_server_det")

print("✅ Models loaded successfully.")

@GPU  
def recognize_handwritten_text(image_input):
    if image_input is None:
        return "Please upload an image."

    image_pil = Image.fromarray(image_input).convert("RGB")

    detection_results = ocr_det_model.predict(image_input, batch_size=1)

    detected_polys = []
    for res in detection_results:
        polys = res.get('dt_polys', [])
        if polys is not None:
            detected_polys.extend(polys.tolist())

    cropped_images = []
    if detected_polys:
        img_np = np.array(image_pil)

        for box in detected_polys:
            box = np.array(box, dtype=np.float32)

            width = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
            height = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))

            dst_rect = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)

            M = cv2.getPerspectiveTransform(box, dst_rect)
            warped = cv2.warpPerspective(img_np, M, (width, height))
            cropped_images.append(Image.fromarray(warped).convert("RGB"))

        cropped_images.reverse()

    recognized_texts = []
    if cropped_images:
        for crop_img in cropped_images:
            pixel_values = processor(images=crop_img, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_new_tokens=64)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                recognized_texts.append(generated_text)
    else:
        pixel_values = processor(images=image_pil, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_new_tokens=64)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            recognized_texts.append("No text boxes detected. Full image OCR:\n" + generated_text)

    return "\n".join(recognized_texts)

def build_interface():
    return gr.Interface(
        fn=recognize_handwritten_text,
        inputs=gr.Image(type="numpy", label="Upload Handwritten Image"),
        outputs="text",
        title="✍️ Handwritten Text Recognition",
        description="📷 Upload a handwritten image. Uses PaddleOCR (detection) + TrOCR (recognition).",
    )

if __name__ == "__main__":
    iface = build_interface()
    iface.launch()
