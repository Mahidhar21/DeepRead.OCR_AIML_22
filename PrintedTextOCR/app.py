import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from paddleocr import PaddleOCR, TextDetection
from functools import lru_cache


MODEL_HUB_ID = "imperiusrex/printedpaddle"
# Setup
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Set device to CPU
device = "cpu"
clip_model.to(device)

# Language map for OCR models
def process_image(img_path):
    """
    Processes an image to detect, crop, and OCR text, returning it in reading order.

    Args:
        img_path: The path to the image file.

    Returns:
        A string containing the reconstructed text.
    """
    # Load CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    # Candidate language phrases for detection
    candidates = [
        "This is English text",
        # "This is Hindi text",
        # "This is Tamil text",
        "This is Telugu text",
        # "This is Bengali text",
        # "This is Arabic text",
        "This is Chinese text",
        # "This is Japanese text",
        "This is Korean text",
        "This is Russian text",
        # "This is Kannada text",
        # "This is Malayalam text",
        # "This is Marathi text",
        # "This is Urdu text",
        "This is French text",
        # "This is Spanish text",
        # "This is Italian text",
        # "This is Portuguese text",
        # "This is Romanian text",
        # "This is Hungarian text",
        # "This is Indonesian text",
        # "This is Lithuanian text",
        # "This is Chinese Traditional text",
        # "This is Malay text",
        # "This is Dutch text",
        # "This is Norwegian text",
        # "This is Bosnian text",
        # "This is Polish text",
        # "This is Czech text",
        # "This is Slovak text",
        # "This is Welsh text",
        # "This is Slovenian text",
        # "This is Danish text",
        # "This is Albanian text",
        # "This is Estonian text",
        # "This is Swedish text",
        # "This is Irish text",
        # "This is Swahili text",
        # "This is Croatian text",
        # "This is Uzbek text",
        # "This is Turkish text",
        "This is Latin text",
        # "This is Belarusian text",
        # "This is Ukrainian text"
    ]

    # Map detected languages to PaddleOCR language codes
    lang_map = {
        "english": "en",
        # "hindi": "hi",
        # "tamil": "ta",
        "telugu": "te",
        # "bengali": "bn",
        # "arabic": "ar",
        "chinese": "ch",
        # "japanese": "japan",
        "korean": "korean",
        "russian": "ru",
        # "kannada": "kn",
        # "malayalam": "ml",
        # "marathi": "mr",
        # "urdu": "ur",
        "french": "fr",
        # "spanish": "es",
        # "italian": "it",
        # "portuguese": "pt",
        # "romanian": "ro",
        # "hungarian": "hu",
        # "indonesian": "id",
        # "lithuanian": "lt",
        # "chinese traditional": "chinese_cht",
        # "malay": "ms",
        # "dutch": "nl",
        # "norwegian": "no",
        # "bosnian": "bs",
        # "polish": "pl",
        # "czech": "cs",
        # "slovak": "sk",
        # "welsh": "cy",
        # "slovenian": "sl",
        # "danish": "da",
        # "albanian": "sq",
        # "estonian": "et",
        # "swedish": "sv",
        # "irish": "ga",
        # "swahili": "sw",
        # "croatian": "hr",
        # "uzbek": "uz",
        # "turkish": "tr",
        "latin": "la",
        # "belarusian": "be",
        # "ukrainian": "uk"
    }

    # Text Detection
    arr = []
    model_det = TextDetection(model_name="PP-OCRv5_server_det")
    output = model_det.predict(img_path, batch_size=1)
    for res in output:
        polys = res['dt_polys']
        if polys is not None:
            arr.extend(polys.tolist())
    arr = sorted(arr, key=lambda box: (box[0][1], box[0][0]))

    # Image Cropping and Warping
    img = cv2.imread(img_path)
    cropped_images = []
    for i, box in enumerate(arr):
        box = np.array(box, dtype=np.float32)
        width_a = np.linalg.norm(box[0] - box[1])
        width_b = np.linalg.norm(box[2] - box[3])
        height_a = np.linalg.norm(box[0] - box[3])
        height_b = np.linalg.norm(box[1] - box[2])
        width = int(max(width_a, width_b))
        height = int(max(height_a, height_b))
        dst_rect = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(box, dst_rect)
        warped = cv2.warpPerspective(img, M, (width, height))
        cropped_images.append(warped)

    # Perform language detection for each cropped image and then OCR
    predicted_texts = []
    for i, cropped_img in enumerate(cropped_images):
        # Get probabilities
        inputs = processor(text=candidates, images=cropped_img, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits_per_image = clip_model(**inputs).logits_per_image
            probs = logits_per_image.softmax(dim=1)

        # Get best language match
        best = probs.argmax().item()
        detected_lang_phrase = candidates[best]
        detected_lang = detected_lang_phrase.split()[-2].lower()
        lang_code = lang_map.get(detected_lang, "en")

        # Perform OCR for the current cropped image with the detected language
        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=lang_code,
            device="cpu"
        )

        result = ocr.predict(cropped_img)

        text_for_this_image = ""
        if result and result[0] and 'rec_texts' in result[0]:
             text_for_this_image = " ".join(result[0]['rec_texts'])

        predicted_texts.append(text_for_this_image)


    def get_box_center(box):
      """Calculates the center of a bounding box."""
      x_coords = [p[0] for p in box]
      y_coords = [p[1] for p in box]
      center_x = sum(x_coords) / len(x_coords)
      center_y = sum(y_coords) / len(y_coords)
      return center_x, center_y

    # --- Step 1: Read all text and their centroid coordinates ---
    all_text_blocks = []
    for i, box in enumerate(arr):
        # Use the predicted text from the list
        text = predicted_texts[i]
        if text: # Only add if text is not empty
            center_x, center_y = get_box_center(box)
            all_text_blocks.append({
                "text": text,
                "center_x": center_x,
                "center_y": center_y
            })


    # --- Step 2: Sort by y-coordinate, then by x-coordinate, and group into lines ---
    reconstructed_text = ""
    if all_text_blocks:
        # Sort by center_y, then by center_x
        sorted_blocks = sorted(all_text_blocks, key=lambda item: (item["center_y"], item["center_x"]))

        lines = []
        if sorted_blocks:
            current_line = [sorted_blocks[0]]
            for block in sorted_blocks[1:]:
                # Check if the vertical centers are close enough to be on the same line
                if abs(block["center_y"] - current_line[-1]["center_y"]) < 40: # Y-threshold
                    current_line.append(block)
                else:
                    # Sort the current line by x-coordinate and add it to the lines list
                    current_line.sort(key=lambda item: item["center_x"])
                    lines.append(" ".join([item["text"] for item in current_line]))
                    current_line = [block]

            # Add the last line
            if current_line:
                current_line.sort(key=lambda item: item["center_x"])
                lines.append(" ".join([item["text"] for item in current_line]))

        # --- Step 3: Join the lines into a single string ---
        reconstructed_text = "\n".join(lines)

    return reconstructed_text

iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Text(),
    title="Image OCR and Text Reconstruction",
    description="Upload an image to perform text detection, cropping, language detection, OCR, and reconstruct the text in reading order."
)

if __name__== "__main__":
    iface.launch(debug=True)

