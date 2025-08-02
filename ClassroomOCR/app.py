import gradio as gr
import os
import cv2
import numpy as np
import torch
import spaces
from ultralytics import YOLO
from tqdm import tqdm

os.environ["YOLO_CONFIG_DIR"] = "/tmp"

device = "cuda" if torch.cuda.is_available() else "cpu"

extract_model = YOLO("best.pt").to(device)
detect_model  = YOLO("yolov8n.pt").to(device)

@spaces.GPU
def process_video(video_path):
    os.makedirs("frames", exist_ok=True)

    frames, idx = [], 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = extract_model(frame)
        labels = [extract_model.names[int(c)] for c in results[0].boxes.cls.cpu().numpy()]
        if "board" in labels and "person" not in labels:
            frames.append(frame)
            cv2.imwrite(f"frames/frame_{idx:04d}.jpg", frame)
        idx += 1
    cap.release()
    if not frames:
        raise RuntimeError("No frames with only 'board' and no 'person' found.")

    def align_frames(ref, tgt):
        orb = cv2.ORB_create(500)
        k1, d1 = orb.detectAndCompute(ref, None)
        k2, d2 = orb.detectAndCompute(tgt, None)
        if d1 is None or d2 is None:
            return None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(d1, d2)
        if len(matches) < 10:
            return None
        src = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
        return None if H is None else cv2.warpPerspective(tgt, H, (ref.shape[1], ref.shape[0]))

    base = frames[0]
    aligned = [base]
    for f in tqdm(frames[1:], desc="Aligning"):
        a = align_frames(base, f)
        if a is not None:
            aligned.append(a)
    if not aligned:
        raise RuntimeError("Alignment failed for all frames.")

    stack = np.stack(aligned, axis=0).astype(np.float32)
    median_board = np.median(stack, axis=0).astype(np.uint8)
    cv2.imwrite("clean_board.jpg", median_board)

    sum_img = np.zeros_like(aligned[0], dtype=np.float32)
    count = np.zeros(aligned[0].shape[:2], dtype=np.float32)
    for f in tqdm(aligned, desc="Masking persons"):
        res = detect_model(f, verbose=False)
        m = np.zeros(f.shape[:2], dtype=np.uint8)
        for box in res[0].boxes:
            if detect_model.names[int(box.cls)] == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(m, (x1, y1), (x2, y2), 255, -1)
        inv = cv2.bitwise_not(m)
        masked = cv2.bitwise_and(f, f, mask=inv)
        sum_img += masked.astype(np.float32)
        count += (inv > 0).astype(np.float32)

    count[count == 0] = 1
    selective = (sum_img / count[:, :, None]).astype(np.uint8)
    cv2.imwrite("fused_board_selective.jpg", selective)

    blur = cv2.GaussianBlur(selective, (5, 5), 0)
    sharp = cv2.addWeighted(selective, 1.5, blur, -0.5, 0)
    cv2.imwrite("sharpened_board_color.jpg", sharp)

    return "sharpened_board_color.jpg"


demo = gr.Interface(
    fn=process_video,
    inputs=[
        gr.File(
            label="Upload Classroom Video (.mp4)",
            file_types=['.mp4'],
            file_count="single",
            type="filepath"
        )
    ],
    outputs=[
        gr.Image(label="Sharpened Final Board")
    ],
    title="📹 Classroom Board Cleaner",
    description=(
        "Upload your classroom video (.mp4). \n"
        "Automatic extraction, alignment, masking, fusion & sharpening. \n"
        "View three stages of the cleaned board output."
    )
)

if __name__ == "__main__":
    if device == "cuda":
        print(f"[INFO] ✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] ⚠️ Using CPU (GPU not available or not assigned)")
    demo.launch()
