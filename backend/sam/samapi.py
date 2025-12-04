import io
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import torch
from segment_anything import sam_model_registry, SamPredictor

app = FastAPI(title="SAM Segmentation API")

sam = None
predictor = None


# -------------------------------------------------------
# 1) LOAD SAM MODEL (NO CHECKPOINT ARG NEEDED)
# -------------------------------------------------------
@app.post("/load_sam")
async def load_sam():
    global sam, predictor

    if sam is not None:
        return {"status": "already_loaded"}

    checkpoint_path = "SAM.pth"

    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device="cuda")

    predictor = SamPredictor(sam)

    return {"status": "loaded", "checkpoint": checkpoint_path}


# -------------------------------------------------------
# 2) RUN SEGMENTATION AND RETURN COORDINATES + SAVE IMAGE
# -------------------------------------------------------
@app.post("/segment")
async def segment_object(
    file: UploadFile = File(...),
    xs: List[int] = Form(...),
    ys: List[int] = Form(...),
    labels: List[int] = Form(...),
    save_path: str = Form("sam_output.png")
):
    global sam, predictor
    if sam is None:
        return {"error": "SAM NOT LOADED — call /load_sam first."}

    if not (len(xs) == len(ys) == len(labels)):
        return {"error": "xs, ys, labels must have same length."}

    # Load input
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)

    # SAM requires 512x512 input
    resized = cv2.resize(img_np, (512, 512))
    predictor.set_image(resized)

    # Prepare points
    points = np.array(list(zip(xs, ys)), dtype=np.int32)
    lbls = np.array(labels, dtype=np.int32)

    # Predict mask
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=lbls,
        multimask_output=True
    )

    best = int(np.argmax(scores))
    mask_255 = (masks[best] * 255).astype(np.uint8)

    # -------------------------------------------------------
    # Extract contour coordinates
    # -------------------------------------------------------
    contours, _ = cv2.findContours(
        mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    coords = [c.reshape(-1, 2).tolist() for c in contours]

    # -------------------------------------------------------
    # Save mask image
    # -------------------------------------------------------
    mask_img = Image.fromarray(mask_255)
    mask_img.save(save_path)

    return {
        "status": "success",
        "mask_path": save_path,
        "mask_coords": coords,
        "confidence": float(scores[best])
    }


# -------------------------------------------------------
# 3) UNLOAD MODEL
# -------------------------------------------------------
@app.post("/free_sam")
async def free_sam():
    global sam, predictor

    if sam is None:
        return {"status": "already_empty"}

    del sam
    del predictor
    sam = None
    predictor = None

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return {"status": "freed"}


@app.get("/")
def root():
    return {"message": "SAM API running!"}




    
# curl -X POST https://<url>/load_sam

# curl -X POST "https://<ur_url>/segment" \
#   -H "accept: application/json" \
#   -F "file=@direwolf.png" \
#   -F "xs=250" -F "ys=275" -F "labels=1" \
#   -F "xs=180" -F "ys=260" -F "labels=0" \
#   -F "xs=260" -F "ys=310" -F "labels=1" \
#   -F "save_mask=true"

# curl -X POST "https://<url>/unload_sam"
