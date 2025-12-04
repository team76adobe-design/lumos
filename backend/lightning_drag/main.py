import io
import base64
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
from lightning_drag_inference import load_lightningdrag, run_inference

app = FastAPI(title="LightningDrag API")

PIPE = None

BASE_SD_PATH = "/workspace/checkpoints/dreamshaper-8-inpainting"
VAE_PATH = "/workspace/checkpoints/sd-vae-ft-ema"
IP_ADAPTER_PATH = "/workspace/checkpoints/IP-Adapter/models"
LIGHTNING_DRAG_PATH = "/workspace/checkpoints/lightning-drag-sd15"
LCM_LORA_PATH = "/workspace/checkpoints/lcm-lora-sdv1-5/pytorch_lora_weights.safetensors"
DEVICE = "cuda"


 
@app.post("/load_model")
async def load_model():
    global PIPE
    if PIPE is not None:
        return {"status": "already_loaded"}
    try:
        PIPE = load_lightningdrag(
            base_sd_path=BASE_SD_PATH,
            vae_path=VAE_PATH,
            ip_adapter_path=IP_ADAPTER_PATH,
            lightning_drag_path=LIGHTNING_DRAG_PATH,
            lcm_lora_path=LCM_LORA_PATH,
            device=DEVICE
        )
        return {
            "status": "loaded",
            "device": DEVICE,
            "base_sd": BASE_SD_PATH
        }
    except Exception as e:
        return {"error": str(e)}



@app.post("/run_drag")
async def run_drag(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    handle_y: int = Form(...),
    handle_x: int = Form(...),
    target_y: int = Form(...),
    target_x: int = Form(...),
    num_inference_steps: int = Form(25),
    guidance_scale_points: float = Form(4.0),
    output_dir: str = Form("./outputs")
):
    global PIPE
    if PIPE is None:
        return {"error": "Model not loaded. Call /load_model first."}

    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = img.size

    if not (0 <= handle_x < width and 0 <= handle_y < height and
            0 <= target_x < width and 0 <= target_y < height):
        return {"error": "Points are out of image bounds"}

    image_path = "./tmp_input.png"
    mask_path = "./tmp_mask.png"
    img.save(image_path)

    mask_bytes = await mask.read()
    mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
    mask_img.save(mask_path)

    handle_points = [[handle_y, handle_x]]
    target_points = [[target_y, target_x]]

    try:
        out_paths = run_inference(
            pipe=PIPE,
            image_path=image_path,
            mask_path=mask_path,
            handle_points=handle_points,
            target_points=target_points,
            num_inference_steps=num_inference_steps,
            guidance_scale_points=guidance_scale_points,
            output_dir=output_dir
        )

        return {
            "status": "success",
            "saved_images": out_paths
        }

    except Exception as e:
        return {"error": str(e)}



@app.post("/unload_model")
async def unload_model():
    global PIPE
    if PIPE is None:
        return {"status": "already_empty"}
    del PIPE
    PIPE = None
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return {"status": "freed"}


@app.get("/")
def root():
    return {"message": "LightningDrag API Running with Hardcoded Paths!"}


