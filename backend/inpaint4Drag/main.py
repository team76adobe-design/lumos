import io
import base64
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from helper_functions import get_inpaint_pipeline, drag_inpaint
import gc

app = FastAPI(title="Inpaint4Drag API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_PIPE = None

@app.post("/load_model")
async def load_model():
    global _PIPE

    if _PIPE is not None:
        return {"status": "already_loaded"}

    try:
        _PIPE = get_inpaint_pipeline(device="cuda")
        return {"status": "loaded", "device": "cuda"}

    except Exception as e:
        return {"error": str(e)}



@app.post("/run_drag")
async def run_drag(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...),
    num_steps: int = Form(8),
    guidance_scale: float = Form(1.0),
    strength: float = Form(1.0),
):
    global _PIPE
    if _PIPE is None:
        return {"error": "Model not loaded. Call /load_model first."}

    
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img)

    
    mask_bytes = await mask.read()
    mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
    mask_np = np.array(mask_img)
    H, W = mask_np.shape

    
    if not (0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H):
        return {"error": "Drag points are outside image bounds."}

    points = [(x1, y1), (x2, y2)]

    try:
        
        result = drag_inpaint(
            image=img_np,
            mask=mask_np,
            points=points,
            output_dir=None,
            device="cuda",
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            strength=strength,
        )

        
        out_img = Image.fromarray(result)
        buffer = io.BytesIO()
        out_img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "status": "success",
            "image_base64": img_base64
        }

    except Exception as e:
        return {"error": str(e)}



@app.post("/unload_model")
async def unload_model():
    global _PIPE

    if _PIPE is None:
        return {"status": "already_empty"}

    try:
        del _PIPE
        _PIPE = None
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()

        return {"status": "freed_all_gpu_memory"}

    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def root():
    return {"message": "Inpaint4Drag API Running!"}


