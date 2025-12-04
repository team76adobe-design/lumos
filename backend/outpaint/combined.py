import io
import cv2
import base64
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# ========== INPAINT IMPORTS ==========
from iopaint.download import cli_download_model
from iopaint.model_manager import ModelManager
from iopaint.helper import decode_base64_to_image, pil_to_bytes, concat_alpha_channel
from iopaint.model.utils import torch_gc

# ========== OUTPAINT IMPORTS ==========
from .model import load_model as outpaint_load_model
from .model import unload_model as outpaint_unload_model
from .model import run_outpaint

# ============================================================
# APP INIT
# ============================================================
app = FastAPI(title="Unified IOPaint API (Separate Loaders)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda")

# Model holders
INPAINT_MODEL = None
OUTPAINT_READY = False


# ============================================================
# 1) LOAD / UNLOAD FOR INPAINT ONLY
# ============================================================
@app.post("/load_inpaint")
async def load_inpaint():
    global INPAINT_MODEL
    if INPAINT_MODEL is not None:
        return {"status": "inpaint_already_loaded"}

    cli_download_model("runwayml/stable-diffusion-v1-5")

    INPAINT_MODEL = ModelManager(
        name="runwayml/stable-diffusion-v1-5",
        device=device,
        enable_powerpoint_v2=True,
        disable_nsfw=True,
        sd_cpu_textencoder=False,
        cpu_offload=False,
        local_files_only=False,
        enable_brushnet=False
    )

    return {"status": "inpaint_loaded"}


@app.post("/unload_inpaint")
async def unload_inpaint():
    global INPAINT_MODEL
    if INPAINT_MODEL is None:
        return {"status": "inpaint_already_unloaded"}

    del INPAINT_MODEL
    INPAINT_MODEL = None
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch_gc()

    return {"status": "inpaint_unloaded"}


# ============================================================
# 2) LOAD / UNLOAD FOR OUTPAINT ONLY
# ============================================================
@app.post("/load_outpaint")
async def load_outpaint():
    global OUTPAINT_READY
    if OUTPAINT_READY:
        return {"status": "outpaint_already_loaded"}

    status = outpaint_load_model()
    OUTPAINT_READY = True
    return {"status": status}


@app.post("/unload_outpaint")
async def unload_outpaint():
    global OUTPAINT_READY
    if not OUTPAINT_READY:
        return {"status": "outpaint_already_unloaded"}

    status = outpaint_unload_model()
    OUTPAINT_READY = False
    return {"status": status}


# ============================================================
# 3) INPAINT INFERENCE
# ============================================================
@app.post("/inpaint")
async def inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(""),
    negative: str = Form("")
):
    global INPAINT_MODEL

    if INPAINT_MODEL is None:
        return {"error": "Inpaint model not loaded. Call /load_inpaint"}

    img_b64 = base64.b64encode(await image.read()).decode()
    mask_b64 = base64.b64encode(await mask.read()).decode()

    # decode + preprocess
    image_np, alpha, infos, ext = decode_base64_to_image(img_b64)
    mask_np, _, _, _ = decode_base64_to_image(mask_b64, gray=True)

    mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)[1]
    mask_np = cv2.dilate(mask_np, np.ones((15, 15), np.uint8), iterations=1)

    # inference
    class Req(BaseModel):
        image: str
        mask: str
        prompt: str
        negative_prompt: str

    req = Req(image=img_b64, mask=mask_b64, prompt=prompt, negative_prompt=negative)

    out_np = INPAINT_MODEL(image_np, mask_np, req)
    torch_gc()

    out_rgb = cv2.cvtColor(out_np.astype(np.uint8), cv2.COLOR_BGR2RGB)
    out_rgba = concat_alpha_channel(out_rgb, alpha)

    out_bytes = pil_to_bytes(Image.fromarray(out_rgba), ext=ext, quality=100, infos=infos)
    out_b64 = base64.b64encode(out_bytes).decode()

    return {"image_base64": out_b64}


# ============================================================
# 4) OUTPAINT INFERENCE
# ============================================================
class OutpaintReq(BaseModel):
    image_base64: str
    scale: float = 1.2
    positive_prompt: str = ""
    negative_prompt: str = ""


@app.post("/outpaint")
async def outpaint(req: OutpaintReq):
    global OUTPAINT_READY

    if not OUTPAINT_READY:
        return {"error": "Outpaint model not loaded. Call /load_outpaint"}

    out_bytes = run_outpaint(
        image_b64=req.image_base64,
        scale=req.scale,
        prompt=req.positive_prompt,
        negative_prompt=req.negative_prompt
    )

    return {"image_base64": base64.b64encode(out_bytes).decode()}


# ============================================================
@app.get("/")
def root():
    return {"message": "Unified API Ready (Separate model loaders)"}
