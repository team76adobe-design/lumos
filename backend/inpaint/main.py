import io
import base64
import numpy as np
import cv2
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from PIL import Image
from pydantic import BaseModel
from iopaint.download import cli_download_model
from iopaint.model_manager import ModelManager
from iopaint.helper import decode_base64_to_image, pil_to_bytes, concat_alpha_channel
from iopaint.model.utils import torch_gc

app = FastAPI(title="IOPaint SD1.5 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda")
MODEL = None  


@app.post("/load_model")
async def load_model():
    global MODEL

    if MODEL is not None:
        return {"status": "already_loaded"}

    cli_download_model("runwayml/stable-diffusion-v1-5")

    MODEL = ModelManager(
        name="runwayml/stable-diffusion-v1-5",
        device=device,
        enable_powerpoint_v2=True,
        disable_nsfw=True,
        sd_cpu_textencoder=False,
        cpu_offload=False,
        local_files_only=False,
        enable_brushnet=False
    )

    return {"status": "loaded"}




def inpaint(image: str, mask: str, positive_prompt: str, negative_prompt: str):

    class InpaintRequest(BaseModel):
        image: Optional[str]
        mask: Optional[str]
        ldm_steps: int = 20
        ldm_sampler: str = "plms"
        zits_wireframe: bool = True
        hd_strategy: str = "Crop"
        hd_strategy_crop_trigger_size: int = 800
        hd_strategy_crop_margin: int = 128
        hd_strategy_resize_limit: int = 1280
        prompt: str = ""
        negative_prompt: str = ""
        use_croper: bool = False
        croper_x: int = 0
        croper_y: int = 0
        croper_height: int = 512
        croper_width: int = 512
        use_extender: bool = False
        extender_x: int = 0
        extender_y: int = 0
        extender_width: int = 640
        extender_height: int = 640
        sd_scale: float = 1.0
        sd_mask_blur: int = 12
        sd_strength: float = 1.0
        sd_steps: int = 50
        sd_guidance_scale: float = 7.5
        sd_sampler: str = "UniPC"
        sd_seed: int = 42
        sd_match_histograms: bool = False
        sd_outpainting_softness: float = 20.0
        sd_outpainting_space: float = 20.0
        sd_lcm_lora: bool = False
        sd_keep_unmasked_area: bool = True
        cv2_flag: str = "INPAINT_NS"
        cv2_radius: int = 4
        paint_by_example_example_image: Optional[str] = None
        p2p_image_guidance_scale: float = 1.5
        enable_controlnet: bool = False
        controlnet_conditioning_scale: float = 0.4
        controlnet_method: str = "lllyasviel/control_v11p_sd15_canny"
        enable_brushnet: bool = False
        brushnet_method: str = "Sanster/brushnet_random_mask"
        brushnet_conditioning_scale: float = 1.0
        enable_powerpaint_v2: bool = True
        powerpaint_task: str = "object-remove"
        fitting_degree: float = 1.0

    request = InpaintRequest(
        image=image,
        mask=mask,
        prompt=positive_prompt,
        negative_prompt=negative_prompt
    )

    
    image_np, alpha_channel, infos, ext = decode_base64_to_image(image)
    mask_np, _, _, _ = decode_base64_to_image(mask, gray=True)

    mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)[1]
    mask_np = cv2.dilate(mask_np, np.ones((15, 15), np.uint8), iterations=1)

    out_np = MODEL(image_np, mask_np, request)
    torch_gc()

    out_rgb = cv2.cvtColor(out_np.astype(np.uint8), cv2.COLOR_BGR2RGB)
    out_rgba = concat_alpha_channel(out_rgb, alpha_channel)

    out_bytes = pil_to_bytes(Image.fromarray(out_rgba), ext=ext, quality=100, infos=infos)
    return out_bytes




@app.post("/run_inpaint")
async def run_inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(""),
    negative: str = Form("")
):
    global MODEL
    if MODEL is None:
        return {"error": "Model not loaded. Call /load_model first."}

    try:
        
        img_b64 = base64.b64encode(await image.read()).decode("utf-8")
        mask_b64 = base64.b64encode(await mask.read()).decode("utf-8")

        
        result_bytes = inpaint(img_b64, mask_b64, prompt, negative)

        
        out_b64 = base64.b64encode(result_bytes).decode("utf-8")

        return {
            "status": "success",
            "image_base64": out_b64
        }

    except Exception as e:
        return {"error": str(e)}




@app.post("/unload_model")
async def unload_model():
    global MODEL

    if MODEL is None:
        return {"status": "already_empty"}

    try:
        del MODEL
        MODEL = None

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch_gc()

        return {"status": "freed"}

    except Exception as e:
        return {"error": str(e)}



@app.get("/")
def root():
    return {"message": "IOPaint SD1.5 API running!"}




