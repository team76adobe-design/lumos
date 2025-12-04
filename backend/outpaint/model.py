# model_handler.py

import torch
from iopaint.model_manager import ModelManager
from iopaint.helper import decode_base64_to_image, pil_to_bytes
from iopaint.model.utils import torch_gc
from PIL import Image
import numpy as np
import cv2
import base64
from .utils import InpaintRequest, PowerPaintTask


model = None
device = None


def load_model():
    global model, device
    if model is not None:
        return "Model already loaded"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModelManager(
        name='runwayml/stable-diffusion-v1-5',
        device=device,
        enable_powerpoint_v2=True,
        disable_nsfw=True,
        sd_cpu_textencoder=False,
        local_files_only=False,
        cpu_offload=False,
    )

    return "Model loaded successfully"


def unload_model():
    global model, device

    if model is None:
        return "Model already unloaded"

    del model
    model = None

    torch_gc()
    return "Model unloaded and VRAM cleared"


def run_outpaint(image_b64, scale, prompt, negative_prompt):

    
    img_bytes = base64.b64decode(image_b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    H, W = img.shape[:2]

    extender_w = int(W * scale)
    extender_h = int(H * scale)
    extender_x = (extender_w - W) // 2
    extender_y = (extender_h - H) // 2

    
    mask = np.zeros((H, W), dtype=np.uint8)
    _, mask_buffer = cv2.imencode(".png", mask)
    mask_b64 = base64.b64encode(mask_buffer).decode()

    
    req = InpaintRequest(
        image=image_b64,
        mask=mask_b64,
        prompt=prompt,
        negative_prompt=negative_prompt,
        use_extender=True,
        extender_x=-extender_x,
        extender_y=-extender_y,
        extender_width=extender_w,
        extender_height=extender_h,
        enable_powerpaint_v2=True,
        powerpaint_task=PowerPaintTask.outpainting
    )

    
    mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    
    np_out = model(image_b64, mask_bin, req)

    
    np_out = cv2.cvtColor(np_out.astype(np.uint8), cv2.COLOR_BGR2RGB)
    img = Image.fromarray(np_out)
    out_bytes = pil_to_bytes(img, ext=".png")

    return out_bytes
