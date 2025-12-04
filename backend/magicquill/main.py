import os
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, APIRouter, UploadFile
from fastapi.responses import JSONResponse,Response
from PIL import Image
import io
import asyncio
from pydantic import BaseModel, Field
from typing import Optional
import cv2
import numpy as np
import base64
from PIL import Image, UnidentifiedImageError, ImageDraw, ImageOps
import json
from typing import List
from MagicQuill import folder_paths
from MagicQuill.llava_new import LLaVAModel
from MagicQuill.scribble_color_edit import ScribbleColorEditModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MagicQuill API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()

MODEL = None  
MODEL_LOCK = asyncio.Lock()
WARMED = True



def _allocate_device(device: str) -> str:
    if device.startswith('cuda') and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'



@app.post('/load')
async def load_model(revision: str | None = Form(default=None), device: str = Form(default='cuda')):
    """Load the MagicQuill model into GPU/CPU memory and perform a warmup pass.

    Args:
        revision: Optional specific model revision to load. If omitted, latest.
        device: Preferred device (cuda/cpu); auto-falls back if unavailable.
    """
    global MODEL
    dev = _allocate_device(device)
    async with MODEL_LOCK:
        if MODEL is not None:
            return {"status": "already_loaded", "device": dev, "warmed": WARMED}
        try:
            kwargs = {
                'trust_remote_code': True,
            }
            llavaModel = LLaVAModel()
            scribbleColorEditModel = ScribbleColorEditModel()
            if revision:
                kwargs['revision'] = revision
            MODEL = {
                'llava':llavaModel,
                'sd1.5':scribbleColorEditModel
            }
        except Exception as e:
            MODEL = None
            raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
        return {"status": "loaded", "device": dev, "warmed": WARMED}


        
def read_base64_image(base64_image):
    if base64_image.startswith("data:image/png;base64,"):
        base64_image = base64_image.split(",")[1]
    elif base64_image.startswith("data:image/jpeg;base64,"):
        base64_image = base64_image.split(",")[1]
    elif base64_image.startswith("data:image/webp;base64,"):
        base64_image = base64_image.split(",")[1]
    else:
        raise ValueError("Unsupported image format.")
    image_data = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_data))
    image = ImageOps.exif_transpose(image)
    return image

def create_alpha_mask(base64_image):
    """Create an alpha mask from the alpha channel of an image."""
    image = read_base64_image(base64_image)
    mask = torch.zeros((1, image.height, image.width), dtype=torch.float32, device="cpu")
    if 'A' in image.getbands():
        alpha_channel = np.array(image.getchannel('A')).astype(np.float32) / 255.0
        mask[0] = 1.0 - torch.from_numpy(alpha_channel)
    return mask







def load_and_preprocess_image(base64_image, convert_to='RGB', has_alpha=False):
    """Load and preprocess a base64 image."""
    image = read_base64_image(base64_image)
    image = image.convert(convert_to)
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array)[None,]
    return image_tensor


def tensor_to_base64(tensor):
    tensor = tensor.squeeze(0) * 255.
    pil_image = Image.fromarray(tensor.cpu().byte().numpy())
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def prepare_images_and_masks(total_mask, original_image, add_color_image, add_edge_image, remove_edge_image):
    total_mask = create_alpha_mask(total_mask)
    original_image_tensor = load_and_preprocess_image(original_image)
    if add_color_image:
        add_color_image_tensor = load_and_preprocess_image(add_color_image)
    else:
        add_color_image_tensor = original_image_tensor
    
    add_edge_mask = create_alpha_mask(add_edge_image) if add_edge_image else torch.zeros_like(total_mask)
    remove_edge_mask = create_alpha_mask(remove_edge_image) if remove_edge_image else torch.zeros_like(total_mask)
    return add_color_image_tensor, original_image_tensor, total_mask, add_edge_mask, remove_edge_mask








def guess(original_image_tensor, add_color_image_tensor, add_edge_mask):
    llavaModel=MODEL['llava']
    description, ans1, ans2 = llavaModel.process(original_image_tensor, add_color_image_tensor, add_edge_mask)
    ans_list = []
    if ans1 and ans1 != "":
        ans_list.append(ans1)
    if ans2 and ans2 != "":
        ans_list.append(ans2)

    return ", ".join(ans_list)


def guess_prompt_handler(original_image, add_color_image, add_edge_image):
    original_image_tensor=original_image
    add_color_image_tensor=add_color_image
    width, height = original_image_tensor.shape[1], original_image_tensor.shape[2]
    add_edge_mask = create_alpha_mask(add_edge_image) if add_edge_image else torch.zeros((1, height, width), dtype=torch.float32, device="cpu")
    res = guess(original_image_tensor, add_color_image_tensor, add_edge_mask)
    return res

    


        
def generate(ckpt_name, total_mask, original_image, add_color_image, add_edge_image, remove_edge_image, positive_prompt, negative_prompt, grow_size, stroke_as_edge, fine_edge, edge_strength, color_strength, inpaint_strength, seed, steps, cfg, sampler_name, scheduler):
    scribbleColorEditModel=MODEL['sd1.5']
    add_color_image, original_image, total_mask, add_edge_mask, remove_edge_mask = prepare_images_and_masks(total_mask, original_image, add_color_image, add_edge_image, remove_edge_image)
    progress = None
    positive_prompt=guess_prompt_handler(original_image, add_color_image, add_edge_image)
    if torch.sum(remove_edge_mask).item() > 0 and torch.sum(add_edge_mask).item() == 0:
        if positive_prompt == "":
            positive_prompt = "empty scene"
        edge_strength /= 3.
    
    latent_samples, final_image, lineart_output, color_output = scribbleColorEditModel.process(
        ckpt_name,
        original_image, 
        add_color_image, 
        positive_prompt, 
        negative_prompt, 
        total_mask, 
        add_edge_mask, 
        remove_edge_mask, 
        grow_size, 
        stroke_as_edge, 
        fine_edge,
        edge_strength, 
        color_strength,  
        inpaint_strength, 
        seed, 
        steps, 
        cfg, 
        sampler_name, 
        scheduler,
        progress
    )

    final_image_base64 = tensor_to_base64(final_image)
    return final_image_base64


def get_hw_from_base64(base64_str):
    """
    Returns (height, width) from a base64 encoded image string.
    Accepts both raw base64 and 'data:image/...;base64,' prefixed formats.
    """
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes))

    return img.height, img.width

model_name = os.path.join('SD1.5', 'realisticVisionV60B1_v51VAE.safetensors')


import base64
from fastapi.responses import Response

@app.post('/edit_image')
async def magic_image(
    original_image: str = Form(...),
    total_mask: str = Form(...),
    add_color_image: Optional[str] = Form(None),
    add_edge_image: str = Form(...),
    remove_edge_image: str = Form(...),
    positive_prompt: Optional[str] = Form(None),
    negative_prompt: Optional[str] = Form(None)
):
    
    
    # --- Generate base64 image ---
    try:
        final_base64 = generate(
            ckpt_name=model_name,
            total_mask=total_mask,
            original_image=original_image,
            add_color_image=add_color_image,
            add_edge_image=add_edge_image,
            remove_edge_image=remove_edge_image,
            positive_prompt=None,
            negative_prompt=negative_prompt or (
                "out of frame, lowres, error, cropped, worst quality, low quality, "
                "jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, "
                "mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, "
                "extra limbs, disfigured, gross proportions, malformed limbs, watermark, signature"
            ),
            grow_size=15,
            stroke_as_edge='enable',
            fine_edge='disable',
            edge_strength=0.55,
            color_strength=0.55,
            inpaint_strength=1,
            seed=-1,
            steps=20,
            cfg=5,
            sampler_name="euler_ancestral",
            scheduler='karras'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

    if "," in final_base64:
        final_base64 = final_base64.split(",")[1]

    try:
        image_bytes = base64.b64decode(final_base64)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Base64 decode failed: {str(e)}")

    return Response(content=image_bytes, media_type="image/png")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {repr(exc)}"}
    )



@app.get('/')
def root():
    return {"message": "MagicQuill API running", "loaded": MODEL is not None}

@app.post('/unload')
async def unload_model():
    """Unload the model and free GPU memory."""
    global MODEL, WARMED
    async with MODEL_LOCK:
        if MODEL is None:
            return {"status": "not_loaded"}
        try:
            del MODEL['llava']
            del MODEL['sd1.5']
        except Exception:
            pass
        MODEL = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"status": "unloaded"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8069, reload=False)