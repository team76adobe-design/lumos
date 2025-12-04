import io
import base64
import torch
from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import huggingface_hub
huggingface_hub.cached_download = huggingface_hub.hf_hub_download
from diffusers import StableDiffusionUpscalePipeline


app = FastAPI()

pipe = None
device = "cuda" if torch.cuda.is_available() else "cpu"


@app.post("/load")
def load_model():
    global pipe

    if pipe is not None:
        return {"status": "already_loaded"}

    model_id = "stabilityai/stable-diffusion-x4-upscaler"

    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to(device)

    return {
        "status": "model_loaded",
        "device": device,
        "dtype": "float16",
    }


@app.post("/run")
async def run(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    global pipe

    if pipe is None:
        return {"error": "Model not loaded. Use /load first."}


    img_bytes = await file.read()
    low_res = Image.open(io.BytesIO(img_bytes)).convert("RGB")


    low_res = low_res.resize((512, 512))


    with torch.autocast("cuda", dtype=torch.float16):
        upscaled = pipe(prompt=prompt, image=low_res).images[0]


    buf = io.BytesIO()
    upscaled.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return {"upscaled_image_base64": img_b64}


@app.post("/unload")
def unload_model():
    global pipe

    if pipe is None:
        return {"status": "already_unloaded"}

    del pipe
    pipe = None
    torch.cuda.empty_cache()

    return {"status": "model_unloaded"}
