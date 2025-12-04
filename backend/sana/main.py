import torch
import base64
import io
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import SanaPipeline
from nunchaku import NunchakuSanaTransformer2DModel
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
pipe = None
transformer = None


# ----------------------------
# Request Schema
# ----------------------------
class InferenceRequest(BaseModel):
    prompt: str
    height: int = 1024
    width: int = 1024
    guidance_scale: float = 4.5
    steps: int = 20
    seed: int = 42


# ----------------------------
# LOAD MODEL
# ----------------------------
@app.get("/load")
def load_model():
    global pipe, transformer

    if pipe is not None:
        return {"status": "already_loaded"}

    try:
        transformer = NunchakuSanaTransformer2DModel.from_pretrained(
            "nunchaku-tech/nunchaku-sana/svdq-int4_r32-sana1.6b.safetensors"
        )

        pipe = SanaPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
            transformer=transformer,
            variant="bf16",
            torch_dtype=torch.bfloat16,
        ).to("cuda")

        pipe.vae.to(torch.bfloat16)
        pipe.text_encoder.to(torch.bfloat16)

        return {"status": "model_loaded"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# UNLOAD MODEL (Free GPU)
# ----------------------------
@app.get("/unload")
def unload_model():
    global pipe, transformer

    try:
        pipe = None
        transformer = None
        torch.cuda.empty_cache()
        return {"status": "model_unloaded"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# RUN INFERENCE
# ----------------------------
@app.post("/run")
def run_inference(req: InferenceRequest):
    global pipe

    if pipe is None:
        raise HTTPException(status_code=400, detail="Model is not loaded. Call /load first.")

    try:
        generator = torch.Generator(device="cuda").manual_seed(req.seed)

        image = pipe(
            prompt=req.prompt,
            height=req.height,
            width=req.width,
            guidance_scale=req.guidance_scale,
            num_inference_steps=req.steps,
            generator=generator,
        ).images[0]

        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"image_base64": img_str}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
