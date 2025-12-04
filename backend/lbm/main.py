import io
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from diffusers.utils import load_image
from lbm.inference import evaluate, get_model
from PIL import Image
import base64
app = FastAPI(title="LBM Relighting API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
DEVICE = "cuda"

@app.post("/load_model")
async def load_model():
    global MODEL

    if MODEL is not None:
        return {"status": "already_loaded"}

    try:
        MODEL = get_model(
            "jasperai/LBM_relighting",
            torch_dtype=torch.bfloat16,
            device=DEVICE,
        )
        return {
            "status": "loaded",
            "model": "jasperai/LBM_relighting",
            "device": DEVICE
        }

    except Exception as e:
        return {"error": str(e)}




@app.post("/run_relighting")
async def run_relighting(
    image: UploadFile = File(...),
    steps: int = Form(1)
):
    global MODEL
    if MODEL is None:
        return {"error": "Model is not loaded. Call /load_model first."}

    
    image_bytes = await image.read()
    input_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    try:
        print("[INFO] Running evaluate()")

        out_pil = evaluate(MODEL, input_pil, num_sampling_steps=steps)

        
        w, h = input_pil.size
        out_pil = out_pil.resize((w, h))

       
        buf = io.BytesIO()
        out_pil.save(buf, format="PNG")
        out_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

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

        return {"status": "freed_gpu_memory"}

    except Exception as e:
        return {"error": str(e)}



@app.get("/")
def root():
    return {"msg": "LBM Relighting API Running"}



