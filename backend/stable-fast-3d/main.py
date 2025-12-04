import os
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
import rembg
import torch
from contextlib import nullcontext
from sf3d.system import SF3D
from sf3d.utils import remove_background, resize_foreground

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model placeholder
model = None
rembg_session = None
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.post("/load")
def load_model(pretrained_model: str = "stabilityai/stable-fast-3d"):
    global model, rembg_session
    if model:
        return {"status": "Model already loaded"}
    
    try:
        rembg_session = rembg.new_session()
        model = SF3D.from_pretrained(
            pretrained_model,
            config_name="config.yaml",
            weight_name="model.safetensors",
        )
        model.to(device)
        model.eval()
        return {"status": "Model loaded", "device": device}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/run")
async def run_inference(
    image: UploadFile = File(...),
    foreground_ratio: float = 0.85,
    texture_resolution: int = 1024,
    remesh_option: str = "none",
    target_vertex_count: int = -1,
):
    global model, rembg_session
    if not model:
        return JSONResponse(status_code=400, content={"error": "Model not loaded. Call /load first."})
    
    try:
        # Load and preprocess image
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGBA")
        pil_image = remove_background(pil_image, rembg_session)
        pil_image = resize_foreground(pil_image, foreground_ratio)
        
        with torch.no_grad():
            with torch.autocast(
                device_type=device, dtype=torch.bfloat16
            ) if "cuda" in device else nullcontext():
                mesh, _ = model.run_image(
                    [pil_image],
                    bake_resolution=texture_resolution,
                    remesh=remesh_option,
                    vertex_count=target_vertex_count,
                )
        # Export mesh as .glb to memory
        glb_path = "output/mesh.glb"
        os.makedirs("output", exist_ok=True)
        mesh.export(glb_path, include_normals=True)
        
        with open(glb_path, "rb") as f:
            mesh_data = base64.b64encode(f.read()).decode("utf-8")
        
        return {"status": "Success", "mesh_glb_base64": mesh_data}
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/unload")
def unload_model():
    global model
    if model:
        del model
        model = None
        torch.cuda.empty_cache()
        return {"status": "Model unloaded and GPU memory cleared"}
    else:
        return {"status": "No model to unload"}

@app.get("/")
def root():
    return {"message": "SF3D Model API. Use /load, /run, /unload."}
