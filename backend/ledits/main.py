import io
import base64
import torch
from typing import Optional, List
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import ledits_patch
from leditspp import StableDiffusionPipeline_LEDITS
from leditspp.scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject


app = FastAPI(title="LEDITS++ Manager API")
pipe = None
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/load_ledits")
async def load_ledits():
    global pipe

    if pipe is not None:
        return {"status": "already_loaded"}

    pipe = StableDiffusionPipeline_LEDITS.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        safety_checker=None
    )

    pipe.scheduler = DPMSolverMultistepSchedulerInject.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="scheduler",
        algorithm_type="sde-dpmsolver++",
        solver_order=2
    )

    pipe.to("cuda")

    return {"status": "loaded", "device": "cuda"}



@app.post("/run_ledits")
async def run_ledits(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    thresholds: str = Form("0.7,0.9"),
    guidance: str = Form("3,4"),
    reverse: str = Form("false,false"),
    save_result: bool = Form(True)  
):
    global pipe

    if pipe is None:
        return {"error": "Model not loaded. Call /load_ledits first."}

    
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((512, 512))

    
    edit_prompt = [p.strip() for p in prompt.split(",")]
    edit_thresholds = [float(x) for x in thresholds.split(",")]
    edit_guidance = [float(x) for x in guidance.split(",")]
    reverse_edit = [(x.lower() == "true") for x in reverse.split(",")]

    
    gen = torch.manual_seed(42)

    with torch.no_grad():
        _ = pipe.invert(
            img,
            num_inversion_steps=50,
            generator=gen,
            verbose=False,
            skip=0.15
        )

        out = pipe(
            editing_prompt=edit_prompt,
            edit_threshold=edit_thresholds,
            edit_guidance_scale=edit_guidance,
            reverse_editing_direction=reverse_edit,
            use_intersect_mask=True,
        )

    result_img = out.images[0]

    save_path = "ledits_output.png"
    result_img.save(save_path)

    return {
        "message": "LEDITS++ edit complete",
        "saved_to": save_path
    }



@app.post("/free_ledits")
async def free_ledits():
    global pipe

    if pipe is None:
        return {"status": "already_empty"}

    pipe = None
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return {"status": "freed"}


@app.get("/")
def root():
    return {"message": "LEDITS++ Manager API running!"}



