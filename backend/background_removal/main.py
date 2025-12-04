import io
import torch
from PIL import Image
import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import huggingface_hub
huggingface_hub.cached_download = huggingface_hub.hf_hub_download

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'


image_size = (512,512)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
])


@app.post("/load")
def load_model():
    global model

    if model is not None:
        return {"status": "already_loaded"}

    model = AutoModelForImageSegmentation.from_pretrained(
        "briaai/RMBG-2.0", trust_remote_code=True
    ).eval().to(device)

    return {"status": "model_loaded", "device": device}


@app.post("/run")
async def run_model(file: UploadFile = File(...)):
    global model
    if model is None:
        return {"error": "Model not loaded. Call /load first."}

    
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    original_size = image.size

    
    input_tensor = transform_image(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()


    pred_mask = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred_mask)


    mask = pred_pil.resize(original_size)
    image.putalpha(mask)


    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    base64_img = base64.b64encode(buffer.getvalue()).decode()

    return {"image_base64": base64_img}



@app.post("/unload")
def unload_model():
    global model

    if model is None:
        return {"status": "model already unloaded"}


    del model
    model = None
    torch.cuda.empty_cache()

    return {"status": "model_unloaded", "device": device}



