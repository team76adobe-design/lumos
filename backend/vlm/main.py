from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from io import BytesIO
from .model import load_model, unload_model, get_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/load_model/")
async def load_model_endpoint():
    load_model()
    return {"status": "Model loaded into GPU"}

@app.post("/caption/")
async def caption_image(image: UploadFile = File(...)):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Please call /load_model first.")

    contents = await image.read()
    pil_image = Image.open(BytesIO(contents)).convert("RGB")
    caption = model.caption(pil_image, length="normal")["caption"]
    return {"caption": caption}

@app.post("/query/")
async def query_image(question: str, image: UploadFile = File(...)):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Please call /load_model first.")
    
    contents = await image.read()
    pil_image = Image.open(BytesIO(contents)).convert("RGB")
    answer = model.query(pil_image, question)["answer"]
    return {"answer": answer}

@app.post("/unload_model/")
async def unload_model_endpoint():
    unload_model()
    return {"status": "Model unloaded from GPU"}


