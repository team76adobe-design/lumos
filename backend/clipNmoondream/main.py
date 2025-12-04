import io
import torch
import base64
import clip
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="CLIP + Moondream Photo Defect Detector API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLIP_MODEL = None
CLIP_PREPROCESS = None
MOONDREAM = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

texts = [
    'Overexposed lighting', 'Dim lighting', 'visible scars/blemishes/acne on face',
    'Harsh shadows', 'Blown-out highlights', 'Murky low contrast', 'Excessive contrast',
    'Washed-out colors', 'Oversaturated colors', 'Unbalanced white balance',
    'Yellow color cast', 'Green color cast', 'Blue color cast', 'Magenta color cast',
    'Grainy noise', 'Pixelation', 'Motion blur', 'Unsharp focus', 'Back-focus issues',
    'Foreground blur', 'Lens flare streaks', 'Dirty lens spots', 'Chromatic aberration',
    'Barrel distortion', 'Pincushion distortion', 'Warped perspective', 'Tilted horizon',
    'Crooked composition', 'Bad cropping', 'Visible red skin blemishes',
    'Awkward framing', 'Distractions in background', 'Cluttered scene',
    'Overcrowded subjects', 'Empty boring backdrop', 'Uneven lighting patches',
    'Hotspots on skin (scars/blemishes/acne)', 'Red-eye effect', 'Over-smoothing skin',
    'Plastic-looking skin', 'Facial scars', 'AI deformities', 'Extra limbs',
    'Missing fingers', 'Incorrect anatomy', 'Distorted facial features',
    'Wrong eye symmetry', 'Off-angle faces', 'Uneven proportions', 'Bad bokeh',
    'Dirty sensor spots', 'Harsh vignetting', 'Color banding', 'Texture smearing',
    'Low dynamic range', 'Over HDR look', 'Extreme sharpening halos', 'Muddy shadows',
    'Posterization', 'Compression artifacts', 'Rough skin texture', 'Blurry text',
    'Ghosting', 'Double exposure mishap', 'Wrong aspect ratio', 'Stretching distortion',
    'Cropped head', 'Cropped limbs', 'Overbusy patterns', 'Uneven skin tones',
    'Flat lighting', 'Unflattering angles', 'Awkward poses', 'Perspective mismatch',
    'Wrong scale', 'Object floating', 'Misaligned elements', 'Inconsistent shadows',
    'Fake reflections', 'Cakey makeup', 'Aliasing', 'Color bleeding', 'Tilt-shift misuse',
    'Incorrect depth cues', 'Bad green screen key', 'Halo from keying',
    'Wrong color grading', 'Monochrome imbalance', 'Patchy retouching',
    'Oversized watermark', 'Distracting watermark', 'Cropped watermark leftovers',
    'Mismatched lighting sources', 'Texture repetition', 'Low-res upscaling artifacts',
    'Wrong perspective lines', 'Dark color crush', 'Oversoft background',
    'Fake depth blur rings', 'Object duplication error', 'Unnatural highlight edges'
]



@app.post("/load_models")
def load_models():
    global CLIP_MODEL, CLIP_PREPROCESS, MOONDREAM

    if CLIP_MODEL is not None and MOONDREAM is not None:
        return {"status": "already_loaded"}

    
    CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", device=DEVICE)

    
    MOONDREAM = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-06-21",
        trust_remote_code=True,
        device_map={"": DEVICE},
    )

    return {"status": "loaded", "device": DEVICE}


 
@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    topk: int = Form(3)
):
    global CLIP_MODEL, CLIP_PREPROCESS, MOONDREAM

    if CLIP_MODEL is None:
        return {"error": "Models not loaded. Call /load_models first."}


    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    img_input = CLIP_PREPROCESS(img).unsqueeze(0).to(DEVICE)

    
    text_tokens = clip.tokenize(texts).to(DEVICE)

   
    with torch.no_grad():
        img_feat = CLIP_MODEL.encode_image(img_input)
        txt_feat = CLIP_MODEL.encode_text(text_tokens)

    
    img_feat /= img_feat.norm(dim=-1, keepdim=True)
    txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

    
    sim = (img_feat @ txt_feat.T).squeeze(0)

    values, indices = sim.topk(topk)

    detected = [texts[idx] for idx in indices.tolist()]
    defects_string = ", ".join(detected)

    
    question = (
        f"Act as a professional photo editor. Analyze the image specifically for these "
        f"flaws: {defects_string}. For each flaw, confirm if it exists and describe what you see."
    )

    answer = MOONDREAM.query(img, question)["answer"]

    torch.cuda.empty_cache()

    return {
        "detected_defects": detected,
        "analysis": answer
    }



@app.post("/unload_models")
def unload_models():
    global CLIP_MODEL, CLIP_PREPROCESS, MOONDREAM

    CLIP_MODEL = None
    CLIP_PREPROCESS = None
    MOONDREAM = None

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return {"status": "models_unloaded"}
