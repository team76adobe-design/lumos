import uvicorn
import torch
import clip
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# -----------------------
# 1. DEFINE LABELS
# -----------------------
texts = [
    "abuse", "abduction", "adult_content", "adult_nudity", "aggressive_behavior", "alcohol", "altercation",
    "ammunition", "animal_abuse", "arson", "assault", "at_risk_individual", "banned_material", "bar_fight",
    "battlefield", "biological_hazard", "blood", "bodily_fluids", "body_horror", "bullying", "burning",
    "cartel_activity", "catastrophe", "censorship", "choking_hazard", "civil_unrest", "coercion", "combat",
    "contraband", "corruption", "crime", "criminal_organization", "cult_symbols", "dangerous_behavior",
    "dark_theme", "dead_body", "death", "deception", "defacement", "destructive_behavior", "detainment",
    "disaster", "disturbing_content", "domestic_violence", "drug_paraphernalia", "drug_use",
    "drunk_behavior", "endangered_person", "explicit_nudity", "exploitation", "extremist_propaganda",
    "extremist_symbol", "fear_inducing", "firearm", "fire_hazard", "forced_activity", "gambling",
    "gang_activity", "graphic_injury", "graphic_violence", "harassment", "hate_speech", "hate_symbol",
    "hazardous_material", "hostage_situation", "human_trafficking", "humiliation", "illegal_activity",
    "illegal_substances", "impersonation", "inappropriate_touching", "incident_scene", "injury",
    "intimidation", "intrusion", "looting", "manipulation", "medical_procedure", "medical_graphic",
    "minor_endangerment", "misconduct", "mistreatment", "mob_violence", "narcotics", "non_consensual_act",
    "nudity", "objectionable_behavior", "overdose", "panic", "peril", "physical_conflict", "poisoning",
    "police_raid", "predatory_behavior", "prohibited_content", "provoking", "psychological_harm",
    "public_intoxication", "questionable_behavior", "radicalization", "raid", "raw_injury", "repression",
    "risky_behavior", "robbery", "ruin", "scandal", "self_harm", "sensitive_content", "sexual_content",
    "sexual_exploitation", "sexual_harassment", "sexual_violence", "shocking_content", "smoke", "smuggling",
    "stalking", "substance_abuse", "suicidal_behavior", "terrorism", "terrorist_symbol", "threat",
    "toxic_exposure", "trafficking", "traumatic_content", "trespassing", "troubling_content", "unsettling",
    "vandalism", "verbal_abuse", "victimization", "violence", "violent_act", "violent_threat", "war",
    "warfare", "weapon", "weapon_brandishing", "wrongdoing"
]


# -----------------------
# 2. LOAD MODEL ON STARTUP
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Tokenize text once (for speed)
text_tokens = clip.tokenize(texts).to(device)

app = FastAPI(title="CLIP Content Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------
# 3. API ROUTE
# -----------------------
@app.post("/detect")
async def detect_content(
    file: UploadFile = File(...), 
    topk: int = Form(3)
):
    # Load image
    image = Image.open(file.file).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_tokens)

    # Normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Similarity
    sims = (image_features @ text_features.T).squeeze(0)

    # Top-K labels
    values, indices = sims.topk(topk)
    labels = [texts[i] for i in indices.tolist()]
    scores = [float(v) for v in values.tolist()]

    return {
        "top_k": topk,
        "labels": labels,
        "scores": scores
    }


