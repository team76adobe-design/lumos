import torch
import huggingface_hub
huggingface_hub.cached_download = huggingface_hub.hf_hub_download
from sentence_transformers import SentenceTransformer, util
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import numpy as np

# ------------------------------
# 1. Your LoRA list
# ------------------------------
lora_list = [{
    
    "name": "Greenery - SD 1.5 and SDXL",
    "description":"A Stable Diffusion 1.5 LoRA that transforms ordinary or urban scenes into lush, green natural environments. It adds dense foliage, moss, plants, and natural textures to images, making them appear overgrown or reclaimed by nature. The model enhances lighting with soft cinematic tones and vivid greenery while maintaining realistic detail and depth. Ideal for generating forests, jungles, natural landscapes, or turning man-made environments into eco-rich, overgrown sceneries. Works best with moderate strength to preserve human subjects while modifying the background.",
    "path_to_lora":"/workspace/S/sd1.5loras/Greenery.safetensors",
    "strength":0.9,
    "guidance":7.0,
    "num_inference_steps":200,
    "image_description": "A man standing in an urbanic setting in front of a lake",
    "positive_prompt":"Create a scenic natural environemnt, lush green natural background, nature reclaiming civilization, vivid greenery, ultra realistic, highly detailed, <lora:Greenery:0.4>",
    "negative_prompt":"tie, distorted human, erased person, missing face, blur, low quality, text, watermark, logo, oversaturated, artifacts, deformed structures, unrealistic textures, unnatural patterns, distorted perspective",
    "trigger_words": ["greenery"]
},

{
    "name": "Oil painting(oil brush stroke) ",
    "description":"Used for generating oil panting based images",
    "path_to_lora":"/workspace/S/sd1.5loras/bichu-v0612.safetensors",
    "strength":0.9,
    "guidance":7.0,
    "num_inference_steps":200,
    "image_description": "A man standing in an urbanic setting in front of a lake",
    "positive_prompt":"portrait of a person, oil painting style, artistic brush strokes, rich texture, soft lighting, detailed face, warm tones, painterly realism, canvas texture, dramatic contrast, Rembrandt-style lighting, masterpiece, highly detailed, bichu,<lora:Oil painting(oil brush stroke) - 油画笔触:0.5>",
    "negative_prompt":"photorealistic, modern digital render, blur, distorted face, extra limbs, cartoon, low contrast, flat color, deformed anatomy, text, watermark, logo, noisy background, pixelated texture, unnatural proportions",
    "trigger_words": ["bichu"]
},
{
    "name": "xiaorenshu",
    "description":"Used for generating japanese style images",
    "path_to_lora":"/workspace/S/sd1.5loras/xrs2.0.safetensors",
    "strength":0.9,
    "guidance":7.0,
    "num_inference_steps":200,
    "image_description": "A man standing in an urbanic setting in front of a lake",
    "positive_prompt":"Convert to japanese art style <lora:小人书·连环画 xiaorenshu:0.4>",
    "negative_prompt":"blurry, distorted face, low detail, overexposed, flat lighting, low resolution, multiple faces, deformed anatomy,oversaturated colors, harsh shadows, watermark, text, unnatural pose, unrealistic proportions, artifacts, noise",
    "trigger_words": [""]
},

{
    "name": "cosmic_nebula",
    "description":"Transforms images into vibrant cosmic scenes filled with swirling nebula clouds, glowing stardust, and ethereal interstellar lighting",
    "path_to_lora":"/workspace/S/sd1.5loras/cosmic.safetensors",
    "strength":0.9,
    "guidance":0.7,
    "num_inference_steps":70,
    "image_description": "A young chinese boy laughing joyfully in a checked shirt",
    "positive_prompt":"martius_nebula, a young chinese boy laughing joyfully in checked shirt, cosmic nebula clouds swirling in the sky behind him, glowing purple and blue stardust drifting around, soft astral light reflecting on his face, vibrant interstellar colors surrounding him, dreamy cosmic atmosphere, ethereal lighting, high detail, realistic anatomy, cosmic energy glow outlining his silhouette",
    "negative_prompt":"low detail, blurry, dull colors, washed out, flat lighting, cartoonish, distorted face, deformed hands, extra limbs, mutated features, bad anatomy, watermark, text, logo, grainy, noisy background, overexposed, underexposed",
    "trigger_words": ["martius_nebula"]
},

{
    "name": "steel_magic",
    "description":"adds molten-metal runes, forged-steel energy, and dramatic magical armor effects to any character.",
    "path_to_lora":"/workspace/S/sd1.5loras/steel.safetensors",
    "strength":0.65,
    "guidance":8.5,
    "num_inference_steps":70,
    "image_description": "a black man standing in gray shirt and black jeans",
    "positive_prompt":["DonMSt33lM4g1c, a Black man standing confidently, short well-trimmed hair, realistic face and anatomy",
        "arcane steel-magic energy swirling subtly around him",
        "glowing molten-metal runes orbiting his body like forged symbols",
        "faint sparks and metallic dust drifting through the air",
        "cool blue and fiery orange steel-forge lighting reflecting on his clothes",
        "epic steel-magic fantasy atmosphere, sharp details, high clarity"],
    "negative_prompt": ["fiery face, blurry, low detail, cartoonish",
        "distorted hands, extra limbs, bad anatomy",
        "washed out colors, text, watermark",
        "flat lighting, unrealistic proportions"],  
    "trigger_words": ["mDonMSt33lM4g1c"]
},


{
    "name": "retro_game_art",
    "description":"transforms images into vibrant 16-bit retro game art with bold outlines, pixel shading, and classic arcade aesthetics",
    "path_to_lora":"/workspace/S/sd1.5loras/retro.safetensors",
    "strength":0.8,
    "guidance":7,
    "num_inference_steps":70,
    "image_description": "a black man standing in gray shirt and black jeans",
    "positive_prompt":["r3tr0, a Black man standing confidently in a retro 16-bit game art style",
        "chunky pixel-art shading, bold outlines, vibrant retro color palette",
        "dynamic side-scroll hero stance, slight forward lean, powerful silhouette",
        "retro game background with neon-lit city buildings and pixel clouds",
        "crisp pixel details, sharp sprites, authentic SNES/Genesis game aesthetic"],
    "negative_prompt" : "blurry pixels, low detail, broken sprite shapes, pastel colors, text, watermark, realistic photo style",
    "trigger_words": ["r3tr0"]
},

{
    "name": "ghibli",
    "description":"LoRA for Stable Diffusion 1.5 that transfers the signature Studio Ghibli art style—soft colors, hand-painted textures, and whimsical lighting—onto any image or concept. Ideal for Ghibli-style landscapes, characters, and illustrations",
    "path_to_lora":"/workspace/S/sd1.5loras/ghibli.safetensors",
    "strength":0.9,
    "guidance":5.5,
    "num_inference_steps":60,
    "image_description": "elephant in a grassland with hills in background and cloudy sky",
    "positive_prompt": [
    "StdGBRedmAF , elephant , detailed hand-painted illustration, dreamy atmosphere, ",
    "soft watercolor shading, warm and nostalgic lighting, lush scenery, ",
    "cinematic composition, whimsical and heartwarming tone, ",
    "painterly brush strokes, vibrant natural colors"
    ],
    "negative_prompt": [
    "photorealistic, realistic, CGI, 3D render, plastic texture, harsh lighting, ",
    "text, watermark, signature, oversaturated, distorted"
    ],
    "trigger_words": ["Studio Ghibli", "StdGBRedmAF"]
},

{
    "name": "nyalia",
    "description":"A versatile LoRA for Stable Diffusion that applies the distinct “Nyalia” AI art aesthetic: ethereal lighting, soft gradients, and refined character focus. Ideal for portraits, fantasy scenes, and stylised concept art.",
    "path_to_lora":"/workspace/S/sd1.5loras/nyalia.safetensors",
    "strength":0.9,
    "guidance":7.0,
    "num_inference_steps":200,
    "image_description": "young smiling black boy with curly hair in a blue tshirt.",
    "positive_prompt":"nyalia A portrait of a young Black boy with radiant brown skin and deep expressive eyes, softly illuminated by golden light, wearing a simple white shirt, surrounded by dreamy pastel tones and glowing particles, in the Nyalia AI style — painterly textures, cinematic color grading, elegant composition, ultra-realistic yet artistic, masterpiece quality.",
    "negative_prompt":"blurry, bad anatomy, deformed, extra limbs, glitch, text, watermark, signature, cartoonish, oversaturated neon, flat lighting.",
    "trigger_words": ["nyalia"]
},
{
    "name": "gta",
    "description":"A versatile LoRA for Stable Diffusion that applies the distinct “Grand Theft Auto” art aesthetic: gritty urban environments, dramatic lighting, and stylized character designs. Ideal for action scenes, character portraits, and dynamic compositions.",
    "path_to_lora":"/workspace/S/sd1.5loras/gta.safetensors",
    "strength":0.9,
    "guidance":7.0,
    "num_inference_steps":200,
    "image_description": "young smiling black boy with curly hair in a blue tshirt.",
    "positive_prompt":"gr4nd_th3ft4_4 A portrait of a young Black boy  standing in an urban city street at sunset, wearing a leather jacket and jeans, confident expression, hands in pockets, dramatic cinematic lighting, detailed face, stylized shading and contour lines, painted in Grand Theft Auto IV loading screen art style, semi-realistic comic illustration, gritty yet vibrant tone, detailed background, masterpiece, 4K",
    "negative_prompt":"blurry, bad anatomy, deformed, extra limbs, glitch, text, watermark, signature, cartoonish, oversaturated neon, flat lighting.",
    "trigger_words": ["gr4nd_th3ft4_4"]
},
{
    "name": "zombie",
    "description":"A versatile LoRA for Stable Diffusion that applies the distinct “Zombie” art aesthetic: post-apocalyptic environments, dramatic lighting, and stylized character designs. Ideal for horror scenes, character portraits, and dynamic compositions.",
    "path_to_lora":"/workspace/S/sd1.5loras/zombie.safetensors",
    "strength":0.9,
    "guidance":6.0,
    "num_inference_steps":200,
    "image_description": "young smiling black boy with curly hair in a blue tshirt.",
    "positive_prompt":"post-apocalyptic world post-apocalyptic worldClose-up portrait of a young Black boy, face half turned toward camera, wide fearful eyes, dark skin with undead pallor beginning, subtle decay details: cracked dry skin, faint greenish tint under cheeks, faintly visible bone structure under skin, tattered school hoodie, ambient moonlight from above, cinematic horror lighting, dripping slight mist in background, in the style of Uzombies SD15 — ultra-detailed zombie concept art, high resolution, textured skin, dramatic shadows.",
    "negative_prompt":"blurry, bad anatomy, deformed, extra limbs, glitch, text, watermark, signature, cartoonish, oversaturated neon, flat lighting.",
    "trigger_words": ["post-apocalyptic world", "zombie","zombies"]
},

{
    "name": "toy_story_pixar",
    "description":"A LoRA for Stable Diffusion 1.5 that applies the vibrant, whimsical art style of Toy Story and Pixar animations—bright colors, expressive character designs, and playful environments. Ideal for generating 3D-rendered toy characters and lively scenes reminiscent of Pixar films.",
    "path_to_lora":"/workspace/S/sd1.5loras/disney.safetensors",
    "strength":0.75,
    "guidance":7.1,
    "num_inference_steps":100,
    "image_description": "young smiling black boy with curly hair in a blue tshirt",
    "positive_prompt": [
        "JessieWaifu, Toy Story style Pixar 3D render, young Black boy toy character, wearing a cowboy hat, ",
        "colorful shirt, animal print pants, boots, and belt, expressive face and bright smile, ",
        "soft cinematic lighting, detailed fabric and plastic textures, high-quality Pixar 3D render, ",
        "colorful bedroom toy environment, warm and whimsical atmosphere, ",
        "Disney Pixar movie aesthetic, vibrant color palette"
    ],
    "negative_prompt": [
        "realistic photo, creepy, low-poly, distorted face, text, watermark, ",
        "oversaturated, harsh lighting, messy background, extra limbs, broken anatomy"
    ],
    "trigger_words": ["JessieWaifu", "(hat, shirt, animal print pants, boots, belt)", "single braid"]
},

{
    "name": "cyberpunkAI",
    "description":"A LoRA for Stable Diffusion 1.5 that applies a vibrant cyberpunk art style—neon lights, futuristic cityscapes, and high-tech fashion. Ideal for generating dynamic sci-fi scenes and characters with a cyberpunk aesthetic.",
    "path_to_lora":"/workspace/S/sd1.5loras/cyberpunk.safetensors",
    "strength":0.75,
    "guidance":6.5,
    "num_inference_steps":100,
    "image_description": "young smiling black boy with curly hair in a blue tshirt",
    "positive_prompt": [
        "CyberpunkAI style, neon-lit futuristic cityscape, young Black boy in high-tech streetwear", 
        "glowing cybernetic visor, chrome jacket with neon trims, reflective wet pavement",
        "dark alley illuminated by holograms, cinematic lighting and deep shadows",
        "detailed 3D render, vivid neon palette, high-tech urban atmosphere, animated movie quality"
    ],
    "negative_prompt": [
        "blurry, low resolution, photorealistic, grainy, text, watermark, distorted anatomy" ,
        "pastel colors, daylight scene, low contrast"
    ],
    "trigger_words": ["CyberpunkAI"]
},

{
    "name": "sketch",
    "description":"A LoRA for Stable Diffusion 1.5 that transforms images into detailed pencil sketches with messy lines, hatching textures, and a traditional hand-drawn look. Ideal for generating black-and-white illustrations with a focus on line work and texture rather than full color.",
    "path_to_lora":"/workspace/S/sd1.5loras/sketch.safetensors",
    "strength":0.99,
    "guidance":6.8,
    "num_inference_steps":100,
    "image_description": "young smiling black boy with curly hair in a blue tshirt",
    "positive_prompt": [
        "black and white pencil_Sketch:1.2, messy lines, greyscale, traditional media, sketch, ",
        "no colors, traditional hand-drawn look,",
        "unfinished, hatching texture, ",
        "anime style illustration of a young Black boy, ",
        "large expressive eyes, slightly tousled hair, smiley face",
        "focus on line work and texture rather than full color",
        "dynamic pose, slight motion in his hair and clothes, ",
        "soft ambient light, subtle shadowing "
    ],
    "negative_prompt": [
        "photorealistic, full color, smooth lines, CG render, ",
        "watermark, text, signature, oversaturated, ",
        "blurred, low detail, distorted anatomy, extra limbs"
    ],
    "trigger_words": [
        "(Pencil_Sketch:1.2, messy lines, greyscale, traditional media, sketch)",
        "(unfinished, hatching (texture))"
    ]
},

{
    "name": "brush_painting",
    "description":"A LoRA for Stable Diffusion 1.5 that applies a bold brush painting art style—impressionist oil painting with rich textures, dramatic lighting, and expressive brush strokes. Ideal for generating painterly landscapes and artistic compositions with a hand-painted aesthetic.",
    "path_to_lora":"/workspace/S/sd1.5loras/paint.safetensors",
    "strength":0.8,
    "guidance":8,
    "num_inference_steps":200,
    "image_description": "elephant in a grassland with hills in background and cloudy sky",
    "positive_prompt": [
        "BigBrush painting artstyle, impressionist oil painting, bold brush strokes, ",
        "elephant in a wide grassland with rolling hills in the background, ",
        "dramatic cloudy sky with expressive strokes, soft sunlight breaking through clouds, ",
        "rich textured canvas feel, painterly composition, natural earth tones blended with vivid highlights, ",
        "hand-painted aesthetic, cinematic depth, artistic mood, ",
        "thick paint texture and visible brushwork throughout"
    ],
    "negative_prompt": [
        "photorealistic, smooth digital render, flat shading, CGI, ",
        "sharp edges, 3D model look, text, watermark, oversaturated colors, ",
        "cartoonish style, distorted anatomy, low detail"
    ],
    "trigger_words": [" "]
}
]
print("Loading embedding model...")
model = SentenceTransformer("all-mpnet-base-v2")  # or "all-MiniLM-L6-v2" for speed
model = model.to("cuda")
model.eval()  # Evaluation mode for inference
torch.backends.cudnn.benchmark = True  # Enable cuDNN optimizations

# PRE-COMPUTE LoRA embeddings (done once at startup)
print("Pre-computing LoRA embeddings...")
lora_texts = []
for l in lora_list:
    combined = f"{l['name']} {l['description']} {' '.join(l['trigger_words'][0])}"
    lora_texts.append(combined)

# Cache embeddings globally
LORA_EMBEDDINGS = model.encode(
    lora_texts, 
    convert_to_tensor=True,
    show_progress_bar=False,
    batch_size=32  # Process in batches
)
print(f"Cached {len(lora_list)} LoRA embeddings on GPU")

def semantic_find_best_lora(prompt: str):
    """
    Optimized version - only encodes the prompt, uses cached LoRA embeddings
    """
    # Only encode the user prompt (fast!)
    with torch.no_grad():  # Disable gradient computation for inference
        emb_prompt = model.encode(
            prompt, 
            convert_to_tensor=True,
            show_progress_bar=False
        )
    
    # Compute similarity with pre-cached embeddings
    scores = util.cos_sim(emb_prompt, LORA_EMBEDDINGS)[0]
    idx = int(torch.argmax(scores))
    
    return lora_list[idx], float(scores[idx])


# Optional: Batch processing version
def semantic_find_best_loras_batch(prompts: list):
    """
    Process multiple prompts at once (even faster per-prompt)
    """
    with torch.no_grad():
        emb_prompts = model.encode(
            prompts, 
            convert_to_tensor=True,
            show_progress_bar=False,
            batch_size=32
        )
    
    scores = util.cos_sim(emb_prompts, LORA_EMBEDDINGS)
    
    results = []
    for i, prompt in enumerate(prompts):
        idx = int(torch.argmax(scores[i]))
        results.append((lora_list[idx], float(scores[i][idx])))
    
    return results


def unify_prompt(x):
    if isinstance(x, list):
        return " ".join([str(t) for t in x])
    return str(x)


def run_img2img(prompt, image_path):
    # A) Find best LoRA (now much faster!)
    best_lora, score = semantic_find_best_lora(prompt)
    print("\nSelected LoRA:", best_lora["name"], "| Score:", score)

    # Build FINAL prompt
    trigger_words = best_lora["trigger_words"][0]# " ".join(best_lora.get("trigger_words", []))
    final_prompt = f"{trigger_words} {prompt}".strip()
    final_negative = unify_prompt(best_lora.get("negative_prompt", ""))

    print("\nFinal Positive Prompt:\n", final_prompt)
    print("\nFinal Negative Prompt:\n", final_negative)
    print("\nSteps:", best_lora["num_inference_steps"])
    print("Guidance:", best_lora["guidance"])
    print("LoRA Strength:", best_lora["strength"])
    print("LoRA File:", best_lora["path_to_lora"])

    # Load SD 1.5 Img2Img Pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to("cuda")

    # Load LoRA
    pipe.load_lora_weights(best_lora["path_to_lora"])

    # Load input image
    image = Image.open(image_path).convert("RGB")#.resize((512,512))

    # SD Parameters
    denoise_strength = 0.6

    result = pipe(
        prompt=final_prompt,
        negative_prompt=final_negative,
        image=image,
        strength=denoise_strength,
        num_inference_steps=best_lora["num_inference_steps"],
        guidance_scale=best_lora["guidance"],
        cross_attention_kwargs={"scale": best_lora["strength"]},
    )

    out = result.images[0]
    out.save("output.png")
    print("\nSaved output as output.png")

    return out


if __name__ == "__main__":
    prompt = "martius_nebula, a majestic direwolf standing proudly in a dense forest, cosmic nebula clouds swirling through the trees, vibrant purple and blue stardust drifting around its fur, ethereal interstellar light filtering between the branches, glowing cosmic energy outlining the direwolf’s silhouette, soft astral illumination on the forest floor, highly detailed fur texture, powerful stance, mystical atmosphere, dreamy cosmic realism, high detail, sharp features, dramatic celestial lighting"
    run_img2img(prompt, "direwolf.png")
    
    