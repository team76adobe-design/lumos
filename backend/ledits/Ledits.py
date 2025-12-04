import os, torch, numpy as np
import ledits_patch  
from leditspp import StableDiffusionPipeline_LEDITS
from leditspp.scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject
from ledits_helper import load_image, image_grid
from PIL import Image


# --- Load LEDITS++ model (uses HF cache, no re-download) ---
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

# --- Load and process image ---
im = Image.open("ledit.jpg").convert("RGB").resize((512, 512))
gen = torch.manual_seed(42)
with torch.no_grad():
    _ = pipe.invert(im, num_inversion_steps=50, generator=gen, verbose=True, skip=0.15)
    out = pipe(
        editing_prompt=["sunglasses"],
        edit_threshold=[0.7, 0.9],
        edit_guidance_scale=[3, 4],
        reverse_editing_direction=[False, False],
        use_intersect_mask=True,
    )

out.images[0].save("ledits_result.png")