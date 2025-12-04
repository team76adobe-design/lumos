import os
import torch
import numpy as np
from PIL import Image
from einops import rearrange
from safetensors.torch import load_file
import ledits_patch
# from pytorch_lightning import seed_everything
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel #, LCMScheduler
from lightning_drag_models.utils import import_model_class_from_model_name_or_path
from lightning_drag_models.ip_adapter import ImageProjModel
from lightning_drag_models.appearance_encoder import AppearanceEncoderModel
from lightning_drag_models.point_embedding import PointEmbeddingModel
from lightning_drag_models.pipeline import LightningDragPipeline


def load_lightningdrag(
    base_sd_path,
    vae_path,
    ip_adapter_path,
    lightning_drag_path,
    lcm_lora_path=None,
    device="cuda",
):
    dtype = torch.float16

    # --- Load components ---
    tokenizer = AutoTokenizer.from_pretrained(base_sd_path, subfolder="tokenizer", use_fast=False)
    text_encoder_cls = import_model_class_from_model_name_or_path(base_sd_path, revision=None)
    text_encoder = text_encoder_cls.from_pretrained(base_sd_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base_sd_path if vae_path == "default" else vae_path)
    unet = UNet2DConditionModel.from_pretrained(base_sd_path, subfolder="unet")
    config = unet.config
    config["in_channels"] = 4
    appearance_encoder = AppearanceEncoderModel.from_config(config)
    noise_scheduler = DDIMScheduler.from_pretrained(base_sd_path, subfolder="scheduler")

    # --- IP Adapter ---
    image_encoder_path = os.path.join(ip_adapter_path, "image_encoder")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
    clip_image_processor = CLIPImageProcessor()

    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )

    ip_ckpt_path = os.path.join(ip_adapter_path, "ip-adapter_sd15.bin")
    ip_state_dict = torch.load(ip_ckpt_path, map_location="cpu", weights_only=True)
    image_proj_model.load_state_dict(ip_state_dict["image_proj"])

    point_embedding = PointEmbeddingModel(embed_dim=16)

    # Move models to device
    for m in [unet, vae, text_encoder, appearance_encoder, image_encoder, image_proj_model, point_embedding]:
        m.to(device).to(dtype)

    # --- Pipeline ---
    pipe = LightningDragPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        appearance_encoder=appearance_encoder,
        scheduler=noise_scheduler,
        feature_extractor=clip_image_processor,
        image_encoder=image_encoder,
        point_embedding=point_embedding,
        safety_checker=None,
        fusion_blocks="full",
        initialize_attn_processor=True,
        use_norm_attn_processor=True,
        initialize_ip_attn_processor=True,
        image_proj_model=image_proj_model,
    )

    # --- Load weights ---
    attn_processors = torch.nn.ModuleList(pipe.unet.attn_processors.values())
    state_dict = torch.load(os.path.join(lightning_drag_path, "lightning-drag-sd15-attn.bin"))
    state_dict.update(ip_state_dict["ip_adapter"])
    attn_processors.load_state_dict(state_dict)

    appearance_state_dict = load_file(os.path.join(lightning_drag_path, "appearance_encoder/diffusion_pytorch_model.safetensors"))
    pipe.appearance_encoder.load_state_dict(appearance_state_dict)

    point_embedding_state_dict = torch.load(os.path.join(lightning_drag_path, "point_embedding/point_embedding.pt"))
    pipe.point_embedding.load_state_dict(point_embedding_state_dict)

    # --- Optional LCM ---
    # if lcm_lora_path is not None:
    #     pipe.load_lora_weights(lcm_lora_path)
    #     pipe.fuse_lora()
    #     pipe.scheduler = LCMScheduler.from_pretrained(base_sd_path, subfolder="scheduler")

    pipe = pipe.to(device).to(dtype)
    return pipe


def run_inference(
    pipe,
    image_path,
    mask_path,
    handle_points,
    target_points,
    seed=42,
    num_inference_steps=25,
    guidance_scale_points=4.0,
    guidance_scale_decay="inv_square",
    output_dir="./outputs",
):
    os.makedirs(output_dir, exist_ok=True)

    # seed_everything(seed)
    device = "cuda"

    # --- Load image ---
    source_image = Image.open(image_path).convert("RGB")
    width, height = source_image.size

    # 🔧 Ensure dimensions are multiples of 8
    new_width = (width // 64) * 64
    new_height = (height // 64) * 64
    if (new_width, new_height) != (width, height):
        print(f"⚙️ Resizing image from ({width}, {height}) → ({new_width}, {new_height}) for UNet compatibility")
        source_image = source_image.resize((new_width, new_height), Image.BICUBIC)
        width, height = new_width, new_height

    # --- Convert image to tensor ---
    np_image = np.array(source_image)
    tensor_image = torch.from_numpy(np_image).float()
    tensor_image = rearrange(tensor_image, "h w c -> 1 c h w")
    tensor_image = 2.0 * tensor_image / 255.0 - 1.0

    # --- Load and resize mask ---
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((width, height), Image.NEAREST)
    mask = torch.from_numpy(np.array(mask)).float() / 255.0

    # --- Points ---
    handle_points = torch.tensor(handle_points).long()  # [[y, x], ...]
    target_points = torch.tensor(target_points).long()

    # --- Inference ---
    with torch.inference_mode():
        result = pipe(
            ref_image=tensor_image,
            mask_image=mask,
            prompt="",
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale_points=guidance_scale_points,
            guidance_scale_decay=guidance_scale_decay,
            num_guidance_steps=None,
            num_images_per_prompt=4,
            output_type="pt",
            handle_points=handle_points,
            target_points=target_points,
            skip_cfg_appearance_encoder=False,
        ).images

    # --- Save outputs ---
    result = (result * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    out_paths = []
    for i, img in enumerate(result):
        out_path = os.path.join(output_dir, f"result_{i+1}.png")
        Image.fromarray(img).save(out_path)
        out_paths.append(out_path)

    print(f"✅ Saved {len(out_paths)} outputs to {output_dir}")
    return out_paths



if __name__ == "__main__":
    base_sd_path = "/workspace/checkpoints/dreamshaper-8-inpainting"
    vae_path = "/workspace/checkpoints/sd-vae-ft-ema"
    ip_adapter_path = "/workspace/checkpoints/IP-Adapter/models"
    lightning_drag_path = "/workspace/checkpoints/lightning-drag-sd15"
    lcm_lora_path = "/workspace/checkpoints/lcm-lora-sdv1-5/pytorch_lora_weights.safetensors"


    pipe = load_lightningdrag(base_sd_path, vae_path, ip_adapter_path, lightning_drag_path, lcm_lora_path)

    image_path = "tmp_out/run1/input_image.png"
    mask_path = "tmp_out/run1/refined_mask.png"

    # Example: one handle → one target
    handle_points = [[235, 250]]  # (y, x)
    target_points = [[235, 350]]  # (y, x)

    run_inference(
        pipe,
        image_path=image_path,
        mask_path=mask_path,
        handle_points=handle_points,
        target_points=target_points,
        seed=42,
        num_inference_steps=8,  # shorter since LCM LoRA is used
        guidance_scale_points=3.0,
    )
