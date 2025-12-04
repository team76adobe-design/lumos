import os
import json
import numpy as np
import torch
from PIL import Image
import argparse

import ledits_patch
from diffusers import AutoPipelineForInpainting

from inpaint4drag_utils.drag import bi_warp
from inpaint4drag_utils.refine_mask import SamMaskRefiner

__all__ = ['get_inpaint_pipeline', 'drag_inpaint']

_PIPE = None
_SAM_REFINER = None


def get_inpaint_pipeline(device: str = 'cuda'):
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    model_id = 'runwayml/stable-diffusion-inpainting'

    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'

    dtype = torch.float16 if (device.startswith('cuda') and torch.cuda.is_available()) else torch.float32
    pipe = AutoPipelineForInpainting.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    _PIPE = pipe
    return _PIPE


def _refine_mask_if_enabled(image, mask, use_sam, kernel_size):
    if not use_sam or mask is None or mask.sum() == 0:
        return mask
    global _SAM_REFINER
    try:
        if _SAM_REFINER is None:
            _SAM_REFINER = SamMaskRefiner()
        return _SAM_REFINER.refine_mask(image, mask, kernel_size)
    except Exception:
        return mask


def _run_inpaint(
    pipe,
    image,
    inpaint_mask,
    num_steps=8,
    guidance_scale=1.0,
    strength=1.0,
):
    if image is None or inpaint_mask is None:
        return image

    image_pil = Image.fromarray(image)
    inpaint_mask_pil = Image.fromarray(inpaint_mask)

    width, height = inpaint_mask_pil.size
    if width % 8 != 0 or height % 8 != 0:
        width, height = round(width / 8) * 8, round(height / 8) * 8
        image_pil = image_pil.resize((width, height))
        image = np.array(image_pil)
        inpaint_mask_pil = inpaint_mask_pil.resize((width, height), Image.NEAREST)
        inpaint_mask = np.array(inpaint_mask_pil)

    out = pipe(
        prompt='',
        image=image_pil,
        mask_image=inpaint_mask_pil,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        strength=strength,
    ).images[0]

    result = np.array(out)

    inpaint_mask_01 = (inpaint_mask[..., np.newaxis] / 255).astype(np.uint8)
    return (result.astype(np.uint8)) * inpaint_mask_01 + image * (1 - inpaint_mask_01)


def drag_inpaint(
    image,
    mask,
    points,
    sam_kernel=21,
    inpaint_kernel=5,
    use_sam=False,
    device='cuda',
    output_dir=None,
    num_steps=8,
    guidance_scale=1.0,
    strength=1.0,
):
    if image is None or mask is None or len(points) < 2:
        return image

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    orig_image = image
    orig_mask = mask

    mask = _refine_mask_if_enabled(image, mask, use_sam, sam_kernel)

    handle_pts, target_pts, inpaint_mask01 = bi_warp(mask, points, inpaint_kernel)

    warped = image.copy()
    warped[target_pts[:, 1], target_pts[:, 0]] = warped[handle_pts[:, 1], handle_pts[:, 0]]

    inpaint_mask255 = (inpaint_mask01 * 255).astype(np.uint8)

    pipe = get_inpaint_pipeline(device=device)
    result = _run_inpaint(
        pipe, warped, inpaint_mask255, num_steps=num_steps, guidance_scale=guidance_scale, strength=strength
    )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        Image.fromarray(orig_image).save(os.path.join(output_dir, 'input_image.png'))
        Image.fromarray(orig_mask).save(os.path.join(output_dir, 'input_mask.png'))
        Image.fromarray(mask).save(os.path.join(output_dir, 'refined_mask.png'))
        Image.fromarray(warped).save(os.path.join(output_dir, 'warped.png'))
        Image.fromarray(inpaint_mask255).save(os.path.join(output_dir, 'inpaint_mask.png'))
        Image.fromarray(result).save(os.path.join(output_dir, 'result.png'))

        meta = {
            'device': device,
            'points': [list(map(int, p)) for p in points],
        }
        with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f)

    return result


def main():
    parser = argparse.ArgumentParser(description="Drag-based inpainting (1 drag pair)")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--mask", required=True, help="Path to the binary mask")
    parser.add_argument("--x1", type=int, required=True)
    parser.add_argument("--y1", type=int, required=True)
    parser.add_argument("--x2", type=int, required=True)
    parser.add_argument("--y2", type=int, required=True)
    parser.add_argument("--output_dir", default="output_drag", help="Folder to save outputs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    image = np.array(Image.open(args.image).convert("RGB"))
    mask = np.array(Image.open(args.mask).convert("L"))

    points = [(args.x1, args.y1), (args.x2, args.y2)]

    print("Running drag inpaint...")
    result = drag_inpaint(
        image=image,
        mask=mask,
        points=points,
        output_dir=args.output_dir,
        device=args.device,
    )

    print(f"Done! Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()


