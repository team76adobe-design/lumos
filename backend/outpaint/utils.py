
import torch
from iopaint.model_manager import ModelManager
from iopaint.download import cli_download_model
import base64
cli_download_model('runwayml/stable-diffusion-v1-5')

device=torch.device('cuda')

from iopaint.helper import (
    load_img,
    decode_base64_to_image,
    pil_to_bytes,
    numpy_to_bytes,
    concat_alpha_channel,
    gen_frontend_mask,
    adjust_mask,
)

from iopaint.const import (
    INSTRUCT_PIX2PIX_NAME,
    KANDINSKY22_NAME,
    POWERPAINT_NAME,
    ANYTEXT_NAME,
    SDXL_CONTROLNET_CHOICES,
    SD2_CONTROLNET_CHOICES,
    SD_CONTROLNET_CHOICES,
    SD_BRUSHNET_CHOICES,
    SDXL_BRUSHNET_CHOICES
)

from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional
import cv2
import numpy as np
from iopaint.model.utils import torch_gc
import base64
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
from matplotlib import pyplot as plt
class CV2Flag(str, Enum):
    INPAINT_NS = "INPAINT_NS"
    INPAINT_TELEA = "INPAINT_TELEA"


class LDMSampler(str, Enum):
    ddim = "ddim"
    plms = "plms"


class SDSampler(str, Enum):
    dpm_plus_plus_2m = "DPM++ 2M"
    dpm_plus_plus_2m_karras = "DPM++ 2M Karras"
    dpm_plus_plus_2m_sde = "DPM++ 2M SDE"
    dpm_plus_plus_2m_sde_karras = "DPM++ 2M SDE Karras"
    dpm_plus_plus_sde = "DPM++ SDE"
    dpm_plus_plus_sde_karras = "DPM++ SDE Karras"
    dpm2 = "DPM2"
    dpm2_karras = "DPM2 Karras"
    dpm2_a = "DPM2 a"
    dpm2_a_karras = "DPM2 a Karras"
    euler = "Euler"
    euler_a = "Euler a"
    heun = "Heun"
    lms = "LMS"
    lms_karras = "LMS Karras"

    ddim = "DDIM"
    pndm = "PNDM"
    uni_pc = "UniPC"
    lcm = "LCM"



class HDStrategy(str, Enum):
    # Use original image size
    ORIGINAL = "Original"
    # Resize the longer side of the image to a specific size(hd_strategy_resize_limit),
    # then do inpainting on the resized image. Finally, resize the inpainting result to the original size.
    # The area outside the mask will not lose quality.
    RESIZE = "Resize"
    # Crop masking area(with a margin controlled by hd_strategy_crop_margin) from the original image to do inpainting
    CROP = "Crop"


class Choices(str, Enum):
    @classmethod
    def values(cls):
        return [member.value for member in cls]


class PowerPaintTask(Choices):
    text_guided = "text-guided"
    context_aware = "context-aware"
    shape_guided = "shape-guided"
    object_remove = "object-remove"
    outpainting = "outpainting"

class InpaintRequest(BaseModel):
    image: Optional[str] = Field(None, description="base64 encoded image")
    mask: Optional[str] = Field(None, description="base64 encoded mask")

    ldm_steps: int = Field(20, description="Steps for ldm model.")
    ldm_sampler: str = Field(LDMSampler.plms, description="Sampler for ldm model.")
    zits_wireframe: bool = Field(True, description="Enable wireframe for zits model.")

    hd_strategy: str = Field(
        HDStrategy.CROP,
        description="Different way to preprocess image, only used by erase models(e.g. lama/mat)",
    )
    hd_strategy_crop_trigger_size: int = Field(
        800,
        description="Crop trigger size for hd_strategy=CROP, if the longer side of the image is larger than this value, use crop strategy",
    )
    hd_strategy_crop_margin: int = Field(
        128, description="Crop margin for hd_strategy=CROP"
    )
    hd_strategy_resize_limit: int = Field(
        1280, description="Resize limit for hd_strategy=RESIZE"
    )

    prompt: str = Field("", description="Prompt for diffusion models.")
    negative_prompt: str = Field(
        "", description="Negative prompt for diffusion models."
    )
    use_croper: bool = Field(
        False, description="Crop image before doing diffusion inpainting"
    )
    croper_x: int = Field(0, description="Crop x for croper")
    croper_y: int = Field(0, description="Crop y for croper")
    croper_height: int = Field(512, description="Crop height for croper")
    croper_width: int = Field(512, description="Crop width for croper")

    use_extender: bool = Field(
        False, description="Extend image before doing sd outpainting"
    )
    extender_x: int = Field(0, description="Extend x for extender")
    extender_y: int = Field(0, description="Extend y for extender")
    extender_height: int = Field(640, description="Extend height for extender")
    extender_width: int = Field(640, description="Extend width for extender")

    sd_scale: float = Field(
        1.0,
        description="Resize the image before doing sd inpainting, the area outside the mask will not lose quality.",
        gt=0.0,
        le=1.0,
    )
    sd_mask_blur: int = Field(
        12,
        description="Blur the edge of mask area. The higher the number the smoother blend with the original image",
    )
    sd_strength: float = Field(
        1.0,
        description="Strength is a measure of how much noise is added to the base image, which influences how similar the output is to the base image. Higher value means more noise and more different from the base image",
        le=1.0,
    )
    sd_steps: int = Field(
        50,
        description="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.",
    )
    sd_guidance_scale: float = Field(
        7.5,
        description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality.",
    )
    sd_sampler: str = Field(
        SDSampler.uni_pc, description="Sampler for diffusion model."
    )
    sd_seed: int = Field(
        42,
        description="Seed for diffusion model. -1 mean random seed",
        validate_default=True,
    )
    sd_match_histograms: bool = Field(
        False,
        description="Match histograms between inpainting area and original image.",
    )

    sd_outpainting_softness: float = Field(20.0)
    sd_outpainting_space: float = Field(20.0)

    sd_lcm_lora: bool = Field(
        False,
        description="Enable lcm-lora mode. https://huggingface.co/docs/diffusers/main/en/using-diffusers/inference_with_lcm#texttoimage",
    )

    sd_keep_unmasked_area: bool = Field(
        True, description="Keep unmasked area unchanged"
    )

    cv2_flag: CV2Flag = Field(
        CV2Flag.INPAINT_NS,
        description="Flag for opencv inpainting: https://docs.opencv.org/4.6.0/d7/d8b/group__photo__inpaint.html#gga8002a65f5a3328fbf15df81b842d3c3ca05e763003a805e6c11c673a9f4ba7d07",
    )
    cv2_radius: int = Field(
        4,
        description="Radius of a circular neighborhood of each point inpainted that is considered by the algorithm",
    )

    # Paint by Example
    paint_by_example_example_image: Optional[str] = Field(
        None, description="Base64 encoded example image for paint by example model"
    )

    # InstructPix2Pix
    p2p_image_guidance_scale: float = Field(1.5, description="Image guidance scale")

    # ControlNet
    enable_controlnet: bool = Field(False, description="Enable controlnet")
    controlnet_conditioning_scale: float = Field(
        0.4, description="Conditioning scale", ge=0.0, le=1.0
    )
    controlnet_method: str = Field(
        "lllyasviel/control_v11p_sd15_canny", description="Controlnet method"
    )

    # BrushNet
    enable_brushnet: bool = Field(False, description="Enable brushnet")
    brushnet_method: str = Field(SD_BRUSHNET_CHOICES[0], description="Brushnet method")
    brushnet_conditioning_scale: float = Field(
        1.0, description="brushnet conditioning scale", ge=0.0, le=1.0
    )

    # PowerPaint
    enable_powerpaint_v2: bool = Field(False, description="Enable PowerPaint v2")
    powerpaint_task: PowerPaintTask = Field(
        PowerPaintTask.object_remove, description="PowerPaint task"
    )
    fitting_degree: float = Field(
        1.0,
        description="Control the fitting degree of the generated objects to the mask shape.",
        gt=0.0,
        le=1.0,
    )



# model=ModelManager(name='runwayml/stable-diffusion-v1-5',device=device,enable_powerpoint_v2=True,disable_nsfw=True,sd_cpu_textencoder=False,
#             local_files_only=False,
#             cpu_offload=False,
#             )


def get_size_from_base64(b64_str):
    # Decode base64 → bytes
    img_bytes = base64.b64decode(b64_str)

    # Bytes → numpy buffer
    nparr = np.frombuffer(img_bytes, np.uint8)

    # Decode into image
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    # Height, Width
    h, w = img.shape[:2]
    return h, w
def decode_base64_to_np(b64, gray=False):
    """Same functionality as your decode_base64_to_image."""
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imdecode(arr, flag)
    return img

def np_to_base64(img, format=".png"):
    """Encode numpy image to base64 string."""
    _, buffer = cv2.imencode(format, img)
    return base64.b64encode(buffer).decode("utf-8")

def outpaint(image_base64, scale, positive_prompt, negative_prompt):
    H, W = get_size_from_base64(image_base64)

    # Compute extended size
    extender_width  = int(W * scale)
    extender_height = int(H * scale)
    extender_x = (extender_width - W) // 2
    extender_y = (extender_height - H) // 2
    # image_base64 = pad_base64_image(image_base64, extender_width, extender_height,extender_x,extender_y)
    # Generate outpaint mask (base64)
    dummy_mask_np=np.full((H,W),0, dtype=np.uint8)
    mask_base64= np_to_base64(dummy_mask_np)
    # Decode image
    image, alpha, infos, ext = decode_base64_to_image(image_base64)

    # Decode mask
    mask_img, _, _, _ = decode_base64_to_image(mask_base64, gray=True)
    # plt.imshow(mask_img,cmap='gray')
    # 🔥 Force grayscale (important fix)
    if len(mask_img.shape) == 3:
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

    # Binary mask
    mask_bin = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)[1]

    # Build request
    request = InpaintRequest(
        image=image_base64,
        mask=mask_base64,
        prompt=positive_prompt,
        negative_prompt=negative_prompt,

        use_croper=False,
        use_extender=True,
        extender_x=-extender_x,
        extender_y=-extender_y,
        extender_width=extender_width,
        extender_height=extender_height,

        enable_brushnet=False,
        enable_powerpaint_v2=True,
        powerpaint_task=PowerPaintTask.outpainting
    )
    kernel = np.ones((15, 15), np.uint8)   # increase the size for more expansion

# apply dilation
    # mask_bin = cv2.dilate(mask_bin, kernel, iterations=1)
    # Run model
    rgb_np_img = model(image, mask_bin, request)
    torch_gc()

    # Convert to RGB
    rgb_np_img = cv2.cvtColor(rgb_np_img.astype(np.uint8), cv2.COLOR_BGR2RGB)

    # Output (no alpha restore)
    res_img_bytes = pil_to_bytes(
        Image.fromarray(rgb_np_img),
        ext=ext,
        quality=100,
        infos=infos
    )

    return res_img_bytes


# with open("man.jpg", "rb") as f:
#     image_b64 = base64.b64encode(f.read()).decode("utf-8")

# # with open("/content/direwolf_mask_resize_2.png", "rb") as f:
# #     mask_b64 = base64.b64encode(f.read()).decode("utf-8")

# result_bytes = outpaint(
#     image_base64=image_b64,
#     scale=1.2,
#     positive_prompt="",
#     negative_prompt="out of frame, lowres, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, disfigured, gross proportions, malformed limbs, watermark, signature"
# )

# with open("output.png", "wb") as f:
#     f.write(result_bytes)

