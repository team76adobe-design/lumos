# ledits_patch.py — Compatibility patch for LEDITS++
import os, importlib.util

import huggingface_hub

# ---- Patch missing functions for diffusers==0.20.2 ----
if not hasattr(huggingface_hub, "cached_download"):
    print("🧩 Patching: adding cached_download → hf_hub_download")
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

if not hasattr(huggingface_hub, "model_info"):
    from huggingface_hub import HfApi
    def model_info(repo_id, revision=None, token=None):
        return HfApi().model_info(repo_id, revision=revision, token=token)
    huggingface_hub.model_info = model_info

if not hasattr(huggingface_hub, "HfFolder"):
    class HfFolder:
        @staticmethod
        def get_token():
            return None
    huggingface_hub.HfFolder = HfFolder

# ---- Disable fast download if hf_transfer missing ----
if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
    spec = importlib.util.find_spec("hf_transfer")
    if spec is None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        print("⚙ Disabled HF_HUB_ENABLE_HF_TRANSFER (hf_transfer not installed)")

print(f"✅ LEDITS++ environment patched successfully (huggingface_hub {huggingface_hub.__version__})")