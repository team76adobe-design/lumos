import os, importlib.util

import huggingface_hub


if not hasattr(huggingface_hub, "cached_download"):
    
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


if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1":
    spec = importlib.util.find_spec("hf_transfer")
    if spec is None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        print("⚙ Disabled HF_HUB_ENABLE_HF_TRANSFER (hf_transfer not installed)")

