import os
from pathlib import Path
from huggingface_hub import snapshot_download
from tqdm import tqdm

# -------- CONFIGURATION -------- #

ROOT = Path("./checkpoints")

# Each entry is a Hugging Face repo that will be fully downloaded
MODELS = {
    "dreamshaper-8-inpainting": "Lykon/dreamshaper-8-inpainting",
    "lcm-lora-sdv1-5": "latent-consistency/lcm-lora-sdv1-5",
    "sd-vae-ft-ema": "stabilityai/sd-vae-ft-mse",  # despite the name mismatch, this is correct
    "IP-Adapter": "h94/IP-Adapter",
    "lightning-drag-sd15": "LightningDrag/lightning-drag-sd15",
}


# -------- MAIN LOGIC -------- #

def setup_checkpoints():
    ROOT.mkdir(parents=True, exist_ok=True)
    for folder_name, repo_id in MODELS.items():
        dest_dir = ROOT / folder_name
        if dest_dir.exists() and any(dest_dir.iterdir()):
            print(f"✅ Already exists: {dest_dir}")
            continue

        print(f"\n⬇️ Downloading full model '{repo_id}' → {dest_dir}")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=dest_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                tqdm_class=tqdm
            )
            print(f"✅ Finished downloading: {folder_name}")
        except Exception as e:
            print(f"❌ Failed to download {repo_id}: {e}")

    print("\n🎉 All checkpoints downloaded and organized!")


# -------- RUN -------- #

if __name__ == "__main__":
    setup_checkpoints()
