import torch
from transformers import AutoModelForCausalLM

MODEL_NAME = "vikhyatk/moondream2"
REVISION = "2025-06-21"

_model = None

def load_model():
    """
    Loads the model on the GPU if not already loaded.
    """
    global _model
    if _model is None:
        print("Loading model onto GPU...")
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            revision=REVISION,
            trust_remote_code=True,
        ).to("cuda")
        print("Model loaded!")

def get_model():
    """
    Returns the model if it is loaded, else None.
    """
    return _model

def unload_model():
    """
    Unloads the model from GPU and clears cache.
    """
    global _model
    if _model is not None:
        print("Unloading model from GPU...")
        del _model
        torch.cuda.empty_cache()
        _model = None
        print("Model unloaded.")
