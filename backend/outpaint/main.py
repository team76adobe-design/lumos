
import base64
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .model import load_model, unload_model, run_outpaint


app = FastAPI(title="IOPaint Outpaint API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoadResponse(BaseModel):
    status: str


class OutpaintRequest(BaseModel):
    image_base64: str
    scale: float = 1.2
    positive_prompt: str = ""
    negative_prompt: str = ""


@app.post("/load_model", response_model=LoadResponse)
def load():
    status = load_model()
    return LoadResponse(status=status)



@app.post("/infer")
def infer(req: OutpaintRequest):

    if req.scale <= 1.0:
        return {"error": "Scale must be > 1.0 for outpainting"}

    try:
        
        out_bytes = run_outpaint(
            image_b64=req.image_base64,
            scale=req.scale,
            prompt=req.positive_prompt,
            negative_prompt=req.negative_prompt
        )

        
        out_b64 = base64.b64encode(out_bytes).decode("utf-8")

        return {
            "image_base64": out_b64
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/unload_model")
def unload():
    status = unload_model()
    return {"status": status}
