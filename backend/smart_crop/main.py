import os
import pickle
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
import base64
# SmartCrop imports
from .smartcrop_utils import network
from .smartcrop_utils.actions import command2action, generate_bbox, crop_input

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
var_dict = None

# -------------------------
# Model Wrapper
# -------------------------

class AutoCroppingModel:
    def __init__(self, var_dict):
        self.var_dict = var_dict

    @tf.function
    def get_global_features(self, image):
        return network.vfn_rl(image, self.var_dict)

    @tf.function
    def get_action(self, image, global_feature, h, c):
        return network.vfn_rl(
            image,
            self.var_dict,
            global_feature=global_feature,
            h=h,
            c=c
        )


# ---------------------------------------
# Core Auto-cropping Logic
# ---------------------------------------

def auto_cropping(model: AutoCroppingModel, origin_image):
    batch_size = len(origin_image)

    terminals = np.zeros(batch_size)
    ratios = np.repeat([[0, 0, 20, 20]], batch_size, axis=0)

    img = crop_input(origin_image, generate_bbox(origin_image, ratios))
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

    global_feature = model.get_global_features(img_tensor)

    h_np = np.zeros([batch_size, 1024], dtype=np.float32)
    c_np = np.zeros([batch_size, 1024], dtype=np.float32)

    max_iterations = 50
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        h_tensor = tf.convert_to_tensor(h_np, dtype=tf.float32)
        c_tensor = tf.convert_to_tensor(c_np, dtype=tf.float32)

        action_tensor, h_tensor, c_tensor = model.get_action(
            img_tensor, global_feature, h_tensor, c_tensor
        )

        action_np = action_tensor.numpy()
        h_np = h_tensor.numpy()
        c_np = c_tensor.numpy()

        ratios, terminals = command2action(action_np, ratios, terminals)
        bbox = generate_bbox(origin_image, ratios)

        if np.sum(terminals) == batch_size:
            return bbox

        img = crop_input(origin_image, bbox)

    return bbox


# -------------------------
# API Endpoints
# -------------------------

@app.post("/load")
def load_model():
    global model, var_dict

    if model is not None:
        return {"status": "Model already loaded"}

    try:
        with open("app/smart_crop/smartcrop_utils/vfn_rl.pkl", "rb") as f:
            var_dict = pickle.load(f)

        model = AutoCroppingModel(var_dict)

        return {"status": "Model loaded successfully"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/run")
async def run_auto_crop(image: UploadFile = File(...)):
    global model

    if model is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Model not loaded. Call /load first."}
        )

    try:
        image_bytes = await image.read()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # Convert to numpy float32 and normalize
        im_np = np.array(pil_image).astype(np.float32) / 255.0
        im_input = [im_np - 0.5]     # model expects mean-centered input

        # Run auto-cropping
        xmin, ymin, xmax, ymax = auto_cropping(model, im_input)[0]

        # Crop original resolution
        original_np = np.array(pil_image)
        cropped = original_np[ymin:ymax, xmin:xmax]

        # Encode as base64
        output_buffer = BytesIO()
        Image.fromarray(cropped).save(output_buffer, format="JPEG")
        cropped_bytes = output_buffer.getvalue()

        return {
            "status": "Success",
            "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)],
            "cropped_image_base64": base64.b64encode(cropped_bytes).decode("utf-8")
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/unload")
def unload_model():
    global model, var_dict
    model = None
    var_dict = None
    tf.keras.backend.clear_session()

    return {"status": "Model unloaded"}


@app.get("/")
def root():
    return {"message": "SmartCrop RL API. Use /load, /run, /unload."}

# import subprocess
# import requests
# import base64
# from fastapi import FastAPI, UploadFile, File

# app = FastAPI()

# worker_process = None

# @app.post("/load")
# def load_model():
#     global worker_process

#     # start worker
#     worker_process = subprocess.Popen(["python", "app/smart_crop/worker.py"])

#     # give worker a moment
#     import time
#     time.sleep(2)

#     # tell worker to load TF model
#     r = requests.post("http://127.0.0.1:9000/load")
#     return {"status": "worker started", "worker_response": r.json()}


# @app.post("/run")
# async def run_crop(image: UploadFile = File(...)):
#     contents = await image.read()
#     img64 = base64.b64encode(contents).decode()

#     r = requests.post("http://127.0.0.1:9000/run", json={"image_base64": img64})
#     return r.json()


# @app.post("/unload")
# def unload_model():
#     global worker_process
#     if worker_process:
#         worker_process.kill()
#         worker_process = None
#         return {"status": "worker process killed → GPU memory freed"}
#     return {"status": "nothing to unload"}


# @app.get("/")
# def root():
#     return {"message": "SmartCrop Multi-Process API running"}
