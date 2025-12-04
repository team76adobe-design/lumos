import os
import uvicorn
import base64
import pickle
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from io import BytesIO

# ==== Import SmartCrop utilities ====
# from smartcrop_utils import network
# from smartcrop_utils.actions import command2action, generate_bbox, crop_input
from smartcrop_utils import network
from smartcrop_utils.actions import command2action, generate_bbox, crop_input
app = FastAPI()
model = None
var_dict = None

class ImageRequest(BaseModel):
    image_base64: str


class AutoCroppingModel:
    def __init__(self, var_dict):
        self.var_dict = var_dict

    @tf.function
    def get_global_features(self, image):
        return network.vfn_rl(image, self.var_dict)

    @tf.function
    def get_action(self, image, global_feature, h, c):
        return network.vfn_rl(image, self.var_dict,
                              global_feature=global_feature,
                              h=h, c=c)


# ========= Load model =========
@app.post("/load")
def load_model():
    global model, var_dict
    with open("smartcrop_utils/vfn_rl.pkl", "rb") as f:
        var_dict = pickle.load(f)
    model = AutoCroppingModel(var_dict)
    return {"status": "worker model loaded"}


# ========= Run cropping =========
@app.post("/run")
def run(request: ImageRequest):
    global model

    img_bytes = base64.b64decode(request.image_base64)
    pil = Image.open(BytesIO(img_bytes)).convert("RGB")

    im_np = np.array(pil).astype(np.float32) / 255.0
    im_input = [im_np - 0.5]

    # smartcrop loop here
    batch_size = 1
    terminals = np.zeros(batch_size)
    ratios = np.array([[0, 0, 20, 20]])
    img = crop_input(im_input, generate_bbox(im_input, ratios))
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    global_feature = model.get_global_features(img_tensor)

    h_np = np.zeros([batch_size, 1024], dtype=np.float32)
    c_np = np.zeros([batch_size, 1024], dtype=np.float32)

    for _ in range(50):
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
        bbox = generate_bbox(im_input, ratios)

        if terminals.sum() == batch_size:
            break

        img = crop_input(im_input, bbox)

    xmin, ymin, xmax, ymax = bbox[0]
    cropped = np.array(pil)[ymin:ymax, xmin:xmax]

    # encode as base64
    buf = BytesIO()
    Image.fromarray(cropped).save(buf, format="JPEG")
    cropped64 = base64.b64encode(buf.getvalue()).decode()

    return {"bbox": [int(xmin), int(ymin), int(xmax), int(ymax)],
            "cropped": cropped64}


# Worker runs on a fixed port
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)
