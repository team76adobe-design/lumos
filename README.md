<h1 align="center">LUMOS : The Image Editor of 2030</h1>
<p align="center">
<img width="250"  alt="image" src="https://github.com/user-attachments/assets/ca41d8a5-a9ec-4095-a969-b8ffe2058e25" />

## Demo Video
[![Watch the video](https://img.youtube.com/vi/8_Ilw67fWEA/maxresdefault.jpg)](https://youtu.be/i-tt5Jt7GBY)

### [Watch the Demo on YouTube](https://youtu.be/i-tt5Jt7GBY)

</p> <br>

## Table of Contents
- [Introduction](#introduction)
- [How to Use the Repository](#how-to-use-the-repository)
- [Repository Structure](#repository-structure)
- [Overview of Overall Inference](#overview-of-overall-inference)
- [Text to Image](#text-to-image)
- [Workflow 1 : AI-Enhanced Image editing Tools Specs](#workflow-1--ai-enhanced-image-editing-tools-specs)
- [Workflow 2 : Smart Composition and 3D Aware Object Insertion Specs](#workflow-2--smart-composition-and-3d-aware-object-insertion-specs)
- [Ethical Considerations](#ethical-considerations)
- [Compute Profile](#compute-profile)
- [References](#references)

## Introduction
### Problem understanding 
The problem statement asks us to imagine how creative tools - especially Photoshop will evolve by 2030 in the world where mobile devices and AI-assisted workflows dominate. Current editing tools are powerful but still heavily dependent on manual operations, complex interfaces, and high computational resources. In contrast, the brief envisions a future where creators interact with images more naturally and effortlessly, using simple prompts, fluid gestures and most minimal hardware. The challenge is to indentify gap's in today's creative ecosystem and propose how AI can fill these gaps making editing faster, more intuitive, and more context aware. We are expected to deliver two workflows that demonstrate this shift : features that are not just "automated version of existing tools", but genuinely rethink how editing should feel when powered by intelligent models. These workflows must be grounded in real user pain points, supported by a clear market rationale, and implemented using open-source AI models capable of region selection, generation and inpainting. Overall the problem asks us to blend user research, design thinking and cutting edge AI to build a prototype that reflects the creative experience of 2030-lightweight, intelligent and human-centric.
### Our Solution 
Our solution is built around two complementary workflows that together represent the future of AI‑assisted, mobile‑friendly image editing. Before entering either workflow, the user can begin by uploading an image or generating one using our user‑style personalized LoRA, ensuring a highly customized starting point. From there, the system branches into two specialized pipelines designed to support different creative needs. <br>

#### Workflow 1 : AI-Enhanced Image Editing Tools

<p align="center">
  <img src="https://github.com/user-attachments/assets/6e53bbb0-c362-404d-8928-7250cf453d72"
       alt="workflow1"
       height="400">
</p>

The first workflow focuses on intuitive, fine‑grained image editing using a suite of advanced open‑source AI tools. It includes LeDits++ for image‑to‑image transformation, enabling users to refine or restyle their images with high fidelity. For artistic transformations, we integrate a style‑transfer module that automatically selects the most backendropriate style LoRA based on the user’s prompt and backendlies it seamlessly.Region‑level editing is supported through Segment Anything (SAM), which allows users to isolate any part of the image and then choose to erase it, inpaint new content, or manipulate it using Inpaint4Drag, a state‑of‑the‑art drag-based deformation model.Additionally, the workflow includes Lightning Drag, which enables users to adjust the direction or orientation an object is facing, and Generative Expand, an outpainting tool that extends scenes while preserving visual coherence. Together, these tools form an intelligent, flexible editing environment that reflects the natural, prompt‑driven editing experience envisioned for 2030. <br>
#### Workflow 2: Smart Composition and 3D‑Aware Object Insertion
<p align="center">
  <img src="https://github.com/user-attachments/assets/2983a4fb-a409-41b5-baee-3f8a06ac68b3"
       alt="workflow1"
       height="400"
       width="500"
     >
</p>

The second workflow is designed for high‑quality object insertion and blending, enabling users to integrate new elements into a scene with realism and spatial coherence.The process begins with Smart Crop, which prepares and focuses the base image. The user then selects any object image to insert, and the system automatically removes its background, isolating the subject. This extracted object is passed through a 2D‑to‑3D generation model, which reconstructs a lightweight 3D representation that allows proper orientation, scaling, and positioning relative to the target image. Once the 3D orientation is finalized, the object is composited back into the scene. The blended result is then refined through a relighting model, ensuring that shadows, highlights, and color temperature align with the background. Finally, the combined and harmonized output is delivered, producing an integrated and realistic image with minimal user effort.
<br>

## Repository Structure
```
LUMOS/  
├── backend/        # On-device backend modules and model inference code 
├── cloud/          # Cloud pipelines, training scripts, and processing workflows
├── figma/          # Figma frames, wireframes, and design assets
├── frontend/       # Frontend UI code and application components
└── report/         # All project reports and documentation files
```

## How to Use the Repository
```bash
git clone https://github.com/team76adobe-design/submission.git
cd submission
pip install huggingface-hub
hf auth login
```
Use this Huggingface Key --> "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"

There have to be separate virtual environments for running different parts in the workflow. Overall there are 7 virtual environments to be used for the corresponding models- <br>
1)Virtual Environment 1 - ledits, inpaint4drag, style transfer loras, background removal
```bash
python3.10 -m venv venv_1
source venv_1/bin/activate
pip install -r requirements1.txt
```
2)Virtual Environment 2 - lightningDrag, iopainttest, outpaint
```bash
python3.10 -m venv venv_2
source venv_2/bin/activate
pip install -r requirements2.txt
```
3)Virtual Environment 3 - smartcrop
```bash
python3.10 -m venv venv_3
source venv_3/bin/activate
pip install -r requirements3.txt
```
4)Virtual Environment 4 - Stable-Fast 3D
```bash
python3.10 -m venv venv_4
source venv_4/bin/activate
pip install -r requirements4.txt
```
5)Virtual Environment 5 - LBM Relighting Model,clip and Moondream
```bash
python3.10 -m venv venv_5
source venv_5/bin/activate
pip install -r requirements5.txt
```
6)Virtual Environment 6 - Sana 1.6B Text to Image
```bash
python3.10 -m venv venv_6
source venv_6/bin/activate
pip install -r requirements6.txt
```
7)Virtual Environment 7 - MagicQuill
```bash
cd backend/magicquill
git submodule update --init --recursive
wget -O models.zip "https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliucz_connect_ust_hk/EWlGF0WfawJIrJ1Hn85_-3gB0MtwImAnYeWXuleVQcukMg?e=Gcjugg&download=1"
unzip models.zip
python -m venv Magic_venv
source Magic_venv/bin/activate
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install gradio_magicquill-0.0.1-py3-none-any.whl
#Change "torch==2.1.2", "torchvision==0.16.2" to "torch==2.2.0", "torchvision==0.17.0" in pyproject.toml
cp -f pyproject.toml MagicQuill/LLaVA/
pip install -e MagicQuill/LLaVA/
pip install -r requirements.txt
python main.py
```

Running the Model:- <br>
1.Sana 1.6B Text to Image <br>
```bash
source venv_6/bin/activate 
cd backend/sana
uvicorn main:backend --host 0.0.0.0 --port 8000 
```
2.ledits <br>

```bash
source venv_1/bin/activate 
cd backend/ledits
uvicorn main:backend --host 0.0.0.0 --port 8001 
```
3.inpaint4drag <br>
```bash
source venv_1/bin/activate 
cd backend/inpaint4drag
uvicorn main:backend --host 0.0.0.0 --port 8002 
```
4.style transfer loras <br>
Also you have to change the paths in main according to your path
```bash
source venv_1/bin/activate 
cd backend/style_transfer_loras
gdown --fuzzy "https://drive.google.com/file/d/1ouAGb9GIv6hRhUzu8lAtxWXghL76e6VO/view?usp=sharing"
unzip sd1.5loras.zip 
uvicorn main:backend --host 0.0.0.0 --port 8003 
```
5.background removal <br>
```bash
source venv_1/bin/activate 
cd backend/background_removal
uvicorn main:backend --host 0.0.0.0 --port 8004 
```
6.lightningDrag <br>
```bash
source venv_2/bin/activate 
cd backend/lightningDrag
python download.py
uvicorn main:backend --host 0.0.0.0 --port 8005 
```
7.iopainttest <br> 
```bash
gdown --fuzzy "https://drive.google.com/file/d/1DaQyf1010x3pYG6yDaaQuLKWh39ZlWzU/view?usp=drive_link" -O backend/sam/
source venv_2/bin/activate 
cd backend/iopainttest
uvicorn main:backend --host 0.0.0.0 --port 8006 
```
8.outpaint <br>
```bash
source venv_2/bin/activate 
cd backend/outpaint
uvicorn main:backend --host 0.0.0.0 --port 8007 
```
9.smartcrop <br>
```bash
gdown --fuzzy "https://drive.google.com/file/d/1zxS4Qhm3gbfQUHxp097yz4ytB7EWsTfc/view?usp=sharing" -O backend/smartcrop/smartcrop_utils/
source venv_3/bin/activate 
cd backend/smartcrop
uvicorn main:backend --host 0.0.0.0 --port 8008 
```
10.Stable-Fast 3D <br>
```bash
source venv_4/bin/activate 
cd backend/outpaint
uvicorn main:backend --host 0.0.0.0 --port 8009 
```

11.LBM Relighting Model <br>
```bash
source venv_5/bin/activate 
cd backend/LBM
uvicorn main:backend --host 0.0.0.0 --port 8010
```
12.clip and Moondream <br>
```bash
source venv_5/bin/activate 
cd backend/clipNmoondream
uvicorn main:backend --host 0.0.0.0 --port 8011 
```
13.MagicQuill
```bash
source venv_7/bin/activate 
cd backend/MagicQuill
uvicorn main:backend --host 0.0.0.0 --port 8012
```
14.InvisMark - for adding watermark to the image
```bash
source venv_1/bin/activate
cd backend/InvisMark
gdown --fuzzy "https://drive.google.com/file/d/1XslNWwvKAyclYrY6cTczqvWV9V9vilv5/view?usp=drive_link" 
uvicorn main:app --host 0.0.0.0 --port 8013
```
15.Gaurd Rails
```bash
source venv_5/bin/activate
cd backend/NSFW
uvicorn main:app --host 0.0.0.0 --port 8014
```
Or you can also use the python file to run all commands with a single python file
```bash
python run_all.py
```
These are all local pipelines. Similarly the cloud pipelines can be implemented from the given codes in the **CLOUD_FEATURES**.
## Overview of Overall Inference 

Our system is optimized for **low latency**, **efficient GPU usage**, and **high-quality image outputs**. Below is a detailed explanation of each major component in the inference pipeline.

---

###  Dynamic Model Loading (Load → Run → Unload)

To ensure fast response times and optimal GPU memory utilization, each model is loaded **only when the user selects it**. When a tool is activated, a `POST /load_model` request initializes the corresponding model into GPU memory. After the user provides input—either an image or a prompt—the system executes `POST /run` to perform inference. Once the user returns to the main menu, `POST /unload` is triggered, freeing GPU resources. This prevents unnecessary memory consumption, eliminates repetitive heavy loads, and significantly improves inference speed across the system.

---

###  Optimized High-Resolution Image Pipeline (2K → 512 → 2K)

The pipeline supports images up to **2K resolution**, but for faster computation, inputs are first downsampled to **512×512**. The core editing or generation task is performed at this reduced resolution to minimize latency. After processing, a **diffusion-based upscaler** reconstructs the output back to 2K by converting it into latent space, refining it, and producing a detailed, sharp high-resolution result. This approach balances speed with quality by performing heavy operations only where necessary.

---

###  Algorithmic Enhancements on Stable Diffusion 1.5

We incorporate **optimized algorithms on top of SD 1.5**, dramatically improving both inference speed and output accuracy. These enhancements allow SD 1.5 to operate more efficiently than standard implementations while maintaining strong visual fidelity. Because of its versatility, SD 1.5 is reused for **more than 60% of all tasks**, minimizing model-switch overhead and reducing GPU load, which results in smoother and faster operations throughout the system.

---

###  Reuse of Moondream and CLIP for Auxiliary Tasks

To further streamline the pipeline, models like **Moondream** and **CLIP** are reused across multiple subtasks. Moondream assists with lightweight VLM functions, while CLIP powers semantic reasoning, safety filtering, and classification. Reusing these models avoids repeated initializations, reduces memory fragmentation, and significantly accelerates workflows that rely on vision-language understanding or content validation.

---

###  Summary

Through dynamic model loading, efficient image downsampling and upscaling, algorithmic improvements on SD 1.5, and strategic reuse of VLM components, the system delivers:

-  Low latency  
-  Efficient memory usage  
-  High-quality high-resolution outputs  
-  A clean, scalable, production-ready architecture  

This makes the entire AI image editing experience extremely fast, resource-efficient, and user-friendly.

## Text to Image
### 1.Sana 1.6B Text to Image 
#### Pipeline Explanation 
The Sana 1.6B text-to-image pipeline integrates a DiT-based diffusion model with Nunchaku’s Singular Value Decomposition Quantization -compressed transformer to enable fast, memory-efficient generation without sacrificing image quality. In this setup, the original Sana denoiser is replaced by a low-rank INT4/FP8 Nunchaku transformer, significantly reducing VRAM usage while maintaining strong visual fidelity. The pipeline processes prompts through a text encoder, performs iterative denoising using the compressed DiT transformer, and finally decodes latents through a VAE to produce the final image. This combination allows Sana 1.6B to run smoothly even on mid-range GPUs, making it ideal for lightweight, high-performance API deployment.
#### Inference Time & Memory Used 
* Inference Time - under 3 second for Image size of 2048 X 2048 
* Memory Used - Under 9 Gb VRAM

#### Examples
<img width="256" height="256" alt="sana1 6b-int4" src="https://github.com/user-attachments/assets/2286892e-a44e-44cd-803f-e492cd0ec716" /> <br>
### 2.Personalized Flux.1 Dev + Nunchaku:
#### Pipeline Explanation
Our FLUX.1-dev pipeline combines two powerful components—Nunchaku’s SVD-quantized 4-bit diffusion transformer and a personalized LoRA fine-tuning module—to deliver fast, memory-efficient, identity-aware image generation. During training, we fine-tune FLUX.1-dev using a lightweight LoRA (rank 16) on the UNet while keeping the text encoder frozen, allowing the model to learn a user’s identity from just 12 photos with high fidelity and minimal overfitting. Once personalization is complete, the LoRA module is merged into the quantized Nunchaku FLUX transformer, enabling inference in a low-VRAM environment without sacrificing detail, alignment, or identity consistency. The resulting system is capable of generating high-resolution, photorealistic, and identity-preserving images using simple prompts, while running 2× faster and using only ~30% of full-precision memory. This unified fine-tuning + 4-bit inference pipeline forms one of our core features—allowing rapid, personalized, and cost-efficient image generation on consumer GPUs.
#### Time & Memory Used
* Training Time - 25 minutes of A40
* Inference Time - 4 seconds
* Memory Used - **18 Gb VRAM using on INT4 bit with Singular Value Decomposition Quantization over original Flux Dev which runs on 50Gb VRAM**
#### Examples
<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/f0350db6-bbe3-492f-b88e-8e2d5c6cf445" /> 
<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/88cca826-b5b2-4f3c-90a4-0f5d44aac11c" /> 
<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/9a33c1a9-bbbe-406f-8a48-abe84684a2fe" />




## Workflow 1 : AI-Enhanced Image editing Tools Specs


### 1.MagicQuill
#### Pipeline Explanation 
The system operates through a unified stroke-driven editing pipeline in which user-provided add, remove, and color strokes are first converted into structured masks that encode localized geometric and backendearance cues. These masks, along with the original image, are supplied to a fine-tuned LLaVA-1.5 model that performs ``Draw \& Guess'' inference to interpret the semantic intention behind the strokes, enabling the extraction of high-level intent even from abstract signals---for example, recognizing that subtle wavy strokes on a face are meant to introduce realistic wrinkles. The inferred prompt is then combined with the original image and the stroke-derived masks and passed into a fine-tuned Stable Diffusion~1.5 model augmented with inpainting and control branches. This diffusion-based editing module integrates structural (edge) and backendearance (color) conditions to regenerate the modified regions while preserving unedited content, producing high-quality, semantically aligned outputs with fast inference and precise spatial control.
#### Inference Time & Memory Used 
* Inference Time - 5 seconds  
* Memory Used - 11 Gb VRAM

#### Examples
| Input Image | Brush Stroked Used on Image |Auto VLM Generated Prompt | Output Image |
|-------------|---------|--------------|--------------|
|<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/7a998551-085a-4764-9524-f1c092e07701" />|<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/3878abf7-4c84-486a-9a86-a52c8e7c8bc6" />|deer|<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/ae9b8415-0be6-488b-9b00-3de41039bfdc" />|
|<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/3f5118ff-9b68-4943-8aa2-4c2d7e04c8bf" />|<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/88bb16f4-980b-4721-80aa-a96bbd2025c0" />|Wrinkles|<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/a196cdb5-7336-4d30-95ee-e0001d6407d3" />|





### 2.Ledits ++ <br>
#### Pipeline Explanation   
LEDITS++ first inverts the input image into the diffusion latent space using a fast, error-free DPM-Solver++ inversion, ensuring perfect reconstruction. The model then backendlies text-guided edit vectors that modify only the intended semantic regions, guided by implicit masks derived from attention and noise-difference maps. Finally, the edited latent is decoded back into an image, producing precise, localized changes without affecting the rest of the content.
#### Inference Time & Memory Used 
* Inference Time - 6 seconds 
* Memory Used - Under 8.5 Gb VRAM or approximately 9 Gb VRAM

#### Examples
| Input Image | Prompt | Output Image |
|-------------|---------|--------------|
| <img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/3eb2f64d-8aa8-499a-a3a6-2fe78ff83f99" />| Add Sunglasses | <img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/7840ac6a-db2e-4dbf-8bf1-2da2e519c2fc" />|
| <img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/9cceebae-5ee1-4561-852e-91b5bfedd90c" />|Increase the fur, while adding a little blackish shade. Also make ears more stiff. Turn eyes to yellow. | <img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/2ddf23a3-84ee-4b9a-ad28-2d2ba2cbc28d" />|

### 3.Inpaint4Drag 
#### Pipeline Explanation 
Inpaint4Drag first takes a user-defined mask along with drag handles and target points to compute a precise pixel-space deformation of the selected region. The warped image is then analyzed to identify newly exposed or empty areas, which are passed to an inpainting model (such as SD 1.5) to synthesize missing content. This two-stage pipeline—deterministic geometric warping followed by targeted inpainting—enables accurate, high-resolution edits with real-time responsiveness.
#### Inference Time & Memory Used 
* Inference Time - 7 seconds 
* Memory Used - 3 Gb

#### Examples
| Input Image |  Output Image|
|-------------|--------------|
|<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/2d8c5c52-2e33-4138-80ff-c096f7edc245" />|<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/c61b51f2-0cf8-4d7f-b4a3-eab14e023518" />|
|<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/4ee55504-7e52-44ee-a55d-e8f0dfe49ae3" />|<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/48c94705-0b21-430c-b062-937ea0a08360" />|
### 4.Lightning Drag 
#### Pipeline Explanation 
LightningDrag takes the input image, user-defined handle–target point pairs, and an optional mask, and encodes them through a point-embedding network while preserving backendearance features using a reference-based encoder. These embeddings condition a Stable Diffusion–based inpainting backbone, which generates the manipulated image by following point movements while keeping untouched regions intact. Through this conditional generation workflow, LightningDrag produces fast, accurate drag-based edits with strong structural consistency.
#### Inference Time & Memory Used 
* Inference Time - 0.5 seconds 
* Memory Used - 7 Gb

#### Examples
| Input Image |  Output Image|
|-------------|--------------|
|<img width="673" height="440" alt="image" src="https://github.com/user-attachments/assets/81cc3957-5f7e-4011-a82a-00c27d6bb768" />|<img width="801" height="508" alt="image" src="https://github.com/user-attachments/assets/beb7a6e1-687d-4ccc-ac9e-66fc72b4bc26" />|

### 5. SAM (Segment Anything Model): Universal Image Segmentation
#### Pipeline Explanation 
SAM first encodes the input image into a high-dimensional feature map using a ViT-based image encoder. User prompts—such as positive points indicating what to include and negative points indicating what to exclude—are converted into prompt embeddings and fused with the image features in the mask decoder. The decoder then generates precise segmentation masks in real time, using the combination of positive and negative cues to accurately isolate the desired region without requiring any model retraining.
#### Inference Time & Memory Used 
* Inference Time - 2 seconds 
* Memory Used - 3 Gb 


### 6.PowerPaint: Unified Diffusion-Based Region Editing using SD 1.5
#### Pipeline Explanation 
PowerPaint takes an input image, a user-defined mask, and a text instruction, and feeds them into a diffusion-based backbone enhanced with spatial and semantic conditioning. The model interprets the instruction to decide whether it should erase, inpaint, or outpaint, and then synthesizes the masked region while preserving global scene coherence. This unified pipeline enables consistent, high-quality region editing across multiple tasks without switching models.
#### Inference Time & Memory Used 
* Inference Time - 7 seconds 
* Memory Used - 9 Gb

#### Examples
**Erase + Inpaint**
| Input Image |  Output Image|
|-------------|--------------|
|<img width="1240" height="825" alt="image" src="https://github.com/user-attachments/assets/66c33b9d-c973-4a46-8229-47bc38b8987f" />|<img width="1246" height="830" alt="image" src="https://github.com/user-attachments/assets/a018291b-842f-4141-bc9e-10c0317660c2" />|

**Generative Expand (Outpainting)**

| Input Image |  Output Image|
|-------------|--------------|
|<img width="1015" height="671" alt="image" src="https://github.com/user-attachments/assets/a3d6e6c8-d01c-4409-8536-6b831198e6e0" />|<img width="1248" height="832" alt="image" src="https://github.com/user-attachments/assets/c1c9eb45-1366-4797-8df0-19d98035aba6" />|


### 7.stabilityai/stable-diffusion-x4-upscaler : Latent Diffusion Model 
#### Pipeline Explanation 
The ×4 upscaler first encodes the low-resolution input image into a compact latent representation using a pretrained VAE, reducing the problem to a lighter and more expressive latent space. A time-conditioned U-Net then performs diffusion-based denoising on these latents, guided through cross-attention layers that integrate text or other conditioning signals to refine structure and detail. After the latent has been fully denoised, the autoencoder decodes it back into pixel space, producing a high-resolution image with enhanced sharpness and fidelity.
#### Inference Time & Memory Used 
* Inference Time - 2 seconds 
* Memory Used - 4.5 Gb VRAM

### 8.Dynamic Style Transfer with SD1.5 and Semantic LoRA Selection
#### Pipeline Explanation 
Our dynamic style-transfer system uses Stable Diffusion 1.5 together with a scalable semantic LoRA-selection engine to automatically choose the most suitable artistic style for any user input. When the user provides an image and a prompt, the Moondream-2 vision-language model first analyzes the image and enriches the user prompt based on the LoRA’s metadata, improving semantic clarity and artistic intent. The system then embeds this enhanced prompt into a SentenceTransformer index to retrieve the most relevant LoRA through cosine-similarity search. After selection, the LoRA’s trigger words are automatically backendended to the refined prompt and the LoRA is injected on-the-fly into the Img2Img pipeline without reloading the base model. This design scales effortlessly to hundreds or thousands of LoRAs, adapts to changing artistic trends, and completely removes the need for users to manually choose styles—offering a fluid, intelligent, and highly adaptive style-transfer workflow.
#### Inference Time & Memory Used 
* Inference Time - 8 seconds 
* Memory Used - 11 Gb VRAM
#### Examples
| Input Image |  Prompt | LoRA selected| Output Image|
|-------------|--------------|--------------|--------------|
| <img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/84d452cd-3484-46ab-afe4-a289517723aa" /> |Turn this Man into a retro game character|Retro Game Art|<img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/1967c51d-dc04-4227-ba1e-093360136619" />|


### 12. Hybrid CLIP + Moondream2 Defect Analysis System
#### Pipeline Explanation
The Hybrid CLIP + Moondream2 system performs intelligent photographic defect analysis by first using CLIP to embed the input image and compare it against a curated vocabulary of 101 defect descriptions, ranking the top three most likely issues through cosine similarity. These defect candidates, along with the image, are then passed to the Moondream2 vision-language model, which verifies whether each defect is genuinely present and provides a grounded, human-readable explanation based on its visual reasoning. This combined retrieval-and-verification pipeline delivers fast, scalable, and highly reliable defect detection—catching lighting issues, blur, distortions, color problems, and AI artifacts—while ensuring that every prediction is context-aware, interpretable, and accurate.
#### Inference Time & Memory Used
* Inference Time - 9 seconds
* Memory Used - 4 Gb VRAM



---

## Workflow 2 : Smart Composition and 3D Aware Object Insertion Specs
### 1.A2-RL: Aesthetics Aware Reinforcement Learning for Smart Cropping
#### Pipeline Explanation 
A2-RL begins by extracting visual features from the input image and initializes a cropping window that the reinforcement learning agent iteratively adjusts through predefined actions such as scaling, translating, and reshaping. At each step, the agent evaluates the aesthetics-aware reward function to guide the crop toward a more pleasing composition, continuing until the termination action signals that the optimal crop has been found. This sequential decision-making pipeline enables fast, intelligent, and high-quality cropping without exhaustive search.
#### Inference Time & Memory Used 
* Inference Time - 5 seconds 
* Memory Used - 16 Gb

#### Examples
| Input Image |  Output Image|
|-------------|--------------|
|<img width="522" height="296" alt="image" src="https://github.com/user-attachments/assets/b4b18361-80b8-4d93-a2b3-6920a2e8e6e7" />|<img width="524" height="311" alt="image" src="https://github.com/user-attachments/assets/43597d74-41f0-4e27-a005-a50e6d6b43e9" />|
|<img width="525" height="333" alt="image" src="https://github.com/user-attachments/assets/0a3cf156-a655-4e0b-a40c-5f2f5c22fbb5" />|<img width="522" height="339" alt="image" src="https://github.com/user-attachments/assets/1d948183-b54d-438c-be26-7eb7972f31e2" />|

### 2. BRIAAI / RMBG-2.0: High-Resolution Background Removal 
#### Pipeline Explanation
RMBG-2.0 performs high-resolution background removal by leveraging the BiRefNet architecture, which combines global semantic understanding with fine-detail reconstruction. The model first uses a transformer-based Localization Module to generate a coarse foreground map that captures object structure even in cluttered scenes. This is then refined by the bilateral-reference Reconstruction Module, which integrates multi-scale contextual patches with gradient-based edge cues to recover fine contours, hair strands, soft boundaries, and intricate textures. Supported by an auxiliary edge-aware loss, this two-stage pipeline produces sharp, production-grade foreground masks that remain accurate even on large, complex images.
#### Inference Time & Memory 
* Inference Time - 7 seconds
* Memory Used - 11 Gb VRAM
#### Examples 

| Input Image |  Output Image|
|-------------|--------------|
|<img width="598" height="398" alt="image" src="https://github.com/user-attachments/assets/7e541123-9e56-4497-ab6c-54c5c613322e" />|<img width="461" height="305" alt="image" src="https://github.com/user-attachments/assets/59adfb33-1807-44b9-a8e8-2454a3d2095d" />|



### 3. Stable Fast 3D: Single-Image Feed-Forward 3D Reconstruction
#### Pipeline Explanation
Stable Fast 3D reconstructs a complete, textured 3D mesh from a single image through a feed-forward pipeline that predicts geometry, materials, and textures in one pass. A transformer-based network first infers 3D structure, surface normals, and material parameters while generating a latent representation for texture synthesis. The system then performs illumination disentanglement to remove baked-in lighting, producing clean albedo textures suitable for realistic relighting. A differentiable mesh extraction stage generates the 3D shape, followed by efficient UV unwrbackending and texture baking to produce a high-quality texture atlas rather than simple vertex colors. Finally, SF3D outputs complete material maps and normal maps, enabling accurate shading under novel lighting. This unified pipeline produces production-ready 3D assets in ~0.5 seconds, outperforming traditional multi-view or optimization-heavy methods in both speed and fidelity.
#### Inference Time and Memory
* Inference Time - 5 seconds
* Memory Used - 6 Gb VRAM

#### Examples
Input Image :-  
<img width="256" src="https://github.com/user-attachments/assets/6a3be83e-c68f-4b54-bbf2-d37ef4631c70" />

Output from some angles of rotation :-  

<img width="35%" src="https://github.com/user-attachments/assets/d9fbef89-5b3e-46f5-b139-43829c51757d" />
<img width="35%" src="https://github.com/user-attachments/assets/2ae3904a-b0f1-4516-a3a7-b9936685fdf6" />



<img width="35%" src="https://github.com/user-attachments/assets/50aeb36f-9318-44c5-86d9-e4f60657c98e" />
<img width="35%" src="https://github.com/user-attachments/assets/1a5d612b-6551-4172-be02-aece1416250e" />



### 4. LBM Relighting: Single-Step Latent-Space Illumination Transfer
#### Pipeline 
LBM Relighting performs illumination transfer in a single step by mbackending the input image through a learned latent-space transformation that preserves geometry while modifying lighting. The image is first encoded into a VAE latent, where a stochastic latent “bridge” is defined between the source and target illumination conditions, conditioned on parameters such as light direction, strength, or environment maps. A neural denoiser (U-Net) is trained to backendroximate the drift along this bridge, enabling a direct, one-shot conversion of the source latent into a target latent without iterative diffusion. Once decoded, the output exhibits realistic changes in shading, highlights, shadows, and global illumination while maintaining object boundaries and scene structure. This latent-transport pipeline offers relighting quality comparable to multi-step diffusion models but at a fraction of the computational cost, making it ideal for real-time and interactive workflows.
#### Inference Time and Memory 
* Inference Time - 5 seconds
* Memory Used - 10 Gb VRAM
#### Examples
| Input Image |  Output Image|
|-------------|--------------|
|<img width="1476" height="836" alt="image" src="https://github.com/user-attachments/assets/50733bf8-21d8-4cd6-b054-7c0035c05fa5" />|<img width="1497" height="835" alt="image" src="https://github.com/user-attachments/assets/568b1b0c-ee86-4e9a-8035-69f4835b1be8" />|

---

## Ethical Considerations
### 1.InvisMark
#### Pipeline Explanation 
InvisMark embeds an invisible 256-bit watermark by passing the input image through a neural encoder that adds a subtle, imperceptible residual into the pixel space. During training, the embedded image is routed through a robustness module that applies real-world distortions—such as JPEG compression, noise, blur, cropping, and color shifts—to ensure the watermark remains stable under common manipulations. A paired neural decoder is then used to reliably extract the watermark from the distorted outputs, while the loss function jointly optimizes perceptual similarity and extraction accuracy. This feed-forward encode–distort–decode pipeline enables high-capacity, invisible watermarking that remains intact even after aggressive editing or compression.
#### Examples
<img width="1717" height="604" alt="image" src="https://github.com/user-attachments/assets/5ace0c40-c274-4a50-b444-9aac28916a35" />

### 2.NSFW Content Guardrails 
#### Pipeline Explanation 
Using a CLIP-based similarity system as guard rails provides a fast, lightweight, and highly adaptable way to detect harmful or sensitive content across a very broad risk spectrum. While many existing moderation approaches—especially classical detectors and even several commercial VLM-based filters—perform strongly mainly on sexual or nudity-related content, they often miss non-sexual harms such as violence, weapons, extremism, drugs, or psychologically disturbing scenes. In contrast, CLIP embeds both images and a rich taxonomy of safety labels into the same semantic space, enabling it to surface the top 3 highest-scoring NSFW categories and mark them as violations when their similarity exceeds a defined threshold, creating a precise and transparent rule-based moderation pipeline. This makes detection fast (10–50× faster than VLM captioning), far more GPU-efficient, and fully controllable—developers can easily add, remove, or tune categories to match policy requirements. Overall, this CLIP-driven strategy offers a significantly broader, more interpretable, and more scalable guard-rails system compared to methods that specialize only in sexual content moderation.
#### Examples
   <img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/3c886b19-def2-401d-a3e8-e8875772e130" /><br>
   <img width="256" height="256" alt="image" src="https://github.com/user-attachments/assets/42282726-1d30-474c-9095-935ea56e2ff6" /><br>



### 3.Privacy


## Compute Profile
All experiments were conducted on Runpod NVIDIA RTX 4090 GPUs for inference, while Runpod NVIDIA A40 GPUs were used for training the personalized LoRA models.

## References

1. **Black Forest Labs. (2024).** *FLUX.1-dev.* Hugging Face.  
   https://huggingface.co/black-forest-labs/FLUX.1-dev  

2. **Boss, M., Huang, Z., Vasishta, A., & Jampani, V. (2024).**  
   *SF3D: Stable Fast 3D Mesh Reconstruction with UV-unwrbackending and Illumination Disentanglement.*  
   arXiv:2408.00653 [cs.CV]  
   https://arxiv.org/abs/2408.00653  

3. **Brack, M., Friedrich, F., Kornmeier, K., Tsaban, L., Schramowski, P., Kersting, K., & Passos, A. (2023).**  
   *LEDITS++: Limitless Image Editing using Text-to-Image Models.*  
   arXiv:2311.16711 [cs.CV]  
   https://arxiv.org/abs/2311.16711  

4. **BRIA AI. (2024).** *BRIA RMBG-2.0: Background Removal Model.* Hugging Face.  
   https://huggingface.co/briaai/RMBG-2.0  

5. **Chadebec, C., Tasar, O., Sreetharan, S., & Aubin, B. (2025).**  
   *LBM: Latent Bridge Matching for Fast Image-to-Image Translation.*  
   arXiv:2503.07535 [cs.CV]  
   https://arxiv.org/abs/2503.07535  

6. **CivitAI Community. (2024).** *LoRA Models Database.*  
   https://civitai.com  

7. **Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., Dollár, P., & Girshick, R. (2023).**  
   *Segment Anything.*  
   arXiv:2304.02643 [cs.CV]  
   https://arxiv.org/abs/2304.02643  

8. **Li, D., Wu, H., Zhang, J., & Huang, K. (2017).**  
   *A2-RL: Aesthetics Aware Reinforcement Learning for Image Cropping.*  
   arXiv:1709.04595 [cs.CV]  
   https://arxiv.org/abs/1709.04595  

9. **Li, M., Lin, Y., Zhang, Z., Cai, T., Li, X., Guo, J., Xie, E., Meng, C., Zhu, J.-Y., & Han, S. (2024).**  
   *SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models.*  
   arXiv:2411.05007 [cs.CV]  
   https://arxiv.org/abs/2411.05007  

10. **Lu, J., & Han, K. (2025).**  
    *Inpaint4Drag: Repurposing Inpainting Models for Drag-Based Image Editing via Bidirectional Warping.*  
    arXiv:2509.04582 [cs.CV]  
    https://arxiv.org/abs/2509.04582  

11. **MIT HAN Lab & Nunchaku Team. (2024).**  
    *Nunchaku: 4-Bit Diffusion Model Inference.* GitHub.  
    https://github.com/nunchaku-tech/nunchaku  

12. **Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021).**  
    *Learning Transferable Visual Models From Natural Language Supervision.*  
    arXiv:2103.00020 [cs.CV]  
    https://arxiv.org/abs/2103.00020  

13. **Sanster. (2024).**  
    *PowerPaint v2: High-Quality Inpainting and Outpainting.* Hugging Face.  
    https://huggingface.co/Sanster/PowerPaint_v2  

14. **Shi, Y., Liew, J. H., Yan, H., Tan, V. Y. F., & Feng, J. (2024).**  
    *LightningDrag: Lightning Fast and Accurate Drag-based Image Editing Emerging from Videos.*  
    arXiv:2405.13722 [cs.CV]  
    https://arxiv.org/abs/2405.13722  

15. **Stability AI. (2022).**  
    *Stable Diffusion x4 Upscaler.* Hugging Face.  
    https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler  

16. **Liu, Z., Yu, Y., Ouyang, H., Wang, Q., Cheng, K. L., Wang, W., Liu, Z., Chen, Q., & Shen, Y. (2024).**  
    *MagicQuill: An Intelligent Interactive Image Editing System.*  
    arXiv:2411.09703 [cs.CV]  
    https://arxiv.org/abs/2411.09703  

17. **Vikhyat K. (2024).**  
    *Moondream2: A Tiny Vision Language Model.* Hugging Face.  
    https://huggingface.co/vikhyatk/moondream2  

18. **Xie, E., Chen, J., Chen, J., Cai, H., Tang, H., Lin, Y., Zhang, Z., Li, M., Zhu, L., Lu, Y., & Han, S. (2024).**  
    *SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers.*  
    arXiv:2410.10629 [cs.CV]  
    https://arxiv.org/abs/2410.10629

19. **Rui Xu, Mengya Hu, Deren Lei, Yaxi Li, David Lowe, Alex Gorevski, Mingyu Wang, Emily Ching, and Alex Deng (2024).** 
   *InvisMark: Invisible and Robust Watermarking for AI-generated Image Provenance*
    arXiv:2411.07795 [cs.CV]  
   https://arxiv.org/abs/2411.07795
