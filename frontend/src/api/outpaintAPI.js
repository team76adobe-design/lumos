import { hitBackend, PORTS } from "./apiConstants";
// Assuming image resize utilities are still available, though their use might depend on 
// whether the image needs resizing before Base64 encoding.

const PORT = PORTS.OUTPAINT; 
const MODEL_NAME = "Outpainting Model (Stable Diffusion)";

/**
 * 1️⃣ LOAD OUTPAINTING MODEL
 * Sends a POST request to load the outpainting model into memory.
 */
export const loadOutpaintingModel = async () => {
  try {
    const response = await hitBackend(PORT, '/load_model', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json', // Backend expects JSON payload if body is not FormData
      },
    });

    const result = await response.json();
    console.log(`\nLOAD ${MODEL_NAME} RESPONSE:`, result);
    return result;
  } catch (error) {
    console.error(`Failed to load ${MODEL_NAME}:`, error);
    return { status: "error", message: `Connection failed: ${error.message}` };
  }
};

/**
 * Helper to calculate margins and crop the image from a black background.
 */
const calculateMarginsAndCrop = (base64Str) => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imageData.data;
      
      let minX = canvas.width, minY = canvas.height, maxX = 0, maxY = 0;
      let found = false;

      for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
          const i = (y * canvas.width + x) * 4;
          // Check for non-transparent pixel (alpha > 0)
          if (data[i+3] > 0) {
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
            found = true;
          }
        }
      }
      
      if (!found) {
        // Fallback if image is completely black
        resolve({ 
          croppedBase64: base64Str.split(',').pop(), 
          margins: { pixels_left: 0, pixels_right: 0, pixels_top: 0, pixels_down: 0 }
        });
        return;
      }

      // Check if margins are effectively zero (manual crop case)
      const mLeft = minX;
      const mRight = canvas.width - 1 - maxX;
      const mTop = minY;
      const mDown = canvas.height - 1 - maxY;

      if (mLeft <= 0 && mRight <= 0 && mTop <= 0 && mDown <= 0) {
         resolve({
            croppedBase64: base64Str.split(',').pop(),
            margins: { pixels_left: 0, pixels_right: 0, pixels_top: 0, pixels_down: 0 }
         });
         return;
      }

      const width = maxX - minX + 1;
      const height = maxY - minY + 1;
      
      const cropCanvas = document.createElement('canvas');
      cropCanvas.width = width;
      cropCanvas.height = height;
      const cropCtx = cropCanvas.getContext('2d');
      cropCtx.drawImage(canvas, minX, minY, width, height, 0, 0, width, height);
      
      // Get base64 without prefix
      const croppedBase64 = cropCanvas.toDataURL('image/png').split(',').pop();
      
      resolve({
        croppedBase64,
        margins: {
          pixels_left: minX,
          pixels_right: canvas.width - 1 - maxX,
          pixels_top: minY,
          pixels_down: canvas.height - 1 - maxY
        }
      });
    };
    img.onerror = (e) => reject(new Error("Failed to load image for processing"));
    img.src = base64Str;
  });
};

/**
 * 2️⃣ RUN OUTPAINTING INFERENCE
 * @param {string} imageBase64 - Base64 encoded image string (e.g., 'data:image/png;base64,...').
 * @param {number} scale - IGNORED. Scale is calculated from the black image.
 * @param {string} positivePrompt - The positive prompt to guide the generation.
 * @param {string} negativePrompt - The negative prompt to avoid unwanted features.
 * @returns {Promise<{image_base64?: string, error?: string}>} The resulting Base64 image or an error object.
 */
export const inferOutpaint = async (imageBase64, scale = 1.2, positivePrompt = "", negativePrompt = "") => {
  if (!imageBase64) {
    console.error("No image data provided for outpainting");
    return { error: "No image data provided" };
  }

  try {
    // Calculate margins and crop the image
    const { croppedBase64, margins } = await calculateMarginsAndCrop(imageBase64);
    
    console.log("Calculated margins:", margins);

    // If no outpainting is needed (margins are 0), return the cropped image directly
    if (margins.pixels_left === 0 && margins.pixels_right === 0 && 
        margins.pixels_top === 0 && margins.pixels_down === 0) {
      console.log("No margins detected, skipping backend call.");
      return { image_base64: croppedBase64 };
    }

    const payload = {
      image_base64: croppedBase64,
      ...margins,
      positive_prompt: positivePrompt,
      negative_prompt: negativePrompt,
    };

    const response = await hitBackend(PORT, '/infer', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    const result = await response.json();
    console.log('\nINFER RESPONSE:', result);
    
    // The Python response is either { image_base64: '...' } or { error: '...' }
    return result; 

  } catch (error) {
    console.error('Outpainting inference failed:', error);
    return { error: `Network or server error: ${error.message}` };
  }
};

/**
 * 3️⃣ UNLOAD OUTPAINTING MODEL
 * Frees the model resources.
 */
export const unloadOutpaintingModel = async () => {
  try {
    const response = await hitBackend(PORT, '/unload_model', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    const result = await response.json();
    console.log(`\nUNLOAD ${MODEL_NAME} RESPONSE:`, result);
    return result;
  } catch (error) {
    console.error(`Failed to unload ${MODEL_NAME}:`, error);
    return { status: "error", message: `Connection failed: ${error.message}` };
  }
};