import { hitBackend } from "./apiConstants";
const PORT = 8002

// ==========================================
//  IOPAINT API SERVICE
// ==========================================

/**
 * 1) LOAD MODEL
 * POST /load_model
 */
export const loadInpaintModel = async () => {
  const response = await hitBackend(PORT, "/load_model", {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(`Failed to load model. Status: ${response.status}`);
  }

  return await response.json(); // Returns { status: "loaded" | "already_loaded" }
};

/**
 * 2) RUN INPAINT
 * POST /run_inpaint
 * Uploads image + mask as FormData
 */
export const runInpaint = async (
  imageFile, 
  mask, 
  prompt = "", 
  negativePrompt = "", 
  outputPath = "out.png"
) => {
  // Create FormData for file uploads
  const formData = new FormData();
  formData.append("image", imageFile);
  
  if (mask instanceof File) {
    formData.append("mask", mask);
  } else {
    formData.append("mask_path", mask);
  }

  formData.append("prompt", prompt);
  formData.append("negative", negativePrompt);
  formData.append("output_path", outputPath);

  const response = await hitBackend(PORT, "/run_inpaint", {
    method: "POST",
    body: formData, // fetch automatically sets Content-Type to multipart/form-data
  });

  if (!response.ok) {
    try {
      const errData = await response.json();
      throw new Error(errData.error || `Inpaint failed. Status: ${response.status}`);
    } catch (e) {
      throw new Error(`Inpaint failed. Status: ${response.status}`);
    }
  }

  const contentType = response.headers.get("content-type");
  if (contentType && contentType.includes("application/json")) {
    return await response.json();
  } else {
    return await response.blob();
  }
};

/**
 * 3) UNLOAD MODEL
 * POST /unload_model
 */
export const unloadInpaintModel = async () => {
  const response = await hitBackend(PORT, "/unload_model", {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(`Failed to unload model. Status: ${response.status}`);
  }

  return await response.json(); // Returns { status: "freed" | "already_empty" }
};