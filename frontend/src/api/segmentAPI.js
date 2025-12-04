import { hitBackend } from "./apiConstants";
import { resizeTo512, rescaleFrom512 } from "../utils/imageResize";

const PORT = 8000;

/**
 * 1️⃣ LOAD SAM CHECKPOINT
 * Sends a POST request to load the model.
 */
export const loadSam = async () => {
  const formData = new FormData();
  formData.append('checkpoint_path', '/workspace/SAM.pth');

  try {
    const response = await hitBackend(PORT, '/load_sam', {
      method: 'POST',
      body: formData, // fetch automatically sets the Content-Type to multipart/form-data
    });

    const result = await response.json();
    console.log('\nLOAD SAM RESPONSE:', result);
    return result;
  } catch (error) {
    console.error('Failed to load SAM:', error);
  }
};

/**
 * 2️⃣ RUN SEGMENTATION
 * @param {File} imageFile - A File object (e.g., from an file input).
 * @param {Array} xs - Array of x coordinates.
 * @param {Array} ys - Array of y coordinates.
 * @param {Array} labels - Array of labels (1 for foreground, 0 for background).
 */
export const segment = async (imageFile, xs, ys, labels) => {
  if (!imageFile) {
    console.error("No file provided for segmentation");
    return;
  }

  const formData = new FormData();
  formData.append('file', imageFile);
  
  // Append arrays to FormData. 
  // Note: Logic here mimics Python requests behavior. 
  // If the backend expects JSON strings for arrays, use JSON.stringify(xs).
  // If it expects repeated keys (xs=213&xs=214), use a loop.
  // Here we append individually to match standard form-data behavior for lists.
  if (xs) xs.forEach(val => formData.append('xs', val));
  if (ys) ys.forEach(val => formData.append('ys', val));
  if (labels) labels.forEach(val => formData.append('labels', val));
  
  formData.append('save_mask', 'true');

  try {
    const response = await hitBackend(PORT, '/segment', {
      method: 'POST',
      body: formData,
    });

    const result = await response.json();
    console.log('\nSEGMENT RESPONSE:', result);
    return result;
  } catch (error) {
    console.error('Segmentation failed:', error);
  }
};

/**
 * 3️⃣ FREE SAM
 * Frees the model resources.
 */
export const freeSam = async () => {
  try {
    const response = await hitBackend(PORT, '/free_sam', {
      method: 'POST',
    });

    const result = await response.json();
    console.log('\nFREE SAM RESPONSE:', result);
    return result;
  } catch (error) {
    console.error('Failed to free SAM:', error);
  }
};

// Example usage (uncomment to run if you have a valid file object):
// loadSam();
// // const myFile = document.querySelector('input[type="file"]').files[0];
// // segment(myFile);
// // freeSam();