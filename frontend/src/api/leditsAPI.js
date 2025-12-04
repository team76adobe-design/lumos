import { hitBackend } from './apiConstants';

// LEDITS API port (same as inpaint)
const LEDITS_PORT = 8002;

/**
 * Load the LEDITS model into memory
 * @returns {Promise<object>} Response from the load endpoint
 */
export const loadLeditsModel = async () => {
  try {
    console.log('Loading LEDITS model...');
    const response = await hitBackend(LEDITS_PORT, '/load_ledits', {
      method: 'POST',
    });
    const data = await response.json();
    console.log('LEDITS model loaded:', data);
    return data;
  } catch (error) {
    console.error('Failed to load LEDITS model:', error);
    throw error;
  }
};

/**
 * Free/unload the LEDITS model from memory
 * @returns {Promise<object>} Response from the free endpoint
 */
export const freeLeditsModel = async () => {
  try {
    console.log('Freeing LEDITS model...');
    const response = await hitBackend(LEDITS_PORT, '/free_ledits', {
      method: 'POST',
    });
    const data = await response.json();
    console.log('LEDITS model freed:', data);
    return data;
  } catch (error) {
    console.error('Failed to free LEDITS model:', error);
    throw error;
  }
};

/**
 * Run LEDITS image editing with a text prompt (no mask required)
 * @param {string} imageSrc - The source image (data URL or blob URL)
 * @param {string} prompt - The editing prompt (e.g., "george clooney, sunglasses")
 * @param {object} options - Optional parameters
 * @param {string} options.thresholds - Comma-separated thresholds (default: "0.7,0.9")
 * @param {string} options.guidance - Comma-separated guidance values (default: "3,4")
 * @param {string} options.reverse - Comma-separated reverse flags (default: "false,false")
 * @param {boolean} options.save_result - Whether to save result on server (default: true)
 * @returns {Promise<object>} Response containing the edited image
 */
export const runLedits = async (imageSrc, prompt, options = {}) => {
  try {
    console.log('Running LEDITS with prompt:', prompt);

    // Convert imageSrc to a File object
    const response = await fetch(imageSrc);
    const blob = await response.blob();
    const imageFile = new File([blob], 'image.jpg', { type: blob.type || 'image/jpeg' });

    // Create FormData for multipart request
    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('prompt', prompt);
    formData.append('thresholds', options.thresholds || '0.7,0.9');
    formData.append('guidance', options.guidance || '3,4');
    formData.append('reverse', options.reverse || 'false,false');
    formData.append('save_result', options.save_result !== false ? 'true' : 'false');

    const apiResponse = await hitBackend(LEDITS_PORT, '/run_ledits', {
      method: 'POST',
      body: formData,
    });

    // Check if response is an image blob or JSON
    const contentType = apiResponse.headers.get('content-type');

    if (contentType && contentType.includes('image')) {
      // Response is an image blob
      const resultBlob = await apiResponse.blob();
      console.log('LEDITS complete - received image blob');
      return { blob: resultBlob };
    } else {
      // Response is JSON (might contain base64 image or other data)
      const data = await apiResponse.json();
      console.log('LEDITS complete:', data);
      return data;
    }
  } catch (error) {
    console.error('LEDITS failed:', error);
    throw error;
  }
};
