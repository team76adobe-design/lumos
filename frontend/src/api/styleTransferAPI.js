import { hitBackend, PORTS } from './apiConstants';

// Style Transfer API port
const STYLE_TRANSFER_PORT = 8002;

/**
 * Load the style transfer model into memory
 * @returns {Promise<object>} Response from the load endpoint
 */
export const loadStyleTransferModel = async () => {
  try {
    console.log('Loading style transfer model...');
    const response = await hitBackend(STYLE_TRANSFER_PORT, '/load_model', {
      method: 'POST',
    });
    const data = await response.json();
    console.log('Style transfer model loaded:', data);
    return data;
  } catch (error) {
    console.error('Failed to load style transfer model:', error);
    throw error;
  }
};

/**
 * Unload the style transfer model from memory
 * @returns {Promise<object>} Response from the unload endpoint
 */
export const unloadStyleTransferModel = async () => {
  try {
    console.log('Unloading style transfer model...');
    const response = await hitBackend(STYLE_TRANSFER_PORT, '/unload_model', {
      method: 'POST',
    });
    const data = await response.json();
    console.log('Style transfer model unloaded:', data);
    return data;
  } catch (error) {
    console.error('Failed to unload style transfer model:', error);
    throw error;
  }
};

/**
 * Run style transfer with a text prompt and image
 * @param {string} prompt - The style transfer prompt (e.g., "make the scene green and overgrown")
 * @param {string} imageSrc - The source image (data URL or blob URL)
 * @returns {Promise<object>} Response containing the generated/styled image
 */
export const runStyleTransfer = async (prompt, imageSrc) => {
  try {
    console.log('Running style transfer with prompt:', prompt);

    // Convert imageSrc to a File object
    const response = await fetch(imageSrc);
    const blob = await response.blob();
    const imageFile = new File([blob], 'image.jpg', { type: blob.type || 'image/jpeg' });

    // Create FormData for multipart request
    const formData = new FormData();
    formData.append('prompt', prompt);
    formData.append('image', imageFile);

    const apiResponse = await hitBackend(STYLE_TRANSFER_PORT, '/generate', {
      method: 'POST',
      body: formData,
      // Don't set Content-Type header - browser will set it with boundary for FormData
    });

    // Check if response is an image blob or JSON
    const contentType = apiResponse.headers.get('content-type');

    if (contentType && contentType.includes('image')) {
      // Response is an image blob
      const resultBlob = await apiResponse.blob();
      console.log('Style transfer complete - received image blob');
      return { blob: resultBlob };
    } else {
      // Response is JSON (might contain base64 image or other data)
      const data = await apiResponse.json();
      console.log('Style transfer complete:', data);
      return data;
    }
  } catch (error) {
    console.error('Style transfer failed:', error);
    throw error;
  }
};
