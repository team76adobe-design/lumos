import { PORTS, hitBackend } from './apiConstants';

/**
 * Upscale API Workflow
 *
 * This API follows a 3-step process:
 * 1. Load the Stable Diffusion Upscale model
 * 2. Run inference on the image with a prompt
 * 3. Unload the model to free VRAM
 */

/**
 * Load the upscale model
 * @returns {Promise<void>}
 */
export const loadUpscaleModel = async () => {
  try {
    console.log('Loading upscale model...');
    // Ensure you have added UPSCALE to your PORTS object in apiConstants.js
    const response = await hitBackend(PORTS.UPSCALE, '/load', {
      method: 'POST',
    });
    console.log('Upscale model loaded successfully');
    return response;
  } catch (error) {
    console.error('Error loading upscale model:', error);
    throw new Error('Failed to load upscale model');
  }
};

/**
 * Upscale an image using Stable Diffusion
 * @param {File|Blob} imageFile - The low-res image file to process
 * @param {string} prompt - The text prompt to guide the upscaling
 * @returns {Promise<Blob>} - The upscaled image
 */
export const upscaleImage = async (imageFile, prompt) => {
  try {
    console.log('Upscaling image...');
    console.log('File details:', {
      name: imageFile.name,
      type: imageFile.type,
      size: imageFile.size
    });
    console.log('Using prompt:', prompt);

    const formData = new FormData();

    // Ensure the file has proper metadata
    const fileToSend = new File(
      [imageFile],
      imageFile.name || 'image.jpg',
      { type: imageFile.type || 'image/jpeg' }
    );

    formData.append('file', fileToSend);
    formData.append('prompt', prompt);

    const response = await hitBackend(PORTS.UPSCALE, '/run', {
      method: 'POST',
      body: formData,
      // Don't set Content-Type header - browser will set it automatically with boundary
    });

    // API returns JSON with base64-encoded image
    const data = await response.json();
    console.log('Received response from upscaler');

    if (data.error) {
      throw new Error(data.error);
    }
    console.log(data);
    // Note: Python backend returns key 'upscaled_image_base64', not 'image_base64'
    if (!data.image_base64) {
      throw new Error('No image data in response');
    }

    // Convert base64 to blob
    const base64Data = data.image_base64;
    const binaryString = atob(base64Data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    const blob = new Blob([bytes], { type: 'image/png' });

    console.log('Image upscaled successfully');
    return blob;
  } catch (error) {
    console.error('Error upscaling image:', error);
    throw new Error('Failed to upscale image');
  }
};

/**
 * Unload the upscale model
 * @returns {Promise<void>}
 */
export const unloadUpscaleModel = async () => {
  try {
    console.log('Unloading upscale model...');
    const response = await hitBackend(PORTS.UPSCALE, '/unload', {
      method: 'POST',
    });
    console.log('Upscale model unloaded successfully');
    return response;
  } catch (error) {
    console.error('Error unloading upscale model:', error);
    // Don't throw error here - unload failures shouldn't block the user
    console.warn('Failed to unload model, but continuing...');
  }
};

/**
 * Complete workflow: Load model, upscale image, and unload model
 * @param {File|Blob} imageFile - The image file to process
 * @param {string} prompt - The text prompt for the upscaler
 * @returns {Promise<File>} - The processed image as a File object
 */
export const processImageUpscale = async (imageFile, prompt) => {
  try {
    // Step 1: Load the model
    await loadUpscaleModel();

    // Step 2: Process the image
    const processedBlob = await upscaleImage(imageFile, prompt);

    // Step 3: Unload the model
    await unloadUpscaleModel();

    // Convert blob to File object
    const processedFile = new File(
      [processedBlob],
      `upscaled-${imageFile.name || 'image.png'}`,
      { type: 'image/png' }
    );

    return processedFile;
  } catch (error) {
    // Ensure we try to unload even if there's an error
    try {
      await unloadUpscaleModel();
    } catch (unloadError) {
      console.error('Error during cleanup:', unloadError);
    }
    throw error;
  }
};