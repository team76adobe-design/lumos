import { PORTS, hitBackend } from './apiConstants';

/**
 * Background Removal API Workflow
 *
 * This API follows a 3-step process:
 * 1. Load the model
 * 2. Run inference on the image
 * 3. Unload the model
 */

/**
 * Load the background removal model
 * @returns {Promise<void>}
 */
export const loadBackgroundRemovalModel = async () => {
  try {
    console.log('Loading background removal model...');
    const response = await hitBackend(PORTS.BACKGROUND_REMOVAL, '/load', {
      method: 'POST',
    });
    console.log('Background removal model loaded successfully');
    return response;
  } catch (error) {
    console.error('Error loading background removal model:', error);
    throw new Error('Failed to load background removal model');
  }
};

/**
 * Remove background from an image
 * @param {File|Blob} imageFile - The image file to process
 * @returns {Promise<Blob>} - The processed image with background removed
 */
export const removeBackground = async (imageFile) => {
  try {
    console.log('Removing background from image...');
    console.log('File details:', {
      name: imageFile.name,
      type: imageFile.type,
      size: imageFile.size
    });

    const formData = new FormData();

    // Ensure the file has proper metadata
    const fileToSend = new File(
      [imageFile],
      imageFile.name || 'image.jpg',
      { type: imageFile.type || 'image/jpeg' }
    );

    formData.append('file', fileToSend);

    const response = await hitBackend(PORTS.BACKGROUND_REMOVAL, '/run', {
      method: 'POST',
      body: formData,
      // Don't set Content-Type header - browser will set it automatically with boundary
    });

    // API returns JSON with base64-encoded image
    const data = await response.json();
    console.log('Received response:', data);

    if (data.error) {
      throw new Error(data.error);
    }

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

    console.log('Background removed successfully');
    return blob;
  } catch (error) {
    console.error('Error removing background:', error);
    throw new Error('Failed to remove background from image');
  }
};

/**
 * Unload the background removal model
 * @returns {Promise<void>}
 */
export const unloadBackgroundRemovalModel = async () => {
  try {
    console.log('Unloading background removal model...');
    const response = await hitBackend(PORTS.BACKGROUND_REMOVAL, '/unload', {
      method: 'POST',
    });
    console.log('Background removal model unloaded successfully');
    return response;
  } catch (error) {
    console.error('Error unloading background removal model:', error);
    // Don't throw error here - unload failures shouldn't block the user
    console.warn('Failed to unload model, but continuing...');
  }
};

/**
 * Complete workflow: Load model, remove background, and unload model
 * @param {File|Blob} imageFile - The image file to process
 * @returns {Promise<File>} - The processed image as a File object
 */
export const processImageBackgroundRemoval = async (imageFile) => {
  try {
    // Step 1: Load the model
    await loadBackgroundRemovalModel();

    // Step 2: Process the image
    const processedBlob = await removeBackground(imageFile);

    // Step 3: Unload the model
    await unloadBackgroundRemovalModel();

    // Convert blob to File object
    const processedFile = new File(
      [processedBlob],
      'background-removed.png',
      { type: 'image/png' }
    );

    return processedFile;
  } catch (error) {
    // Ensure we try to unload even if there's an error
    try {
      await unloadBackgroundRemovalModel();
    } catch (unloadError) {
      console.error('Error during cleanup:', unloadError);
    }
    throw error;
  }
};
