import { PORTS, hitBackend } from './apiConstants';

/**
 * LBM (Light-Based Model) API Workflow
 *
 * This API performs relighting on composited images
 * 1. Load the LBM model
 * 2. Run relighting inference
 * 3. Unload the model to free VRAM
 */

/**
 * Load the LBM model
 * @returns {Promise<void>}
 */
export const loadLBMModel = async () => {
  try {
    console.log('Loading LBM model...');
    const response = await hitBackend(PORTS.LBM, '/load_model', {
      method: 'POST',
    });
    console.log('LBM model loaded successfully');
    return response;
  } catch (error) {
    console.error('Error loading LBM model:', error);
    throw new Error('Failed to load LBM model');
  }
};

/**
 * Run relighting on an image
 * @param {File|Blob} imageFile - The composited image to relight
 * @param {number} steps - Number of inference steps (default: 1)
 * @returns {Promise<Blob>} - The relit image as a blob
 */
export const runRelighting = async (imageFile, steps = 1) => {
  try {
    console.log('Running relighting...');
    console.log('File details:', {
      name: imageFile.name,
      type: imageFile.type,
      size: imageFile.size
    });
    console.log('Steps:', steps);

    const formData = new FormData();

    // Ensure the file has proper metadata
    const fileToSend = new File(
      [imageFile],
      imageFile.name || 'image.jpg',
      { type: imageFile.type || 'image/jpeg' }
    );

    formData.append('image', fileToSend);
    formData.append('steps', steps.toString());

    const response = await hitBackend(PORTS.LBM, '/run_relighting', {
      method: 'POST',
      body: formData,
    });

    // Check content type to determine response format
    const contentType = response.headers.get('content-type');

    if (contentType && contentType.includes('application/json')) {
      // API returns JSON with base64-encoded image
      const data = await response.json();
      console.log('Received JSON response from LBM');

      if (data.error) {
        throw new Error(data.error);
      }

      // Check for image data in response
      const imageBase64 = data.image_base64 || data.relit_image_base64 || data.output_base64;
      if (!imageBase64) {
        throw new Error('No image data in response');
      }

      // Convert base64 to blob
      const binaryString = atob(imageBase64);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: 'image/png' });
      console.log('Relighting completed successfully');
      return blob;
    } else {
      // API returns binary image directly
      const blob = await response.blob();
      console.log('Relighting completed successfully (binary response)');
      return blob;
    }
  } catch (error) {
    console.error('Error running relighting:', error);
    throw new Error('Failed to run relighting');
  }
};

/**
 * Unload the LBM model
 * @returns {Promise<void>}
 */
export const unloadLBMModel = async () => {
  try {
    console.log('Unloading LBM model...');
    const response = await hitBackend(PORTS.LBM, '/unload_model', {
      method: 'POST',
    });
    console.log('LBM model unloaded successfully');
    return response;
  } catch (error) {
    console.error('Error unloading LBM model:', error);
    // Don't throw error here - unload failures shouldn't block the user
    console.warn('Failed to unload LBM model, but continuing...');
  }
};

/**
 * Complete workflow: Load model, run relighting, and unload model
 * @param {File|Blob} imageFile - The composited image to relight
 * @param {number} steps - Number of inference steps (default: 1)
 * @returns {Promise<Blob>} - The relit image as a blob
 */
export const processRelighting = async (imageFile, steps = 1) => {
  try {
    // Step 1: Load the model
    await loadLBMModel();

    // Step 2: Run relighting
    const relitBlob = await runRelighting(imageFile, steps);

    // Step 3: Unload the model
    await unloadLBMModel();

    return relitBlob;
  } catch (error) {
    // Ensure we try to unload even if there's an error
    try {
      await unloadLBMModel();
    } catch (unloadError) {
      console.error('Error during cleanup:', unloadError);
    }
    throw error;
  }
};
