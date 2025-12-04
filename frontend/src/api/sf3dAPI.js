import { PORTS, hitBackend } from './apiConstants';

/**
 * SF3D API Workflow
 *
 * This API converts a 2D image into a 3D model (GLB format)
 * 1. Load the SF3D model
 * 2. Run inference to generate 3D model
 * 3. Unload the model to free VRAM
 */

/**
 * Load the SF3D model
 * @returns {Promise<void>}
 */
export const loadSF3DModel = async () => {
  try {
    console.log('Loading SF3D model...');
    const response = await hitBackend(PORTS.SF3D, '/load', {
      method: 'POST',
    });
    console.log('SF3D model loaded successfully');
    return response;
  } catch (error) {
    console.error('Error loading SF3D model:', error);
    throw new Error('Failed to load SF3D model');
  }
};

/**
 * Generate 3D model from image
 * @param {File|Blob} imageFile - The image file to convert to 3D
 * @returns {Promise<Blob>} - The GLB model as a blob
 */
export const generateSF3DModel = async (imageFile) => {
  try {
    console.log('Generating 3D model...');
    console.log('File details:', {
      name: imageFile.name,
      type: imageFile.type,
      size: imageFile.size
    });

    const formData = new FormData();

    // Ensure the file has proper metadata
    const fileToSend = new File(
      [imageFile],
      imageFile.name || 'image.png',
      { type: imageFile.type || 'image/png' }
    );

    formData.append('image', fileToSend);

    const response = await hitBackend(PORTS.SF3D, '/run', {
      method: 'POST',
      body: formData,
    });

    // Check content type to determine response format
    const contentType = response.headers.get('content-type');

    if (contentType && contentType.includes('application/json')) {
      // API returns JSON with base64-encoded GLB
      const data = await response.json();
      console.log('Received JSON response from SF3D');

      if (data.error) {
        throw new Error(data.error);
      }

      // Check for GLB data in response (API returns mesh_glb_base64)
      const glbBase64 = data.mesh_glb_base64 || data.glb_base64 || data.model_base64 || data.mesh_base64;
      if (!glbBase64) {
        console.error('Response keys:', Object.keys(data));
        throw new Error('No 3D model data in response');
      }

      // Convert base64 to blob
      const binaryString = atob(glbBase64);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: 'model/gltf-binary' });
      console.log('3D model generated successfully');
      return blob;
    } else {
      // API returns binary GLB directly
      const blob = await response.blob();
      console.log('3D model generated successfully (binary response)');
      return blob;
    }
  } catch (error) {
    console.error('Error generating 3D model:', error);
    throw new Error('Failed to generate 3D model');
  }
};

/**
 * Unload the SF3D model
 * @returns {Promise<void>}
 */
export const unloadSF3DModel = async () => {
  try {
    console.log('Unloading SF3D model...');
    const response = await hitBackend(PORTS.SF3D, '/unload', {
      method: 'POST',
    });
    console.log('SF3D model unloaded successfully');
    return response;
  } catch (error) {
    console.error('Error unloading SF3D model:', error);
    // Don't throw error here - unload failures shouldn't block the user
    console.warn('Failed to unload SF3D model, but continuing...');
  }
};

/**
 * Complete workflow: Load model, generate 3D, and unload model
 * @param {File|Blob} imageFile - The image file to convert to 3D
 * @returns {Promise<Blob>} - The GLB model as a blob
 */
export const processImageTo3D = async (imageFile) => {
  try {
    // Step 1: Load the model
    await loadSF3DModel();

    // Step 2: Generate the 3D model
    const glbBlob = await generateSF3DModel(imageFile);

    // Step 3: Unload the model
    await unloadSF3DModel();

    return glbBlob;
  } catch (error) {
    // Ensure we try to unload even if there's an error
    try {
      await unloadSF3DModel();
    } catch (unloadError) {
      console.error('Error during cleanup:', unloadError);
    }
    throw error;
  }
};
