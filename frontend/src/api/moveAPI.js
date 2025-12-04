// moveAPI.js
import { hitBackend, PORTS, getBackendUrl } from "./apiConstants";

// Correcting Port to 8003 (MATCHES PYTHON SCRIPT)
const PORT = PORTS.MOVE; 

/**
 * 1️⃣ LOAD MODEL
 * Endpoint: /load_model
 */
export const loadInpaintModel = async () => {
  try {
    const response = await hitBackend(PORT, '/load_model', {
      method: 'POST',
    });

    const result = await response.json();
    console.log('✔ LOAD MODEL:', result);
    return result;
  } catch (error) {
    console.error('Failed to load Move Model:', error);
    throw error;
  }
};

/**
 * 2️⃣ RUN DRAG INPAINTING
 * Endpoint: /run_drag
 * * @param {Blob|File} imageFile - The source image.
 * @param {Blob|File} maskFile - The mask image.
 * @param {Object} points - { x1, y1, x2, y2 }
 * @param {Object} config - Optional overrides for steps, etc.
 */
export const runDragInpaint = async (imageFile, maskFile, points, config = {}) => {
  const { x1, y1, x2, y2 } = points;

  const formData = new FormData();
  
  // 1. Files (Matches Python: files={...})
  formData.append('image', imageFile);
  formData.append('mask', maskFile);
  
  // 2. Coordinates (Matches Python: data={...})
  formData.append('x1', Math.round(x1));
  formData.append('y1', Math.round(y1));
  formData.append('x2', Math.round(x2));
  formData.append('y2', Math.round(y2));

  // 3. Configs (Matches Python data keys exactly)
  formData.append('num_steps', config.numSteps ?? 8);
  formData.append('guidance_scale', config.guidanceScale ?? 1.0);
  formData.append('strength', config.strength ?? 1.0);
  
  try {
    // Note: Do NOT set 'Content-Type' header manually. 
    // Fetch automatically sets it to 'multipart/form-data; boundary=...' when body is FormData.
    const response = await hitBackend(PORT, '/run_drag', {
      method: 'POST',
      body: formData, 
    });

    const result = await response.json();
    console.log('✔ RUN DRAG:', result);

    // If the backend returns a path but no base64, try to fetch the image
    if (!result.image_base64 && result.path) {
      console.log('Fetching image from path:', result.path);
      const baseUrl = getBackendUrl(PORT);
      // Remove leading ./ if present
      const cleanPath = result.path.replace(/^\.\//, '');
      const imageUrl = `${baseUrl}/${cleanPath}`;
      
      try {
        const imgResponse = await fetch(imageUrl);
        if (imgResponse.ok) {
          const blob = await imgResponse.blob();
          return blob;
        }
      } catch (e) {
        console.error("Failed to fetch image from path:", e);
      }
    }

    return result;
  } catch (error) {
    console.error('Failed to run Drag Inpaint:', error);
    throw error;
  }
};

/**
 * 3️⃣ UNLOAD MODEL
 * Endpoint: /unload_model
 */
export const unloadInpaintModel = async () => {
  try {
    const response = await hitBackend(PORT, '/unload_model', {
      method: 'POST',
    });

    const result = await response.json();
    console.log('✔ UNLOAD MODEL:', result);
    return result;
  } catch (error) {
    console.error('Failed to unload Move Model:', error);
    throw error;
  }
};