import { hitBackend } from './apiConstants';

// Text-to-Image API port
const TEXT_TO_IMAGE_PORT = 8008;

/**
 * Load the text-to-image model into memory
 * @returns {Promise<object>} Response from the load endpoint
 */
export const loadTextToImageModel = async () => {
  try {
    console.log('Loading text-to-image model...');
    const response = await hitBackend(TEXT_TO_IMAGE_PORT, '/load', {
      method: 'GET',
    });
    const data = await response.json();
    console.log('Text-to-image model loaded:', data);
    return data;
  } catch (error) {
    console.error('Failed to load text-to-image model:', error);
    throw error;
  }
};

/**
 * Unload the text-to-image model from memory
 * @returns {Promise<object>} Response from the unload endpoint
 */
export const unloadTextToImageModel = async () => {
  try {
    console.log('Unloading text-to-image model...');
    const response = await hitBackend(TEXT_TO_IMAGE_PORT, '/unload', {
      method: 'GET',
    });
    const data = await response.json();
    console.log('Text-to-image model unloaded:', data);
    return data;
  } catch (error) {
    console.error('Failed to unload text-to-image model:', error);
    throw error;
  }
};

/**
 * Generate an image from a text prompt
 * @param {string} prompt - The text prompt describing the image to generate
 * @param {object} options - Optional generation parameters
 * @param {number} options.height - Image height (default: 1024)
 * @param {number} options.width - Image width (default: 1024)
 * @param {number} options.guidance_scale - Guidance scale (default: 4.5)
 * @param {number} options.steps - Number of inference steps (default: 20)
 * @param {number} options.seed - Random seed for reproducibility (optional)
 * @returns {Promise<object>} Response containing the generated image
 */
export const generateImage = async (prompt, options = {}) => {
  try {
    console.log('Generating image with prompt:', prompt);

    const payload = {
      prompt,
      height: options.height || 1024,
      width: options.width || 1024,
      guidance_scale: options.guidance_scale || 4.5,
      steps: options.steps || 20,
    };

    // Only add seed if provided
    if (options.seed !== undefined) {
      payload.seed = options.seed;
    }

    const response = await hitBackend(TEXT_TO_IMAGE_PORT, '/run', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    // Check if response is an image blob or JSON
    const contentType = response.headers.get('content-type');

    if (contentType && contentType.includes('image')) {
      // Response is an image blob
      const blob = await response.blob();
      console.log('Image generation complete - received image blob');
      return { blob };
    } else {
      // Response is JSON (might contain base64 image or other data)
      const data = await response.json();
      console.log('Image generation complete:', data);
      return data;
    }
  } catch (error) {
    console.error('Image generation failed:', error);
    throw error;
  }
};
