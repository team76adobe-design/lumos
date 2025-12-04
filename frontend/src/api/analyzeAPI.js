import { PORTS, hitBackend } from './apiConstants';

/**
 * Image Analysis API Workflow
 *
 * This API analyzes images for defects/quality issues.
 * It follows a 3-step process:
 * 1. Load the models
 * 2. Analyze the image
 * 3. Unload the models
 */

/**
 * Load the analysis models
 * @returns {Promise<Object>} - Response with status
 */
export const loadAnalyzeModels = async () => {
  try {
    console.log('Loading analyze models...');
    const response = await hitBackend(PORTS.ANALYZE, '/load_models', {
      method: 'POST',
    });
    const data = await response.json();
    console.log('Analyze models loaded:', data);
    return data;
  } catch (error) {
    console.error('Error loading analyze models:', error);
    throw new Error('Failed to load analyze models');
  }
};

/**
 * Analyze an image for defects/quality issues
 * @param {File|Blob} imageFile - The image file to analyze
 * @param {number} topk - Number of top defects to return (default: 3)
 * @returns {Promise<Object>} - Analysis results with detected_defects and analysis
 */
export const analyzeImage = async (imageFile, topk = 3) => {
  try {
    console.log('Analyzing image...');
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

    formData.append('image', fileToSend);
    formData.append('topk', topk.toString());

    const response = await hitBackend(PORTS.ANALYZE, '/analyze', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    console.log('Analysis result:', data);

    if (data.error) {
      throw new Error(data.error);
    }

    return data;
  } catch (error) {
    console.error('Error analyzing image:', error);
    throw new Error('Failed to analyze image');
  }
};

/**
 * Unload the analysis models
 * @returns {Promise<Object>} - Response with status
 */
export const unloadAnalyzeModels = async () => {
  try {
    console.log('Unloading analyze models...');
    const response = await hitBackend(PORTS.ANALYZE, '/unload_models', {
      method: 'POST',
    });
    const data = await response.json();
    console.log('Analyze models unloaded:', data);
    return data;
  } catch (error) {
    console.error('Error unloading analyze models:', error);
    // Don't throw error here - unload failures shouldn't block the user
    console.warn('Failed to unload models, but continuing...');
  }
};

/**
 * Complete workflow: Load models, analyze image, and unload models
 * @param {File|Blob} imageFile - The image file to analyze
 * @param {number} topk - Number of top defects to return (default: 3)
 * @returns {Promise<Object>} - Analysis results
 */
export const processImageAnalysis = async (imageFile, topk = 3) => {
  try {
    // Step 1: Load the models
    await loadAnalyzeModels();

    // Step 2: Analyze the image
    const analysisResult = await analyzeImage(imageFile, topk);

    // Step 3: Unload the models
    await unloadAnalyzeModels();

    return analysisResult;
  } catch (error) {
    // Ensure we try to unload even if there's an error
    try {
      await unloadAnalyzeModels();
    } catch (unloadError) {
      console.error('Error during cleanup:', unloadError);
    }
    throw error;
  }
};
