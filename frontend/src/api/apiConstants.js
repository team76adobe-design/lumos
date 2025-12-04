// Configuration
const POD_ID = '4ixl6ux70n35tj';

// Port mappings based on your instructions
export const PORTS = {
  SEGMENT: 8000,
  BACKGROUND_REMOVAL: 8002,
  INPAINT: 8002,
  MOVE: 8003,
  DRAG: 8004,
  SMARTFRAME: 8006,
  UPSCALE: 8007,
  OUTPAINT: 8005,
  ANALYZE: 8000,
  SF3D: 8004,      // 3D model generation (shares port with DRAG but different endpoints)
  LBM: 8003,
  OUTPAINT: 8005,
  QUILL: 8000,
     // Relighting (shares port with MOVE but different endpoints)
};

export const getBackendUrl = (port) => {
  return `https://${POD_ID}-${port}.proxy.runpod.net`;
};

/**
 * Helper to construct the RunPod URL and perform the fetch request.
 * @param {number} port - The specific port to hit (e.g., 8002).
 * @param {string} path - The endpoint path (e.g., '/run_drag').
 * @param {object} options - Fetch options (method, body, headers).
 */
export const hitBackend = async (port, path, options = {}) => {
  // Construct the base URL following the pattern: https://{id}-{port}.proxy.runpod.net
  const baseUrl = `https://${POD_ID}-${port}.proxy.runpod.net`;
  
  // Ensure the path handles leading slashes correctly relative to the base
  const url = new URL(path, baseUrl).toString();
  // console.log(`hit url ${url}`);

  try {
    const response = await fetch(url, options);
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Backend API Error (${response.status}): ${errorText}`);
    }

    return response;
  } catch (error) {
    // console.error(`Error hitting backend at ${url}:`, error);
    throw error;
  }
};