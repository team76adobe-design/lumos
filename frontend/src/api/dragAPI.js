    import { hitBackend, PORTS, getBackendUrl } from "./apiConstants";

    // Assuming you have a DRAG entry in your constants, otherwise replace with raw port
    const PORT = PORTS.DRAG; 

    /**
     * Helper to convert File/Blob to Base64 string
     * @param {Blob|File} file 
     * @returns {Promise<string>}
     */
    const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => resolve(reader.result);
        reader.onerror = (error) => reject(error);
    });
    };

    /**
     * 1️⃣ LOAD MODEL
     * Endpoint: /load_model
     */
    export const loadDragModel = async () => {
    try {
        const response = await hitBackend(PORT, '/load_model', {
        method: 'POST',
        });

        const result = await response.json();
        console.log('✔ LOAD DRAG MODEL:', result);
        return result;
    } catch (error) {
        console.error('Failed to load Drag Model:', error);
        throw error;
    }
    };

    /**
     * 2️⃣ RUN DRAG INFERENCE
     * Endpoint: /run_drag
     * @param {Blob|File} imageFile - The source image (RGB).
     * @param {Blob|File} maskFile - The mask image (Grayscale).
     * @param {Object} points - { handle: {x, y}, target: {x, y} }
     * @param {Object} config - Optional overrides for steps, scale, etc.
     */
    export const runDragModel = async (imageFile, maskFile, points, config = {}) => {
    const { handle, target } = points;

    try {
        // 1. Convert Files to Base64 Strings (Required by Python backend)
        const imageB64 = await fileToBase64(imageFile);
        const maskB64 = await fileToBase64(maskFile);

        const formData = new FormData();

        // 2. Image Data (Strings)
        formData.append('image_b64', imageB64);
        formData.append('mask_b64', maskB64);

        // 3. Coordinates
        // Python expects integers. handle_y, handle_x, target_y, target_x
        formData.append('handle_x', Math.round(handle.x));
        formData.append('handle_y', Math.round(handle.y));
        formData.append('target_x', Math.round(target.x));
        formData.append('target_y', Math.round(target.y));

        // 4. Configs
        formData.append('num_inference_steps', config.numSteps ?? 25);
        formData.append('guidance_scale_points', config.guidanceScale ?? 4.0);
        formData.append('output_dir', config.outputDir ?? "./outputs");

        const response = await hitBackend(PORT, '/run_drag', {
        method: 'POST',
        body: formData,
        });

        const result = await response.json();
        console.log('✔ RUN DRAG:', result);

        // 5. Handle Response (Fetch image if backend returned a path)
        // Python returns: { "status": "success", "saved_images": ["./outputs/result.png"] }
        if (result.saved_images && result.saved_images.length > 0) {
            const outputInfo = result.saved_images[0]; // Take the first result
            
            // If it's a file path (string), fetch the blob
            if (typeof outputInfo === 'string') {
                console.log('Fetching image from path:', outputInfo);
                const baseUrl = getBackendUrl(PORT);
                const cleanPath = outputInfo.replace(/^\.\//, ''); // Remove leading ./
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
        }

        return result;
    } catch (error) {
        console.error('Failed to run Drag Inference:', error);
        throw error;
    }
    };

    /**
     * 3️⃣ UNLOAD MODEL
     * Endpoint: /unload_model
     */
    export const unloadDragModel = async () => {
    try {
        const response = await hitBackend(PORT, '/unload_model', {
        method: 'POST',
        });

        const result = await response.json();
        console.log('✔ UNLOAD DRAG MODEL:', result);
        return result;
    } catch (error) {
        console.error('Failed to unload Drag Model:', error);
        throw error;
    }
    };