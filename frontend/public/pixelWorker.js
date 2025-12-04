// Web Worker for pixel-based image processing

// ------------------------------
// Gaussian kernel generator
// ------------------------------
function gaussianKernel(size = 5, sigma = 1.5) {
    const k = [];
    const center = Math.floor(size / 2);
    let sum = 0;

    for (let i = 0; i < size; i++) {
        k[i] = [];
        for (let j = 0; j < size; j++) {
            const x = i - center;
            const y = j - center;
            const v = Math.exp(-(x*x + y*y) / (2 * sigma * sigma));
            k[i][j] = v;
            sum += v;
        }
    }
    // normalize
    for (let i = 0; i < size; i++)
        for (let j = 0; j < size; j++)
            k[i][j] /= sum;

    return k;
}

// ------------------------------
// Apply convolution to entire image (3 channels)
// ------------------------------
function convolve(imgData, width, height, kernel) {
    const size = kernel.length;
    const half = Math.floor(size / 2);
    const output = new Uint8ClampedArray(imgData.length);

    const idx = (x, y) => (y * width + x) * 4;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {

            let r = 0, g = 0, b = 0;

            for (let ky = 0; ky < size; ky++) {
                for (let kx = 0; kx < size; kx++) {

                    const px = Math.min(width - 1, Math.max(0, x + kx - half));
                    const py = Math.min(height - 1, Math.max(0, y + ky - half));

                    const i = idx(px, py);

                    r += imgData[i]     * kernel[ky][kx];
                    g += imgData[i + 1] * kernel[ky][kx];
                    b += imgData[i + 2] * kernel[ky][kx];
                }
            }

            const o = idx(x, y);
            output[o] = r;
            output[o+1] = g;
            output[o+2] = b;
            output[o+3] = 255;
        }
    }
    return output;
}

// ------------------------------
// Worker State
// ------------------------------
let sourcePixels = null;
let sourceWidth = 0;
let sourceHeight = 0;

// Cache for expensive operations
let cachedBlurred = null;
let cachedEdges = null;

self.onmessage = function(e) {
    const { type } = e.data;

    if (type === 'setImage') {
        const { imageData } = e.data;
        sourcePixels = imageData.data;
        sourceWidth = imageData.width;
        sourceHeight = imageData.height;
        
        // Reset caches when image changes
        cachedBlurred = null;
        cachedEdges = null;
        
        // Optionally process immediately if values provided, otherwise wait for update
        if (e.data.editValues) {
            processImage(e.data.editValues);
        }
    } else if (type === 'update') {
        if (!sourcePixels) return;
        processImage(e.data.editValues);
    }
};

function processImage(editValues) {
    const width = sourceWidth;
    const height = sourceHeight;
    // Create a new buffer for output
    const outputPixels = new Uint8ClampedArray(sourcePixels.length);
    
    // Convert slider values to [-1,1]
    const v = {
        exposure:      (editValues.exposure      / 100),
        brightness:    ((editValues.brightness-100) / 100),
        contrast:      ((editValues.contrast-100)   / 100),
        highlights:    (editValues.highlights   / 100),
        brilliance:    ((editValues.brilliance-100) / 100),
        saturation:    ((editValues.saturation-100) / 100),
        vibrancy:      ((editValues.vibrancy-100)   / 100),
        warmth:        (editValues.warmth / 100),
        definition:    ((editValues.definition-100) / 100),
        sharpness:     ((editValues.sharpness-100)  / 100),
        noise:         (editValues.noise_reduction  / 100),
        shadows:       (editValues.shadows || 0) / 100,
    };

    // 1. Compute mean luminance for contrast
    // This could also be cached if we wanted, but it's fast enough (O(N))
    let total = 0;
    let count = sourcePixels.length / 4;
    for (let i = 0; i < sourcePixels.length; i += 4) {
        total += (sourcePixels[i] + sourcePixels[i+1] + sourcePixels[i+2]) / (3*255);
    }
    const mean = total / count;

    // 2. Handle Blurred Image (Lazy Load & Cache)
    // Only calculate if needed AND not already cached
    const needsBlur = Math.abs(v.definition) > 0.01 || Math.abs(v.noise) > 0.01;
    if (needsBlur && !cachedBlurred) {
        const gauss = gaussianKernel(5, 1.5);
        cachedBlurred = convolve(sourcePixels, width, height, gauss);
    }

    // 3. Handle Edges (Lazy Load & Cache)
    const needsEdges = Math.abs(v.sharpness) > 0.01;
    if (needsEdges && !cachedEdges) {
        const laplacian = [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ];
        cachedEdges = convolve(sourcePixels, width, height, laplacian);
    }

    // 4. Process each pixel with all ops
    // We read from sourcePixels (original) and write to outputPixels
    for (let i = 0; i < sourcePixels.length; i += 4) {
        let r = sourcePixels[i] / 255;
        let g = sourcePixels[i+1] / 255;
        let b = sourcePixels[i+2] / 255;

        // Exposure
        if (Math.abs(v.exposure) > 0.01) {
            const expFactor = Math.pow(2, v.exposure * 0.5);
            r *= expFactor; g *= expFactor; b *= expFactor;
        }

        // Brightness
        if (Math.abs(v.brightness) > 0.01) {
            const factor = 1 + v.brightness * 0.5;
            r *= factor; g *= factor; b *= factor;
        }

        // Contrast
        if (Math.abs(v.contrast) > 0.01) {
            const factor = 1 + v.contrast * 0.5;
            r = (r - mean) * factor + mean;
            g = (g - mean) * factor + mean;
            b = (b - mean) * factor + mean;
        }

        // Highlights
        if (Math.abs(v.highlights) > 0.01) {
            const hf = v.highlights * 0.8;
            const lum = (0.299*r + 0.587*g + 0.114*b);
            if (lum > 0.6) {
                const boost = (lum - 0.6) * hf;
                r += boost; g += boost; b += boost;
            }
        }

        // Brilliance
        if (Math.abs(v.brilliance) > 0.01) {
            const factor = 1 + v.brilliance * 0.3;
            const f = (x) => x + factor * x * (1-x) * (x - 0.5);
            r = f(r); g = f(g); b = f(b);
        }

        // Saturation
        if (Math.abs(v.saturation) > 0.01) {
            const factor = 1 + v.saturation * 0.5;
            const gray = 0.299*r + 0.587*g + 0.114*b;
            r = gray + (r - gray)*factor;
            g = gray + (g - gray)*factor;
            b = gray + (b - gray)*factor;
        }

        // Vibrancy
        if (Math.abs(v.vibrancy) > 0.01) {
            const factor = 1 + v.vibrancy * 0.3;
            const gray = 0.299*r + 0.587*g + 0.114*b;
            const sat = (Math.abs(r-gray)+Math.abs(g-gray)+Math.abs(b-gray))/3;
            const mask = 1 - Math.min(1, sat * 2);
            r = r + (r-gray)*factor*mask;
            g = g + (g-gray)*factor*mask;
            b = b + (b-gray)*factor*mask;
        }

        // Warmth
        if (Math.abs(v.warmth) > 0.01) {
            const factor = v.warmth * 0.3;
            r += factor; b -= factor;
        }

        // Definition
        if (needsBlur && cachedBlurred && Math.abs(v.definition) > 0.01) {
            const br = cachedBlurred[i]   / 255;
            const bg = cachedBlurred[i+1] / 255;
            const bb = cachedBlurred[i+2] / 255;
            const detailR = r - br;
            const detailG = g - bg;
            const detailB = b - bb;
            const defFactor = 1 + v.definition * 0.4;
            r = r + detailR * (defFactor - 1);
            g = g + detailG * (defFactor - 1);
            b = b + detailB * (defFactor - 1);
        }

        // Sharpness
        if (needsEdges && cachedEdges && Math.abs(v.sharpness) > 0.01) {
            const er = cachedEdges[i]   / 255;
            const eg = cachedEdges[i+1] / 255;
            const eb = cachedEdges[i+2] / 255;
            const sharpFactor = v.sharpness * 0.5;
            r += er * sharpFactor;
            g += eg * sharpFactor;
            b += eb * sharpFactor;
        }

        // Noise Reduction
        if (needsBlur && cachedBlurred && Math.abs(v.noise) > 0.01) {
            const nr = cachedBlurred[i]   / 255;
            const ng = cachedBlurred[i+1] / 255;
            const nb = cachedBlurred[i+2] / 255;
            const noiseAlpha = Math.min(1, Math.abs(v.noise) * 0.5);
            r = r*(1-noiseAlpha) + nr*noiseAlpha;
            g = g*(1-noiseAlpha) + ng*noiseAlpha;
            b = b*(1-noiseAlpha) + nb*noiseAlpha;
        }
        
        // Shadows
        if (Math.abs(v.shadows) > 0.01) {
            const sf = v.shadows * 0.8;
            const lum = (0.299*r + 0.587*g + 0.114*b);
            if (lum < 0.4) {
                const lift = (0.4 - lum) * sf;
                r += lift; g += lift; b += lift;
            }
        }

        outputPixels[i]   = Math.min(255, Math.max(0, r * 255));
        outputPixels[i+1] = Math.min(255, Math.max(0, g * 255));
        outputPixels[i+2] = Math.min(255, Math.max(0, b * 255));
        outputPixels[i+3] = sourcePixels[i+3]; // Preserve alpha
    }

    self.postMessage({ pixels: outputPixels }, [outputPixels.buffer]);
}
