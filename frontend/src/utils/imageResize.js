// utils/imageResize.js

// Resize any File to EXACTLY 512x512 (stretched) and keep original size
export const resizeTo512 = (file) => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);

    img.onload = () => {
      const originalWidth = img.width;
      const originalHeight = img.height;

      const canvas = document.createElement("canvas");
      canvas.width = 512;
      canvas.height = 512;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });

      // Stretch-fill: original → 512x512 (same as backend will see)
      // Ensure opacity by filling with white first (or black, but white is standard for transparent bg)
      ctx.fillStyle = "white";
      ctx.fillRect(0, 0, 512, 512);
      ctx.drawImage(img, 0, 0, 512, 512);

      canvas.toBlob(
        (blob) => {
          if (!blob) {
            reject(new Error("Failed to create resized blob"));
            return;
          }
          const resizedFile = new File([blob], "image_512.png", {
            type: "image/png",
          });
          URL.revokeObjectURL(url);
          resolve({ file: resizedFile, originalWidth, originalHeight });
        },
        "image/png",
        1.0
      );
    };

    img.onerror = (e) => {
      URL.revokeObjectURL(url);
      reject(e);
    };

    img.src = url;
  });
};

// Take a 512x512 blob and scale it back to originalWidth x originalHeight
export const rescaleFrom512 = (blob, originalWidth, originalHeight) => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(blob);

    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = originalWidth;
      canvas.height = originalHeight;
      const ctx = canvas.getContext("2d", { willReadFrequently: true });

      ctx.drawImage(img, 0, 0, originalWidth, originalHeight);

      canvas.toBlob(
        (outBlob) => {
          if (!outBlob) {
            reject(new Error("Failed to rescale output blob"));
            return;
          }
          URL.revokeObjectURL(url);
          resolve(outBlob);
        },
        "image/png",
        1.0
      );
    };

    img.onerror = (e) => {
      URL.revokeObjectURL(url);
      reject(e);
    };

    img.src = url;
  });
};
