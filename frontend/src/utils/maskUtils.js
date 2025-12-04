export const generateBinaryMask = (maskBlob) => {
  return new Promise((resolve, reject) => {
    if (!maskBlob) {
      reject("No mask blob provided");
      return;
    }

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');

      // 1. Fill with black (Background)
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // 2. Draw the mask (White regions)
      // The maskBlob is already white-on-transparent from PolygonHighlighter
      ctx.drawImage(img, 0, 0);

      // 3. Return Base64
      resolve(canvas.toDataURL('image/png'));
    };
    img.onerror = (e) => reject(e);
    img.src = URL.createObjectURL(maskBlob);
  });
};

export const generateVectorMask = (strokes, imageSrc, useColor = false, drawBackground = false) => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');

      if (drawBackground) {
        // Draw black background first to ensure opacity
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        // Then draw the image
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      } else {
        // Fill black
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      }

      // Draw strokes
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      strokes.forEach(stroke => {
        if (stroke.points && stroke.points.length > 1) {
          // Use stroke color if useColor is true, otherwise default to white
          ctx.strokeStyle = useColor ? (stroke.color || 'white') : 'white';
          ctx.lineWidth = stroke.width;
          ctx.beginPath();
          ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
          for (let i = 1; i < stroke.points.length; i++) {
            const cp = stroke.points[i - 1];
            const ep = stroke.points[i];
            ctx.quadraticCurveTo(
              cp.x, cp.y,
              (cp.x + ep.x) / 2, (cp.y + ep.y) / 2
            );
          }
          ctx.stroke();
        }
      });

      resolve(canvas.toDataURL('image/png'));
    };
    img.onerror = reject;
    img.src = imageSrc;
  });
};

export const generateBackendMask = (strokes, originalWidth, originalHeight, targetWidth, targetHeight) => {
  return new Promise((resolve) => {
    // 1. Create a canvas at ORIGINAL resolution to draw strokes accurately
    const originalCanvas = document.createElement('canvas');
    originalCanvas.width = originalWidth;
    originalCanvas.height = originalHeight;
    const ctxOriginal = originalCanvas.getContext('2d');

    // Fill with Opaque Black (RGB=0, A=1)
    ctxOriginal.fillStyle = 'black';
    ctxOriginal.fillRect(0, 0, originalWidth, originalHeight);

    // Set Composite Operation to 'destination-out'
    ctxOriginal.globalCompositeOperation = 'destination-out';

    // Draw Strokes
    ctxOriginal.lineCap = 'round';
    ctxOriginal.lineJoin = 'round';
    ctxOriginal.strokeStyle = 'black';

    strokes.forEach(stroke => {
      if (stroke.points && stroke.points.length > 1) {
        ctxOriginal.lineWidth = stroke.width; // Assuming stroke.width is in original coordinates
        ctxOriginal.beginPath();
        ctxOriginal.moveTo(stroke.points[0].x, stroke.points[0].y);
        for (let i = 1; i < stroke.points.length; i++) {
          const cp = stroke.points[i - 1];
          const ep = stroke.points[i];
          ctxOriginal.quadraticCurveTo(
            cp.x, cp.y,
            (cp.x + ep.x) / 2, (cp.y + ep.y) / 2
          );
        }
        ctxOriginal.stroke();
      }
    });

    // 2. Resize to TARGET resolution (512x512)
    const targetCanvas = document.createElement('canvas');
    targetCanvas.width = targetWidth;
    targetCanvas.height = targetHeight;
    const ctxTarget = targetCanvas.getContext('2d');

    // Draw the original canvas onto the target canvas, scaling it
    ctxTarget.drawImage(originalCanvas, 0, 0, targetWidth, targetHeight);

    resolve(targetCanvas.toDataURL('image/png'));
  });
};

export const saveMaskLocally = (maskBase64, filename) => {
  const link = document.createElement('a');
  link.href = maskBase64;
  link.download = filename || `sam_mask_${Date.now()}.png`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

export const generateBinaryMaskFile = async (maskBlob, targetWidth = 512, targetHeight = 512) => {
  return new Promise((resolve, reject) => {
    // Handle both File and Blob objects
    if (!maskBlob || (!(maskBlob instanceof Blob) && !(maskBlob instanceof File))) {
      reject(new Error('Invalid maskBlob: must be a Blob or File object'));
      return;
    }

    const img = new Image();
    img.onload = async () => {

      const canvas = document.createElement("canvas");
      canvas.width = targetWidth;
      canvas.height = targetHeight;

      const ctx = canvas.getContext("2d");

      // Fill with black, then draw resized mask
      ctx.fillStyle = "black";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.drawImage(img, 0, 0, targetWidth, targetHeight);

      const base64 = canvas.toDataURL("image/png");
      const res = await fetch(base64);
      const blob = await res.blob();

      resolve(new File([blob], "mask.png", { type: "image/png" }));
    };

    img.onerror = reject;
    img.src = URL.createObjectURL(maskBlob);
  });
};

