/**
 * Utility functions for crop tool calculations and transformations
 */

/**
 * Calculate bounding box of rotated rectangle
 */
export const rotatedBBox = (w, h, angleDeg) => {
  const a = (angleDeg * Math.PI) / 180;
  return {
    w: Math.abs(w * Math.cos(a)) + Math.abs(h * Math.sin(a)),
    h: Math.abs(w * Math.sin(a)) + Math.abs(h * Math.cos(a)),
  };
};

/**
 * Clamp movement within container bounds
 */
export const clampMovement = (x, y, scale, rotation, imgRef, containerRef, rotatedBBoxFn) => {
  const img = imgRef.current;
  const container = containerRef.current;
  if (!img || !container) return { x, y };

  const baseW = img.clientWidth;
  const baseH = img.clientHeight;
  const { w: bw, h: bh } = rotatedBBoxFn(
    baseW * scale,
    baseH * scale,
    rotation
  );

  const CW = container.clientWidth;
  const CH = container.clientHeight;

  const halfX = Math.max(0, (bw - CW) / 2);
  const halfY = Math.max(0, (bh - CH) / 2);

  return {
    x: Math.min(halfX, Math.max(-halfX, x)),
    y: Math.min(halfY, Math.max(-halfY, y)),
  };
};

/**
 * Get overlay piece rectangles for dimming effect
 */
export const getOverlayPieces = (cropBoxRef, containerRef) => {
  const crop = cropBoxRef.current;
  const container = containerRef.current;
  if (!crop || !container) return null;

  const cropRect = crop.getBoundingClientRect();
  const contRect = container.getBoundingClientRect();

  const left = cropRect.left - contRect.left;
  const top = cropRect.top - contRect.top;
  const width = cropRect.width;
  const height = cropRect.height;
  const Cw = container.clientWidth;
  const Ch = container.clientHeight;

  return {
    topRect: { left: 0, top: 0, width: Cw, height: Math.max(0, top) },
    bottomRect: {
      left: 0,
      top: Math.min(Ch, top + height),
      width: Cw,
      height: Math.max(0, Ch - (top + height)),
    },
    leftRect: {
      left: 0,
      top: Math.max(0, top),
      width: Math.max(0, left),
      height: Math.max(0, height),
    },
    rightRect: {
      left: Math.min(Cw, left + width),
      top: Math.max(0, top),
      width: Math.max(0, Cw - (left + width)),
      height: Math.max(0, height),
    },
  };
};

/**
 * Get bounding rectangles for crop box, image, and container
 */
export const getRects = (cropBoxRef, imgRef, containerRef) => {
  const crop = cropBoxRef.current;
  const img = imgRef.current;
  const container = containerRef.current;
  if (!crop || !img || !container) return null;

  const cropRect = crop.getBoundingClientRect();
  const imgRect = img.getBoundingClientRect();
  const contRect = container.getBoundingClientRect();

  return { cropRect, imgRect, contRect };
};

/**
 * Calculate image dimensions to fit container
 */
export const calculateImageFit = (img, maxW, maxH) => {
  const ratio = img.naturalWidth / img.naturalHeight;

  let w = maxW;
  let h = w / ratio;

  if (h > maxH) {
    h = maxH;
    w = h * ratio;
  }

  return { w, h };
};


/**
 * Check if crop box extends outside image bounds
 */
export const isOutsideImageBounds = (newL, newT, newW, newH, img, contRect) => {
  if (!img) return false;

  const imgR = img.getBoundingClientRect();

  const cLeft = newL + contRect.left;
  const cTop = newT + contRect.top;
  const cRight = cLeft + newW;
  const cBottom = cTop + newH;

  return (
    cLeft < imgR.left ||
    cTop < imgR.top ||
    cRight > imgR.right ||
    cBottom > imgR.bottom
  );
};

/**
 * Compute minimum scale required to cover crop area
 */
export const computeRequiredScale = (cropW, cropH, imgW, imgH) => {
  const requiredScaleW = cropW / imgW;
  const requiredScaleH = cropH / imgH;
  return Math.max(requiredScaleW, requiredScaleH);
};
