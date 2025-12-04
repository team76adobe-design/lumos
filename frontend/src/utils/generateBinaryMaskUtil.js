// utils/maskGenerator.js or inside your component

export const generateBinaryMask = (strokes, width, height) => {
  // 1. Create a detached canvas (invisible to user)
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  // 2. Fill background with BLACK (Pixel value 0)
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, width, height);

  // 3. Set brush to WHITE (Pixel value 255 / 1)
  ctx.strokeStyle = 'white';
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  // 4. Draw all stored strokes
  strokes.forEach((stroke) => {
    if (!stroke.points || stroke.points.length === 0) return;

    ctx.lineWidth = stroke.size; // Ensure your stroke object has a size property
    ctx.beginPath();
    
    // Start at first point
    ctx.moveTo(stroke.points[0].x, stroke.points[0].y);

    // Draw lines to subsequent points
    stroke.points.forEach((point) => {
      ctx.lineTo(point.x, point.y);
    });

    ctx.stroke();
  });

  // 5. Return the image data
  return canvas.toDataURL('image/png');
};