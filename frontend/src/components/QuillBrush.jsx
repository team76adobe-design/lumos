import React, { useRef, useEffect, useState, useImperativeHandle } from "react";

/**
 * QuillBrush Component
 * Handles brush-based drawing for Add (green) and Subtract (red) modes
 * Draws directly on the canvas with smooth touch/mouse tracking
 */
const QuillBrush = React.forwardRef(({
  imageSrc,
  quillMode, // 'add' | 'subtract' | 'color' | null
  onComplete, // callback when user completes drawing
  canvasWidth,
  canvasHeight,
  brushColor = "#ff0000",
  brushSize = 8
}, ref) => {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const isDrawingRef = useRef(false);
  
  // Store all drawn points with their type
  const [drawnPoints, setDrawnPoints] = useState([]);
  const currentStroke = useRef([]);

  // Expose methods to parent component via ref
  useImperativeHandle(ref, () => ({
    getDrawnPoints: () => {
      // Flatten all strokes into a single array with type info
      const allPoints = [];
      drawnPoints.forEach(stroke => {
        stroke.points.forEach(point => {
          allPoints.push({
            x: Math.round(point.x),
            y: Math.round(point.y),
            type: stroke.type
          });
        });
      });
      return allPoints;
    },
    clearDrawing: () => {
      setDrawnPoints([]);
    }
  }), [drawnPoints]);

  // Get brush color based on mode
  const getBrushColor = () => {
    if (quillMode === 'add') return 'rgba(0, 255, 0, 0.5)'; // Green semi-transparent
    if (quillMode === 'subtract') return 'rgba(255, 0, 0, 0.5)'; // Red semi-transparent
    if (quillMode === 'color') return brushColor;
    return 'rgba(128, 128, 128, 0.5)'; // Gray fallback
  };

  const getBrushType = () => {
    if (quillMode === 'add') return 1;
    if (quillMode === 'subtract') return 0;
    if (quillMode === 'color') return 2;
    return 0;
  };

  // Load image
  useEffect(() => {
    if (!imageSrc) return;
    
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = imageSrc;
    
    img.onload = () => {
      imageRef.current = img;
      const canvas = canvasRef.current;
      if (!canvas) return;

      canvas.width = img.naturalWidth || img.width;
      canvas.height = img.naturalHeight || img.height;
      
      redrawCanvas();
    };
  }, [imageSrc]);

  // Redraw canvas whenever points change
  useEffect(() => {
    redrawCanvas();
  }, [drawnPoints, quillMode]);

  const redrawCanvas = () => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext('2d');
    
    // Clear only (don't draw image - it's behind us)
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw all strokes
    drawnPoints.forEach(stroke => {
      if (stroke.points.length < 2) return;

      // Display Logic
      if (quillMode === 'color') {
          if (stroke.type !== 2) return;
      } else if (quillMode === 'add' || quillMode === 'subtract') {
          if (stroke.type === 2) return;
      }

      if (stroke.type === 2) {
          ctx.strokeStyle = stroke.color || brushColor;
      } else {
          ctx.strokeStyle = stroke.type === 1 
            ? 'rgba(0, 255, 0, 0.5)' 
            : 'rgba(255, 0, 0, 0.5)';
      }
      
      ctx.lineWidth = stroke.width || brushSize;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      ctx.beginPath();
      ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
      
      for (let i = 1; i < stroke.points.length; i++) {
        const cp = stroke.points[i - 1];
        const ep = stroke.points[i];
        ctx.quadraticCurveTo(cp.x, cp.y, (cp.x + ep.x) / 2, (cp.y + ep.y) / 2);
      }
      
      ctx.stroke();
    });
  };

  const getCanvasCoordinates = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();
    const clientX = e.clientX || (e.touches && e.touches[0]?.clientX) || 0;
    const clientY = e.clientY || (e.touches && e.touches[0]?.clientY) || 0;

    return {
      x: (clientX - rect.left) * (canvas.width / rect.width),
      y: (clientY - rect.top) * (canvas.height / rect.height)
    };
  };

  const handlePointerDown = (e) => {
    if (!quillMode) return;
    
    e.preventDefault();
    e.target.setPointerCapture(e.pointerId);
    
    isDrawingRef.current = true;
    const coords = getCanvasCoordinates(e);
    currentStroke.current = [coords];
    
    // Draw immediately
    drawStroke([coords]);
  };

  const handlePointerMove = (e) => {
    if (!isDrawingRef.current || !quillMode) return;
    
    e.preventDefault();
    
    const coords = getCanvasCoordinates(e);
    const lastPt = currentStroke.current[currentStroke.current.length - 1];
    
    // Distance check to smooth out points
    const dist = Math.hypot(coords.x - lastPt.x, coords.y - lastPt.y);
    if (dist >= 2) {
      currentStroke.current.push(coords);
      drawStroke(currentStroke.current);
    }
  };

  const handlePointerUp = (e) => {
    if (!isDrawingRef.current) return;
    
    e.target.releasePointerCapture(e.pointerId);
    isDrawingRef.current = false;
    
    if (currentStroke.current.length > 0) {
      // Store the completed stroke
      const strokeType = getBrushType();
      const newStroke = {
        points: [...currentStroke.current],
        type: strokeType,
        color: strokeType === 2 ? brushColor : null,
        width: brushSize
      };
      
      setDrawnPoints(prev => [...prev, newStroke]);
      currentStroke.current = [];
    }
  };

  const drawStroke = (points) => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img || points.length < 1) return;

    const ctx = canvas.getContext('2d');
    
    // Clear everything (no image redraw - it's behind us)
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw all previous strokes
    drawnPoints.forEach(stroke => {
      if (stroke.points.length < 2) return;

      // Display Logic
      if (quillMode === 'color') {
          if (stroke.type !== 2) return;
      } else if (quillMode === 'add' || quillMode === 'subtract') {
          if (stroke.type === 2) return;
      }

      if (stroke.type === 2) {
          ctx.strokeStyle = stroke.color || brushColor;
      } else {
          ctx.strokeStyle = stroke.type === 1 
            ? 'rgba(0, 255, 0, 0.5)' 
            : 'rgba(255, 0, 0, 0.5)';
      }
      
      ctx.lineWidth = stroke.width || brushSize;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      ctx.beginPath();
      ctx.moveTo(stroke.points[0].x, stroke.points[0].y);
      
      for (let i = 1; i < stroke.points.length; i++) {
        const cp = stroke.points[i - 1];
        const ep = stroke.points[i];
        ctx.quadraticCurveTo(cp.x, cp.y, (cp.x + ep.x) / 2, (cp.y + ep.y) / 2);
      }
      
      ctx.stroke();
    });

    // Draw current stroke
    if (points.length >= 2) {
      ctx.strokeStyle = getBrushColor();
      ctx.lineWidth = brushSize;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      
      for (let i = 1; i < points.length; i++) {
        const cp = points[i - 1];
        const ep = points[i];
        ctx.quadraticCurveTo(cp.x, cp.y, (cp.x + ep.x) / 2, (cp.y + ep.y) / 2);
      }
      
      ctx.stroke();
    }
  };

  if (!imageSrc) return null;

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <canvas
        ref={canvasRef}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
        style={{
          width: "100%",
          height: "auto",
          display: "block",
          touchAction: "none",
          cursor: quillMode ? 'crosshair' : 'default'
        }}
      />
    </div>
  );
});

export default QuillBrush;
