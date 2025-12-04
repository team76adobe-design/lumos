import { useRef, useCallback, useEffect } from 'react';

/**
 * useBrushTool - Gesture-based brush drawing for image editing
 * 
 * Features:
 * - Touch and mouse support
 * - Smooth stroke tracking with coordinate normalization
 * - Canvas coordinate conversion
 * - Stroke storage in polygons array
 * - Easy enable/disable based on menu state
 */
export const useBrushTool = (
  canvasRef,
  onStrokeComplete,
  onStrokeUpdate = null,
  brushOptions = {}
) => {
  const strokeRef = useRef(null);
  const isDrawingRef = useRef(false);
  const brushEnabledRef = useRef(false);
  const cleanupFnsRef = useRef([]);

  // Default brush options
  const {
    strokeWidth = 3,
    strokeColor = '#ffffffff',
    minDistance = 2, // Minimum pixel distance between points to avoid noise
  } = brushOptions;

  /**
   * Get normalized canvas coordinates from mouse/touch event
   * Handles both canvas coordinate systems and display scaling
   */
  const getCanvasCoords = useCallback((e) => {
    if (!canvasRef.current) return null;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    // Handle both mouse and touch events
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;

    // Get display coordinates (CSS pixels)
    const displayX = clientX - rect.left;
    const displayY = clientY - rect.top;

    // Scale to actual canvas coordinates
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const canvasX = displayX * scaleX;
    const canvasY = displayY * scaleY;

    return { 
      x: canvasX, 
      y: canvasY,
      displayX,
      displayY,
      timestamp: Date.now()
    };
  }, [canvasRef]);

  /**
   * Calculate distance between two points (for smoothing)
   */
  const getDistance = useCallback((p1, p2) => {
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    return Math.sqrt(dx * dx + dy * dy);
  }, []);

  /**
   * Initialize a new stroke
   */
  const startStroke = useCallback((coords) => {
    if (!brushEnabledRef.current || !coords) return;

    strokeRef.current = {
      points: [coords],
      startTime: Date.now(),
      color: strokeColor,
      width: strokeWidth,
    };

    isDrawingRef.current = true;
  }, [strokeColor, strokeWidth]);

  /**
   * Add point to current stroke with smoothing
   */
  const moveStroke = useCallback((coords) => {
    if (!isDrawingRef.current || !strokeRef.current || !coords) return;

    const stroke = strokeRef.current;
    const lastPoint = stroke.points[stroke.points.length - 1];

    // Only add point if it's far enough from the last one (reduces noise)
    if (getDistance(lastPoint, coords) >= minDistance) {
      stroke.points.push(coords);

      // Call onStrokeUpdate if provided (for real-time rendering)
      if (onStrokeUpdate) {
        onStrokeUpdate(stroke);
      }
    }
  }, [getDistance, minDistance, onStrokeUpdate]);

  /**
   * Complete the stroke and return it
   */
  const endStroke = useCallback(() => {
    if (!isDrawingRef.current || !strokeRef.current) return null;

    isDrawingRef.current = false;
    const completedStroke = strokeRef.current;
    strokeRef.current = null;

    if (completedStroke.points.length < 2) {
      // Stroke too short, ignore
      return null;
    }

    // Call completion handler
    if (onStrokeComplete) {
      onStrokeComplete(completedStroke);
    }

    return completedStroke;
  }, [onStrokeComplete]);

  /**
   * Draw stroke on canvas (for preview/real-time feedback)
   */
  const drawStroke = useCallback((ctx, stroke) => {
    if (!stroke || stroke.points.length < 2) return;

    const points = stroke.points;
    ctx.strokeStyle = stroke.color || strokeColor;
    ctx.lineWidth = stroke.width || strokeWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);

    // Use quadratic curves for smooth strokes
    for (let i = 1; i < points.length; i++) {
      const cp = points[i - 1]; // Control point
      const ep = points[i]; // End point
      ctx.quadraticCurveTo(cp.x, cp.y, (cp.x + ep.x) / 2, (cp.y + ep.y) / 2);
    }

    ctx.stroke();
  }, [strokeColor, strokeWidth]);

  /**
   * Enable brush mode - attach event listeners
   */
  const enableBrushMode = useCallback(() => {
    brushEnabledRef.current = true;

    if (!canvasRef.current) return;

    const canvas = canvasRef.current;

    // Mouse events
    const handleMouseDown = (e) => {
      const coords = getCanvasCoords(e);
      if (coords) startStroke(coords);
    };

    const handleMouseMove = (e) => {
      const coords = getCanvasCoords(e);
      if (coords) moveStroke(coords);
    };

    const handleMouseUp = () => {
      endStroke();
    };

    // Touch events
    const handleTouchStart = (e) => {
      e.preventDefault(); // Prevent scrolling while drawing
      const coords = getCanvasCoords(e);
      if (coords) startStroke(coords);
    };

    const handleTouchMove = (e) => {
      e.preventDefault();
      const coords = getCanvasCoords(e);
      if (coords) moveStroke(coords);
    };

    const handleTouchEnd = (e) => {
      e.preventDefault();
      endStroke();
    };

    // Attach listeners
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseup', handleMouseUp);
    canvas.addEventListener('mouseleave', handleMouseUp);

    canvas.addEventListener('touchstart', handleTouchStart);
    canvas.addEventListener('touchmove', handleTouchMove);
    canvas.addEventListener('touchend', handleTouchEnd);
    canvas.addEventListener('touchcancel', handleTouchEnd);

    // Store cleanup functions
    cleanupFnsRef.current = [
      () => canvas.removeEventListener('mousedown', handleMouseDown),
      () => canvas.removeEventListener('mousemove', handleMouseMove),
      () => canvas.removeEventListener('mouseup', handleMouseUp),
      () => canvas.removeEventListener('mouseleave', handleMouseUp),
      () => canvas.removeEventListener('touchstart', handleTouchStart),
      () => canvas.removeEventListener('touchmove', handleTouchMove),
      () => canvas.removeEventListener('touchend', handleTouchEnd),
      () => canvas.removeEventListener('touchcancel', handleTouchEnd),
    ];
  }, [canvasRef, getCanvasCoords, startStroke, moveStroke, endStroke]);

  /**
   * Disable brush mode - remove event listeners
   */
  const disableBrushMode = useCallback(() => {
    brushEnabledRef.current = false;
    
    // End any active stroke
    if (isDrawingRef.current) {
      endStroke();
    }

    // Remove all event listeners
    cleanupFnsRef.current.forEach(fn => fn());
    cleanupFnsRef.current = [];
  }, [endStroke]);

  /**
   * Get current stroke state (for debugging)
   */
  const getCurrentStroke = useCallback(() => {
    return strokeRef.current;
  }, []);

  /**
   * Cancel current stroke without calling onStrokeComplete
   */
  const cancelStroke = useCallback(() => {
    isDrawingRef.current = false;
    strokeRef.current = null;
  }, []);

  return {
    // Control
    enableBrushMode,
    disableBrushMode,
    
    // Stroke operations
    startStroke,
    moveStroke,
    endStroke,
    cancelStroke,
    
    // Utilities
    getCanvasCoords,
    drawStroke,
    getCurrentStroke,
    
    // State
    isDrawing: () => isDrawingRef.current,
    isBrushEnabled: () => brushEnabledRef.current,
  };
};

export default useBrushTool;
