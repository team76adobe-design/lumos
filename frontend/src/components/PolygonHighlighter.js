import React, { useRef, useEffect, useCallback, useState } from "react";
import { useImageEditorProcessor } from "../hooks/useImageEdit";

// -------------------------------------------------------------------------
// 1. Inlined Hooks (Required for standalone execution)
// -------------------------------------------------------------------------

// useBrushTool removed as per requirements

// -------------------------------------------------------------------------
// 2. Main Component
// -------------------------------------------------------------------------

const buildPath = (ctx, pts) => {
    ctx.beginPath();
    pts.forEach((p, i) => i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
    ctx.closePath();
};

const getScaledPolys = (canvas, img, polys) => {
    if (!canvas || !img) return [];
    const rx = canvas.width / (img.naturalWidth || img.width);
    const ry = canvas.height / (img.naturalHeight || img.height);
    return polys.map(p => ({
        ...p,
        points: p.points.map(pt => ({ x: pt.x * rx, y: pt.y * ry }))
    }));
};

const PolygonHighlighter = ({
    baseSrc,
    polygons,
    fgPoints,
    bgPoints,
    onPointClick,
    onPointRemove,
    clickMode,
    editValues = {},
    spotlightPolygons = false,
    activeMenu,
    activeTool,
    maskSelectionMode,
    onBrushStroke = null,
    setPolygons = null,
    quillMode = null,
    quillPoints = [],
    setQuillPoints = null,
    setMaskPath = null,
    maskPath = null,
    isBrushActive,
    brushSize = 20,
    penColor = "#ff0000",
    onDragEnd,
    onImageProcessed,
}) => {
    const staticCanvasRef = useRef(null);
    const animCanvasRef = useRef(null);
    const containerRef = useRef(null);
    const maskCanvasRef = useRef(null);
    const scratchCanvasRef = useRef(null);
    const baseImageRef = useRef(null);
    const animFrameRef = useRef(null);

    // Gesture State
    const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
    const activePointers = useRef(new Map());
    const prevGestureState = useRef(null);
    const isGesturing = useRef(false);
    const gestureHappened = useRef(false);

    // Gesture State
    const isDraggingRef = useRef(false);
    const lastDragPointTime = useRef(0);
    const dragStartPosition = useRef({ x: 0, y: 0 });
    const dragStartOffset = useRef({ x: 0, y: 0 });

    // Floating Selection State (Drag & Move)
    const floatingSelection = useRef(null); // { imageData, bbox: {x,y,w,h}, offset: {x,y}, maskCanvas }

    // Quill drawing state
    const currentQuillStroke = useRef([]);
    const isDrawingQuill = useRef(false);

    // Brush State
    const lastBrushPos = useRef(null);
    const currentCursorPos = useRef(null);
    const { outputUrl: processedUrl } = useImageEditorProcessor(editValues, baseSrc);

    useEffect(() => {
        if (onImageProcessed) {
            onImageProcessed(processedUrl || baseSrc);
        }
    }, [processedUrl, baseSrc, onImageProcessed]);

    const GEMINI_BRUSH_SIZE = 40;
    const isRasterActive = isBrushActive || activeMenu === 'star';
    const currentBrushSize = activeMenu === 'star' ? GEMINI_BRUSH_SIZE : brushSize;

    // Helper: Calculate Coordinates
    const getCanvasCoordinates = (e) => {
        const canvas = staticCanvasRef.current;
        if (!canvas) return { xCanvas: 0, yCanvas: 0, xDisplay: 0, yDisplay: 0 };

        const rect = canvas.getBoundingClientRect();
        const xClient = e.clientX - rect.left;
        const yClient = e.clientY - rect.top;

        // Canvas internal resolution
        const canvasW = canvas.width;
        const canvasH = canvas.height;

        // Element size on screen
        const rectW = rect.width;
        const rectH = rect.height;

        // Calculate the displayed size of the image within the element (object-fit: contain)
        const scaleX = rectW / canvasW;
        const scaleY = rectH / canvasH;
        const scale = Math.min(scaleX, scaleY);

        const displayedW = canvasW * scale;
        const displayedH = canvasH * scale;

        // Calculate offsets (centering)
        const offsetX = (rectW - displayedW) / 2;
        const offsetY = (rectH - displayedH) / 2;

        // Map client coordinates to canvas coordinates
        const xRel = xClient - offsetX;
        const yRel = yClient - offsetY;

        const xCanvas = xRel / scale;
        const yCanvas = yRel / scale;

        return { xCanvas, yCanvas, xDisplay: xClient, yDisplay: yClient };
    };

    // Brush Handlers removed as per requirements

    // Draw Static Layer
    const drawStatic = async () => {
        const canvas = staticCanvasRef.current;
        const img = baseImageRef.current;
        if (!canvas || !img) return;

        const ctx = canvas.getContext("2d");

        let imageToDraw = img;
        if (processedUrl) {
            try {
                await new Promise((resolve, reject) => {
                    const pimg = new Image();
                    pimg.crossOrigin = "anonymous";
                    pimg.onload = () => { imageToDraw = pimg; resolve(); };
                    pimg.onerror = reject;
                    pimg.src = processedUrl;
                });
            } catch (e) { imageToDraw = img; }
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(imageToDraw, 0, 0, canvas.width, canvas.height);

        const scaledPolys = getScaledPolys(canvas, img, polygons);

        // Use raster mask if available, otherwise polygons
        // Show spotlight ONLY when: 1) explicitly requested (addSubtract/second menu), or 2) star menu is active
        const shouldShowSpotlight = spotlightPolygons || activeMenu === 'star';
        
        if (shouldShowSpotlight && !isRasterActive) {
            let maskData = null;

            if (maskCanvasRef.current) {
                const mctx = maskCanvasRef.current.getContext('2d');
                maskData = mctx.getImageData(0, 0, maskCanvasRef.current.width, maskCanvasRef.current.height).data;
            } else if (scaledPolys.length > 0) {
                const mask = document.createElement("canvas");
                mask.width = canvas.width;
                mask.height = canvas.height;
                const mctx = mask.getContext("2d");

                mctx.fillStyle = "black";
                mctx.fillRect(0, 0, mask.width, mask.height);
                mctx.fillStyle = "white";

                scaledPolys.forEach(p => {
                    buildPath(mctx, p.points);
                    mctx.fill();
                });

                maskData = mctx.getImageData(0, 0, mask.width, mask.height).data;
            }

            if (maskData) {
                const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                const d = imgData.data;

                for (let i = 0; i < d.length; i += 4) {
                    // Check if pixel is part of mask
                    // If raster mask (transparent bg, white fg): alpha > 0
                    // If polygon mask (black bg, white fg): red > 128
                    const r = maskData[i];
                    const a = maskData[i + 3];
                    const isMask = a > 50 && r > 100;

                    if (isMask) {
                        d[i] = Math.min(255, d[i] * 1.3);
                        d[i + 1] = Math.min(255, d[i + 1] * 1.3);
                        d[i + 2] = Math.min(255, d[i + 2] * 1.3);
                    } else {
                        d[i] *= 0.5;
                        d[i + 1] *= 0.5;
                        d[i + 2] *= 0.5;
                    }
                }
                ctx.putImageData(imgData, 0, 0);
            }
        }

        // Draw quill brush strokes
        if (quillPoints.length > 0) {
            const imgW = img.naturalWidth || img.width;
            const imgH = img.naturalHeight || img.height;
            const sx = canvas.width / imgW;
            const sy = canvas.height / imgH;

            quillPoints.forEach(stroke => {
                // Display Logic:
                // 1. If Color Quill is active, ONLY show Color strokes (type 2)
                // 2. If Add/Subtract Quill is active, ONLY show Add/Subtract strokes (type 0 or 1)
                // 3. If no specific quill mode (but points exist), show ALL (overlay all)

                if (quillMode === 'color') {
                    if (stroke.type !== 2) return;
                } else if (quillMode === 'add' || quillMode === 'subtract') {
                    if (stroke.type === 2) return;
                }

                if (stroke.points && stroke.points.length > 1) {
                    if (stroke.type === 2) {
                        ctx.strokeStyle = stroke.color || penColor;
                    } else {
                        ctx.strokeStyle = stroke.type === 1
                            ? 'rgba(0, 255, 0, 0.5)'
                            : 'rgba(255, 0, 0, 0.5)';
                    }

                    ctx.lineWidth = stroke.width || brushSize;
                    ctx.lineCap = 'round';
                    ctx.lineJoin = 'round';

                    ctx.beginPath();
                    ctx.moveTo(stroke.points[0].x * sx, stroke.points[0].y * sy);

                    for (let i = 1; i < stroke.points.length; i++) {
                        const cp = stroke.points[i - 1];
                        const ep = stroke.points[i];
                        ctx.quadraticCurveTo(
                            cp.x * sx, cp.y * sy,
                            ((cp.x + ep.x) / 2) * sx, ((cp.y + ep.y) / 2) * sy
                        );
                    }

                    ctx.stroke();
                }
            });
        }

        // Draw current quill stroke
        if (isDrawingQuill.current && currentQuillStroke.current.length > 1) {
            if (quillMode === 'color') {
                ctx.strokeStyle = penColor;
            } else {
                ctx.strokeStyle = quillMode === 'add'
                    ? 'rgba(0, 255, 0, 0.5)'
                    : 'rgba(255, 0, 0, 0.5)';
            }
            ctx.lineWidth = brushSize;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';

            ctx.beginPath();
            ctx.moveTo(currentQuillStroke.current[0].x, currentQuillStroke.current[0].y);
            for (let i = 1; i < currentQuillStroke.current.length; i++) {
                const cp = currentQuillStroke.current[i - 1];
                const ep = currentQuillStroke.current[i];
                ctx.quadraticCurveTo(cp.x, cp.y, (cp.x + ep.x) / 2, (cp.y + ep.y) / 2);
            }
            ctx.stroke();
        }

        // Draw Points (only in addSubtract menu)
        if (activeMenu === 'addSubtract') {
            const imgW = img.naturalWidth || img.width;
            const imgH = img.naturalHeight || img.height;
            const sx = canvas.width / imgW;
            const sy = canvas.height / imgH;

            [...fgPoints, ...bgPoints].forEach((p, i) => {
                const fg = i < fgPoints.length;
                ctx.fillStyle = fg ? "lime" : "red";
                ctx.font = "bold 32px Arial";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(fg ? "+" : "-", p.imageX * sx, p.imageY * sy);
            });
        }
    };

    // Load maskPath into maskCanvasRef
    useEffect(() => {
        if (maskPath && staticCanvasRef.current) {
            const img = new Image();
            img.crossOrigin = "anonymous";
            img.onload = () => {
                if (!maskCanvasRef.current) {
                    maskCanvasRef.current = document.createElement('canvas');
                }
                // Resize if needed
                if (maskCanvasRef.current.width !== staticCanvasRef.current.width || maskCanvasRef.current.height !== staticCanvasRef.current.height) {
                    maskCanvasRef.current.width = staticCanvasRef.current.width;
                    maskCanvasRef.current.height = staticCanvasRef.current.height;
                }

                const ctx = maskCanvasRef.current.getContext('2d');

                // Reset composite operation to ensure we draw the image correctly
                ctx.globalCompositeOperation = 'source-over';

                ctx.clearRect(0, 0, maskCanvasRef.current.width, maskCanvasRef.current.height);
                ctx.drawImage(img, 0, 0, maskCanvasRef.current.width, maskCanvasRef.current.height);

                // Convert Black Background to Transparent for correct compositing
                const imageData = ctx.getImageData(0, 0, maskCanvasRef.current.width, maskCanvasRef.current.height);
                const data = imageData.data;
                for (let i = 0; i < data.length; i += 4) {
                    const r = data[i];
                    const g = data[i + 1];
                    const b = data[i + 2];
                    // If pixel is black (background), make it transparent
                    if (r < 20 && g < 20 && b < 20) {
                        data[i + 3] = 0;
                    }
                }
                ctx.putImageData(imageData, 0, 0);

                drawStatic();
            };
            img.src = URL.createObjectURL(maskPath);
        }
    }, [maskPath]);

    // lmfao
    // Animation Loop
    const startAnimation = () => {
        const canvas = animCanvasRef.current;
        const img = baseImageRef.current;
        if (!canvas || !img) return;

        const ctx = canvas.getContext("2d");
        const scaledPolys = getScaledPolys(canvas, img, polygons);
        // console.log("here");
        const animate = (t) => {
            if (!isRasterActive) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }

            // Only animate existing shapes if NOT drawing a new line
            // (This prevents flickering while drawing)
            const shouldAnimate = (activeMenu === "addSubtract" || activeTool === "move" || activeTool === "drag" || activeMenu === 'star') && !isRasterActive;
            // Show polygons when: 1) normal animation conditions, 2) star menu with segmentation results, 3) second menu, or 4) addSubtract after segmentation
            const shouldShowPolygons = scaledPolys.length > 0 && (
                shouldAnimate || 
                (activeMenu === 'star' && !isBrushActive) || 
                activeMenu === 'second' ||
                (activeMenu === 'addSubtract' && !isBrushActive)
            );
            
            // --- DRAG TOOL RENDERING (Vector Arrow) ---
            if (activeTool === 'drag' && isDraggingRef.current && dragStartPosition.current && currentCursorPos.current) {
                const start = dragStartPosition.current; // Canvas coords
                const end = currentCursorPos.current; // Canvas coords

                // console.log('🎨 Rendering Arrow:', {
                //     start,
                //     end,
                //     canvasSize: { w: canvas.width, h: canvas.height },
                //     isDragging: isDraggingRef.current
                // });
                

                // Dynamic sizing based on canvas width
                const lineWidth = Math.max(3, canvas.width / 200);
                const headLen = Math.max(20, canvas.width / 40);

                ctx.save();

                // Clear the area first
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
                ctx.shadowBlur = 15;
                ctx.globalCompositeOperation = 'source-over';
                ctx.setTransform(1, 0, 0, 1, 0, 0); // Reset transform

                // Draw Arrow Line
                ctx.beginPath();
                ctx.moveTo(start.x, start.y);
                ctx.lineTo(end.x, end.y);
                ctx.strokeStyle = '#00FFFF'; // Cyan
                ctx.lineWidth = lineWidth;
                ctx.lineCap = 'round';
                ctx.stroke();

                // Draw Arrowhead - shifted back slightly along the line to prevent overlap
                const angle = Math.atan2(end.y - start.y, end.x - start.x);
                const arrowOffset = headLen * 0.3; // Shift arrowhead back
                const arrowTipX = end.x + arrowOffset * Math.cos(angle);
                const arrowTipY = end.y + arrowOffset * Math.sin(angle);
                
                ctx.beginPath();
                ctx.moveTo(arrowTipX, arrowTipY);
                ctx.lineTo(arrowTipX - headLen * Math.cos(angle - Math.PI / 6), arrowTipY - headLen * Math.sin(angle - Math.PI / 6));
                ctx.lineTo(arrowTipX - headLen * Math.cos(angle + Math.PI / 6), arrowTipY - headLen * Math.sin(angle + Math.PI / 6));
                ctx.lineTo(arrowTipX, arrowTipY);
                ctx.fillStyle = '#00FFFF';
                ctx.fill();

                // Draw Start Dot - smaller
                ctx.beginPath();
                ctx.arc(start.x, start.y, lineWidth * 2, 0, Math.PI * 2);
                ctx.fillStyle = '#00FFFF';
                ctx.fill();

                ctx.restore();
            }
            // --- MOVE TOOL RENDERING (Floating Selection) ---
            else if (shouldAnimate && activeTool === 'move' && floatingSelection.current && activeMenu === 'second') {
                const { bbox, offset, maskCanvas, imageData } = floatingSelection.current;

                // Layer 2: Dark Overlay
                ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                // Draw Grey Shape at Source Position (to hide original)
                ctx.save();
                const scaleX = canvas.width / maskCanvas.width;
                const scaleY = canvas.height / maskCanvas.height;

                ctx.scale(scaleX, scaleY);
                ctx.drawImage(maskCanvas, 0, 0); // maskCanvas is now Grey/Transparent
                ctx.restore();

                // Layer 3: Floating Selected Pixels
                ctx.save();

                // Create a temp canvas for the floating pixels to handle scaling easily
                if (!scratchCanvasRef.current) {
                    scratchCanvasRef.current = document.createElement('canvas');
                }
                const sCanvas = scratchCanvasRef.current;
                if (sCanvas.width !== bbox.w || sCanvas.height !== bbox.h) {
                    sCanvas.width = bbox.w;
                    sCanvas.height = bbox.h;
                }
                const sCtx = sCanvas.getContext('2d');
                sCtx.putImageData(imageData, 0, 0);

                // Draw at new position
                const drawX = (bbox.x + offset.x) * scaleX;
                const drawY = (bbox.y + offset.y) * scaleY;
                const drawW = bbox.w * scaleX;
                const drawH = bbox.h * scaleY;

                ctx.drawImage(sCanvas, drawX, drawY, drawW, drawH);

                ctx.restore();

            } else if (shouldShowPolygons) {
                // Standard Polygon Animation - show segmentation results
                // scaledPolys.forEach((poly) => {
                //     ctx.save();
                //     buildPath(ctx, poly.points);
                //     ctx.strokeStyle = '#FF5959'; // Solid Red
                //     ctx.lineWidth = 4;
                //     ctx.stroke();
                //     ctx.restore();
                // });
            }

            // Render Raster Mask if Brush Active
            if (isRasterActive) {
                // Check if static canvas is available
                if (!staticCanvasRef.current) {
                    animFrameRef.current = requestAnimationFrame(animate);
                    return;
                }

                // OPTIMIZED RENDERING: "Punch Hole" Technique
                // Instead of complex compositing (which requires multiple full-frame draws),
                // we simply draw a dark overlay and "erase" the masked part to reveal the static image below.

                // 1. Clear Animation Canvas
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                // 2. Draw Dark Overlay (Background)
                ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                if (maskCanvasRef.current) {
                    // 3. "Punch Hole" using the Mask
                    // destination-out removes pixels from the destination (Dark Overlay) 
                    // based on the alpha of the source (Mask).
                    // Where Mask is opaque, Overlay becomes transparent -> Reveals Static Image.
                    ctx.globalCompositeOperation = 'destination-out';
                    ctx.drawImage(maskCanvasRef.current, 0, 0, canvas.width, canvas.height);

                    // Reset Composite Operation for subsequent draws (like the cursor)
                    ctx.globalCompositeOperation = 'source-over';
                }

                // Draw Brush Cursor
                if (currentCursorPos.current) {
                    ctx.save();
                    ctx.beginPath();
                    ctx.arc(currentCursorPos.current.x, currentCursorPos.current.y, currentBrushSize / 2, 0, Math.PI * 2);
                    ctx.strokeStyle = 'white';
                    ctx.lineWidth = 2;
                    ctx.shadowColor = 'black';
                    ctx.shadowBlur = 4;
                    ctx.stroke();
                    ctx.restore();
                }
            }

            animFrameRef.current = requestAnimationFrame(animate);
        };
        animFrameRef.current = requestAnimationFrame(animate);
    };

    useEffect(() => {
        if (!baseSrc) return;
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.src = baseSrc;
        img.onload = () => {
            baseImageRef.current = img;
            const staticCanvas = staticCanvasRef.current;
            const animCanvas = animCanvasRef.current;
            if (!staticCanvas || !animCanvas) return;

            staticCanvas.width = img.naturalWidth || img.width;
            staticCanvas.height = img.naturalHeight || img.height;
            animCanvas.width = staticCanvas.width;
            animCanvas.height = staticCanvas.height;

            // Initialize Mask Canvas
            if (!maskCanvasRef.current) {
                maskCanvasRef.current = document.createElement('canvas');
            }
            maskCanvasRef.current.width = staticCanvas.width;
            maskCanvasRef.current.height = staticCanvas.height;
            const mctx = maskCanvasRef.current.getContext('2d');
            mctx.clearRect(0, 0, maskCanvasRef.current.width, maskCanvasRef.current.height);

            // Restore Mask Content if needed
            if (maskPath) {
                const maskImg = new Image();
                maskImg.crossOrigin = "anonymous";
                maskImg.onload = () => {
                    mctx.drawImage(maskImg, 0, 0, maskCanvasRef.current.width, maskCanvasRef.current.height);

                    // Handle transparency (black -> transparent)
                    const imageData = mctx.getImageData(0, 0, maskCanvasRef.current.width, maskCanvasRef.current.height);
                    const data = imageData.data;
                    for (let i = 0; i < data.length; i += 4) {
                        const r = data[i];
                        const g = data[i + 1];
                        const b = data[i + 2];
                        if (r < 20 && g < 20 && b < 20) {
                            data[i + 3] = 0;
                        }
                    }
                    mctx.putImageData(imageData, 0, 0);

                    drawStatic();
                };
                maskImg.src = URL.createObjectURL(maskPath);
            } else if (isRasterActive && polygons && polygons.length > 0) {
                const scaled = getScaledPolys(staticCanvas, img, polygons);
                mctx.fillStyle = 'white';
                scaled.forEach(p => {
                    buildPath(mctx, p.points);
                    mctx.fill();
                });
            }

            drawStatic();
            startAnimation();
        };
        return () => cancelAnimationFrame(animFrameRef.current);
    }, [baseSrc, activeMenu, isBrushActive, isRasterActive]);

    // Initialize mask canvas when raster brush is active
    useEffect(() => {
        if (isRasterActive) {
            // Ensure mask canvas is initialized if it wasn't
            if (!maskCanvasRef.current && staticCanvasRef.current) {
                maskCanvasRef.current = document.createElement('canvas');
                maskCanvasRef.current.width = staticCanvasRef.current.width;
                maskCanvasRef.current.height = staticCanvasRef.current.height;
                const mctx = maskCanvasRef.current.getContext('2d');
                mctx.clearRect(0, 0, maskCanvasRef.current.width, maskCanvasRef.current.height);
            }

            // Initialize mask with existing polygons if any
            // Only convert polygons to raster if we DON'T have a maskPath (which is likely higher quality)
            if (!maskPath && maskCanvasRef.current && polygons && polygons.length > 0 && baseImageRef.current && staticCanvasRef.current) {
                const ctx = maskCanvasRef.current.getContext('2d');
                // Only draw if the canvas is empty (or we assume we want to start from polygons)
                // For now, let's clear and redraw to be safe when entering brush mode
                ctx.clearRect(0, 0, maskCanvasRef.current.width, maskCanvasRef.current.height);

                const scaled = getScaledPolys(staticCanvasRef.current, baseImageRef.current, polygons);
                ctx.fillStyle = 'white';
                scaled.forEach(p => {
                    buildPath(ctx, p.points);
                    ctx.fill();
                });
            }
        }
    }, [isRasterActive, maskPath, polygons]);

    useEffect(() => {
        if (baseImageRef.current) drawStatic();
    }, [polygons, fgPoints, bgPoints, editValues, spotlightPolygons, processedUrl, activeMenu, quillPoints, isBrushActive]);

    // -------------------------------------------------------------------------
    // Unified Gesture Handlers
    // -------------------------------------------------------------------------

    const handleAddSubtractLogic = (coords, isDragOperation) => {
        // Throttling for drag
        const now = Date.now();
        if (isDragOperation && now - lastDragPointTime.current < 80) {
            return;
        }

        // Drag Logic: Only Add
        if (isDragOperation) {
            if (["add", "subtract", "select"].includes(clickMode)) {
                onPointClick(coords.xCanvas, coords.yCanvas, coords.xDisplay, coords.yDisplay);
                lastDragPointTime.current = now;
            }
            return;
        }

        // Tap Logic: Remove or Add
        const canvas = staticCanvasRef.current;
        const img = baseImageRef.current;
        if (!canvas || !img) return;

        const imgW = img.naturalWidth || img.width;
        const imgH = img.naturalHeight || img.height;

        const pts = [...fgPoints, ...bgPoints];
        for (let i = 0; i < pts.length; i++) {
            const px = pts[i].imageX * (canvas.width / imgW);
            const py = pts[i].imageY * (canvas.height / imgH);

            if (Math.hypot(coords.xCanvas - px, coords.yCanvas - py) < 20) {
                onPointRemove(i < fgPoints.length ? "fg" : "bg", i);
                return;
            }
        }

        if (["add", "subtract", "select"].includes(clickMode)) {
            onPointClick(coords.xCanvas, coords.yCanvas, coords.xDisplay, coords.yDisplay);
        }
    };

    const handleSinglePointerDown = (e) => {
        // Important: Capture pointer so we keep tracking even if mouse leaves canvas
        e.target.setPointerCapture(e.pointerId);
        const coords = getCanvasCoordinates(e);
        isDraggingRef.current = false;
        dragStartPosition.current = { x: coords.xDisplay, y: coords.yDisplay };
        currentCursorPos.current = { x: coords.xCanvas, y: coords.yCanvas }; // Ensure cursor pos is set immediately

        // 1. Quill mode drawing
        if (quillMode && setQuillPoints) {
            isDrawingQuill.current = true;
            currentQuillStroke.current = [{ x: coords.xCanvas, y: coords.yCanvas }];
            return;
        }

        // 0. Raster Brush Start
        if (isRasterActive) {
            // Ensure mask canvas exists
            if (!maskCanvasRef.current && staticCanvasRef.current) {
                maskCanvasRef.current = document.createElement('canvas');
                maskCanvasRef.current.width = staticCanvasRef.current.width;
                maskCanvasRef.current.height = staticCanvasRef.current.height;
                const mctx = maskCanvasRef.current.getContext('2d');
                mctx.clearRect(0, 0, maskCanvasRef.current.width, maskCanvasRef.current.height);
            }

            if (maskCanvasRef.current) {
                lastBrushPos.current = { x: coords.xCanvas, y: coords.yCanvas };
                // Draw initial dot
                const ctx = maskCanvasRef.current.getContext('2d');
                ctx.beginPath();
                ctx.arc(coords.xCanvas, coords.yCanvas, currentBrushSize / 2, 0, Math.PI * 2);

                // Determine operation:
                // Gemini menu (star) is ALWAYS Add (source-over)
                // Add/Subtract menu respects maskSelectionMode
                const isSubtract = maskSelectionMode === 'subtract' && activeMenu !== 'star';
                ctx.globalCompositeOperation = isSubtract ? 'destination-out' : 'source-over';

                ctx.fillStyle = 'white';
                ctx.fill();
            }
            return;
        }

        // 3. Trigger Add/Subtract Start (only if add or subtract is selected)
        // This runs if brush is NOT active, OR if we are in addSubtract menu but brush is OFF
        if (activeMenu === 'addSubtract' && !isRasterActive && (maskSelectionMode === 'add' || maskSelectionMode === 'subtract')) {
            handleAddSubtractLogic(coords, false);
            lastDragPointTime.current = Date.now();
        }

        // 3. Trigger Move/Drag Click
        if (activeTool === "move" || activeTool === "drag") {
            // Drag Tool Logic (Vector Arrow)
            if (activeTool === 'drag') {
                isDraggingRef.current = true;
                dragStartPosition.current = { x: coords.xCanvas, y: coords.yCanvas };
                // console.log('🎯 Drag Start:', { isDragging: true, start: dragStartPosition.current, cursor: currentCursorPos.current });
                return;
            }

            // Move Tool Logic (Floating Selection)
            if (activeTool === 'move' && floatingSelection.current) {
                const { bbox, offset } = floatingSelection.current;
                const img = baseImageRef.current;
                const canvas = staticCanvasRef.current;

                // Convert click to image coordinates
                const scaleX = canvas.width / (img.naturalWidth || img.width);
                const scaleY = canvas.height / (img.naturalHeight || img.height);

                const clickX = coords.xCanvas / scaleX;
                const clickY = coords.yCanvas / scaleY;

                // Check if inside floating bbox
                const currentX = bbox.x + offset.x;
                const currentY = bbox.y + offset.y;

                if (clickX >= currentX && clickX <= currentX + bbox.w &&
                    clickY >= currentY && clickY <= currentY + bbox.h) {

                    isDraggingRef.current = true;
                    dragStartPosition.current = { x: clickX, y: clickY };
                    dragStartOffset.current = { ...floatingSelection.current.offset };
                    return; // Stop propagation
                }
            }

            onPointClick(coords.xCanvas, coords.yCanvas, coords.xDisplay, coords.yDisplay);
        }
    };

    const handleSinglePointerMove = (e) => {
        const coords = getCanvasCoordinates(e);
        currentCursorPos.current = { x: coords.xCanvas, y: coords.yCanvas };

        if (e.buttons === 0) return; // Only process if button is held down

        // 1. Quill mode drawing
        if (isDrawingQuill.current && quillMode && setQuillPoints) {
            const lastPt = currentQuillStroke.current[currentQuillStroke.current.length - 1];
            const dist = Math.hypot(coords.xCanvas - lastPt.x, coords.yCanvas - lastPt.y);
            if (dist >= 2) {
                currentQuillStroke.current.push({ x: coords.xCanvas, y: coords.yCanvas });
                drawStatic(); // Redraw to show current stroke
            }
            return;
        }

        // Drag Tool Logic (Vector Arrow)
        if (activeTool === 'drag' && isDraggingRef.current) {
            // Just update cursor pos, animation loop handles drawing
            
            // console.log('🖱️ Drag Move:', { start: dragStartPosition.current, cursor: currentCursorPos.current });
            startAnimation();
            return;
        }

        // Move Tool Logic (Floating Selection)
        if (activeTool === 'move' && isDraggingRef.current && floatingSelection.current) {
            const img = baseImageRef.current;
            const canvas = staticCanvasRef.current;
            const scaleX = canvas.width / (img.naturalWidth || img.width);
            const scaleY = canvas.height / (img.naturalHeight || img.height);

            const currentX = coords.xCanvas / scaleX;
            const currentY = coords.yCanvas / scaleY;

            const dx = currentX - dragStartPosition.current.x;
            const dy = currentY - dragStartPosition.current.y;

            floatingSelection.current.offset.x += dx;
            floatingSelection.current.offset.y += dy;

            dragStartPosition.current = { x: currentX, y: currentY };
            return; // Stop propagation
        }

        const dist = Math.hypot(
            coords.xDisplay - dragStartPosition.current.x,
            coords.yDisplay - dragStartPosition.current.y
        );

        if (dist > 5) {
            isDraggingRef.current = true;
        }

        // 0. Raster Brush Move
        if (isRasterActive && lastBrushPos.current) {
            if (!maskCanvasRef.current) return;
            const ctx = maskCanvasRef.current.getContext('2d');
            ctx.beginPath();

            // Determine operation:
            // Gemini menu (star) is ALWAYS Add (source-over)
            // Add/Subtract menu respects maskSelectionMode
            const isSubtract = maskSelectionMode === 'subtract' && activeMenu !== 'star';
            ctx.globalCompositeOperation = isSubtract ? 'destination-out' : 'source-over';

            ctx.strokeStyle = 'white';
            ctx.lineWidth = currentBrushSize;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.moveTo(lastBrushPos.current.x, lastBrushPos.current.y);
            ctx.lineTo(coords.xCanvas, coords.yCanvas);
            ctx.stroke();
            lastBrushPos.current = { x: coords.xCanvas, y: coords.yCanvas };

            // Force re-render of animation loop if needed, but loop is running
            return;
        }

        // 3. Trigger Add/Subtract Update (only if add or subtract is selected)
        if (isDraggingRef.current && activeMenu === 'addSubtract' && !isRasterActive && (maskSelectionMode === 'add' || maskSelectionMode === 'subtract')) {
            handleAddSubtractLogic(coords, true);
        }
    };

    const handleSinglePointerLeave = (e) => {
        currentCursorPos.current = null;
        handleSinglePointerUp(e);
    };

    const handleSinglePointerUp = (e) => {
        e.target.releasePointerCapture(e.pointerId);

        // 1. Quill mode drawing complete
        if (isDrawingQuill.current && quillMode && setQuillPoints && currentQuillStroke.current.length > 0) {
            const img = baseImageRef.current;
            const canvas = staticCanvasRef.current;
            if (img && canvas) {
                const imgW = img.naturalWidth || img.width;
                const imgH = img.naturalHeight || img.height;
                const sx = canvas.width / imgW;
                const sy = canvas.height / imgH;

                // Convert to image coordinates
                const imagePoints = currentQuillStroke.current.map(p => ({
                    x: p.x / sx,
                    y: p.y / sy
                }));

                let strokeType = 0;
                let strokeColor = null;

                if (quillMode === 'add') {
                    strokeType = 1;
                } else if (quillMode === 'subtract') {
                    strokeType = 0;
                } else if (quillMode === 'color') {
                    strokeType = 2;
                    strokeColor = penColor;
                }

                const newStroke = {
                    points: imagePoints,
                    type: strokeType,
                    color: strokeColor,
                    width: brushSize
                };

                setQuillPoints(prev => [...prev, newStroke]);
            }

            currentQuillStroke.current = [];
            isDrawingQuill.current = false;
            return;
        }

        // 0. Raster Brush End
        if (isRasterActive) {
            lastBrushPos.current = null;
            // Update mask path
            if (setMaskPath && maskCanvasRef.current) {
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = maskCanvasRef.current.width;
                tempCanvas.height = maskCanvasRef.current.height;
                const tctx = tempCanvas.getContext('2d');
                tctx.fillStyle = 'black';
                tctx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
                tctx.drawImage(maskCanvasRef.current, 0, 0);

                tempCanvas.toBlob((blob) => {
                    const file = new File([blob], "mask.png", { type: "image/png" });
                    setMaskPath(file);
                });
            }
            return;
        }

        // Drag Tool Logic - End
        if (activeTool === 'drag' && isDraggingRef.current) {
            if (onDragEnd && dragStartPosition.current && currentCursorPos.current) {
                // Both are already in Image/Canvas coordinates
                onDragEnd(dragStartPosition.current, currentCursorPos.current);
            }
            isDraggingRef.current = false;
            return;
        }

        // Move Tool Logic - End
        if (activeTool === 'move' && isDraggingRef.current && floatingSelection.current) {
            if (onDragEnd) {
                const { bbox, offset } = floatingSelection.current;
                const startOffset = dragStartOffset.current;

                const centerX = bbox.x + bbox.w / 2;
                const centerY = bbox.y + bbox.h / 2;

                const p1 = { x: centerX + startOffset.x, y: centerY + startOffset.y };
                const p2 = { x: centerX + offset.x, y: centerY + offset.y };

                onDragEnd(p1, p2);
            }
            isDraggingRef.current = false;
            return;
        }

        // Reset drag state
        isDraggingRef.current = false;
    };

    // --- GESTURE HANDLERS (Two-Finger Pan/Zoom) ---
    const handlePointerDown = (e) => {
        if (activePointers.current.size === 0) {
            gestureHappened.current = false;
        }
        e.target.setPointerCapture(e.pointerId);
        activePointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });

        if (activePointers.current.size === 2 && activeMenu !== 'addSubtract') {
            isGesturing.current = true;
            gestureHappened.current = true;
            prevGestureState.current = null;
            
            // Cancel single-finger actions
            isDraggingRef.current = false;
            isDrawingQuill.current = false;
            currentQuillStroke.current = [];
            drawStatic();
        }

        if (!isGesturing.current && !gestureHappened.current) {
            handleSinglePointerDown(e);
        }
    };

    const handlePointerMove = (e) => {
        activePointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });

        if (isGesturing.current && activePointers.current.size === 2) {
            const points = Array.from(activePointers.current.values());
            const p1 = points[0];
            const p2 = points[1];

            // Calculate center in client coordinates
            const centerClient = { x: (p1.x + p2.x) / 2, y: (p1.y + p2.y) / 2 };
            const dist = Math.hypot(p1.x - p2.x, p1.y - p2.y);

            // Convert to relative coordinates
            const container = containerRef.current;
            if (!container) return;
            const rect = container.getBoundingClientRect();
            const center = { x: centerClient.x - rect.left, y: centerClient.y - rect.top };

            if (!prevGestureState.current) {
                prevGestureState.current = { dist, center };
                return;
            }

            const prev = prevGestureState.current;
            const scaleFactor = dist / prev.dist;
            const dx = center.x - prev.center.x;
            const dy = center.y - prev.center.y;

            setTransform(t => {
                let newScale = t.scale * scaleFactor;
                if (newScale < 1) newScale = 1;

                const effectiveScaleFactor = newScale / t.scale;
                let newX = center.x * (1 - effectiveScaleFactor) + t.x * effectiveScaleFactor + dx;
                let newY = center.y * (1 - effectiveScaleFactor) + t.y * effectiveScaleFactor + dy;

                // Bounds Logic
                const w = container.offsetWidth;
                const h = container.offsetHeight;
                
                // Max bounds (cannot pan beyond edges)
                const minX = w * (1 - newScale);
                const maxX = 0;
                const minY = h * (1 - newScale);
                const maxY = 0;

                if (newX > maxX) newX = maxX;
                if (newX < minX) newX = minX;
                if (newY > maxY) newY = maxY;
                if (newY < minY) newY = minY;

                return { x: newX, y: newY, scale: newScale };
            });

            prevGestureState.current = { dist, center };
        } else if (!isGesturing.current && !gestureHappened.current) {
            handleSinglePointerMove(e);
        }
    };

    const handlePointerUp = (e) => {
        activePointers.current.delete(e.pointerId);
        e.target.releasePointerCapture(e.pointerId);

        if (activePointers.current.size < 2) {
            isGesturing.current = false;
            prevGestureState.current = null;
        }

        if (!isGesturing.current && !gestureHappened.current) {
            handleSinglePointerUp(e);
        }
    };

    const handlePointerLeave = (e) => {
        activePointers.current.delete(e.pointerId);
        if (!isGesturing.current && !gestureHappened.current) {
            handleSinglePointerLeave(e);
        }
    };

    // Initialize Floating Selection for Drag & Move
    useEffect(() => {
        if (activeTool === 'move' && maskPath && baseImageRef.current && staticCanvasRef.current) {
            const initFloatingSelection = async () => {
                const img = baseImageRef.current;
                const canvas = staticCanvasRef.current;

                // 1. Load Mask
                const maskImg = new Image();
                maskImg.crossOrigin = "anonymous";
                maskImg.src = URL.createObjectURL(maskPath);
                await new Promise(r => maskImg.onload = r);

                // 2. Create Mask Canvas & Get BBox
                const mCanvas = document.createElement('canvas');
                mCanvas.width = img.naturalWidth || img.width;
                mCanvas.height = img.naturalHeight || img.height;
                const mCtx = mCanvas.getContext('2d');
                mCtx.drawImage(maskImg, 0, 0, mCanvas.width, mCanvas.height);

                const mData = mCtx.getImageData(0, 0, mCanvas.width, mCanvas.height);
                const data = mData.data;

                let minX = mCanvas.width, minY = mCanvas.height, maxX = 0, maxY = 0;
                let hasPixels = false;

                for (let y = 0; y < mCanvas.height; y++) {
                    for (let x = 0; x < mCanvas.width; x++) {
                        const i = (y * mCanvas.width + x) * 4;
                        // Check for white pixels (mask)
                        if (data[i] > 128) { // Red channel > 128
                            if (x < minX) minX = x;
                            if (x > maxX) maxX = x;
                            if (y < minY) minY = y;
                            if (y > maxY) maxY = y;
                            hasPixels = true;
                        }
                    }
                }

                if (!hasPixels) return;

                const bbox = { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 };

                // 3. Extract Selected Pixels from Base Image
                // We need to draw the base image to a temp canvas to extract ImageData
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = mCanvas.width;
                tempCanvas.height = mCanvas.height;
                const tCtx = tempCanvas.getContext('2d');
                tCtx.drawImage(img, 0, 0, tempCanvas.width, tempCanvas.height);

                // Extract only the region within bbox
                const selectedPixels = tCtx.getImageData(bbox.x, bbox.y, bbox.w, bbox.h);

                // Apply mask alpha to selected pixels (optional, but good for irregular shapes)
                // We need to map the mask data to the selected pixels
                for (let y = 0; y < bbox.h; y++) {
                    for (let x = 0; x < bbox.w; x++) {
                        const maskIdx = ((bbox.y + y) * mCanvas.width + (bbox.x + x)) * 4;
                        const pixelIdx = (y * bbox.w + x) * 4;
                        // If mask is black, make pixel transparent
                        if (data[maskIdx] < 128) {
                            selectedPixels.data[pixelIdx + 3] = 0;
                        }
                    }
                }

                // 4. Convert Mask Canvas to "Grey Out" Overlay
                // We want the source position to look greyed out.
                // So we convert the white mask pixels to Grey, and black to Transparent.
                for (let i = 0; i < data.length; i += 4) {
                    if (data[i] > 100) { // Mask Area
                        data[i] = 50;      // R (Dark Grey)
                        data[i + 1] = 50;  // G
                        data[i + 2] = 50;  // B
                        data[i + 3] = 255; // A (Opaque)
                    } else { // Background
                        data[i + 3] = 0;   // Transparent
                    }
                }
                mCtx.putImageData(mData, 0, 0);

                // 5. Store State
                floatingSelection.current = {
                    imageData: selectedPixels,
                    bbox: bbox,
                    offset: { x: 0, y: 0 },
                    maskCanvas: mCanvas, // Now contains the Grey Shape
                    originalMaskImg: maskImg
                };

                // Force redraw
                startAnimation();
            };

            initFloatingSelection();
        } else if (activeTool !== 'move' && activeTool !== 'drag') {
            floatingSelection.current = null;
        }
    }, [activeTool, maskPath]);

    return (
        <div ref={containerRef} style={{ position: "relative", width: "100%", height: "100%", touchAction: "none", overflow: "hidden" }}>
            <div style={{
                width: "100%",
                height: "100%",
                transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`,
                transformOrigin: "0 0",
                touchAction: "none"
            }}>
                <canvas
                    ref={staticCanvasRef}
                    onPointerDown={handlePointerDown}
                    onPointerMove={handlePointerMove}
                    onPointerUp={handlePointerUp}
                    onPointerLeave={handlePointerLeave}
                    style={{
                        width: "100%",
                        height: "100%",
                        objectFit: "contain",
                        display: "block",
                        touchAction: "none",
                        cursor: isBrushActive ? 'none' : (activeMenu === 'addSubtract' ? 'crosshair' : 'default')
                    }}
                />
                <canvas
                    ref={animCanvasRef}
                    style={{
                        width: "100%",
                        height: "100%",
                        objectFit: "contain",
                        position: "absolute",
                        left: 0,
                        top: 0,
                        pointerEvents: "none",
                        zIndex: 39, // Ensure it's above overlays
                    }}
                />
            </div>
        </div>
    );
};

export default PolygonHighlighter;