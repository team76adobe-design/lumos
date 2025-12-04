import { useRef, useEffect } from "react";
import { useGesture } from "@use-gesture/react";

/**
 * Custom hook for handling crop tool interactions:
 * 1. Image panning/zooming/rotating (Gestures)
 * 2. Crop box resizing (Pointer events)
 */
export const useCropInteraction = ({
    cropBoxRef,
    containerRef,
    transform,
    setTransform,
    maxCrop,
    aspectRatio,
    setShowGrid,
    setShowConfirm,
    setSmoothTransition,
}) => {
    const resizingRef = useRef(false);
    const activeDirRef = useRef(null);
    const transformRef = useRef(transform);

    // Keep transformRef in sync
    useEffect(() => {
        transformRef.current = transform;
    }, [transform]);

    // --- Helper: Fit Crop to Screen (Zoom In) ---
    const fitToScreen = (animate = true) => {
        const crop = cropBoxRef.current;
        const cont = containerRef.current;
        if (!crop || !cont) return;

        const cropRect = crop.getBoundingClientRect();
        const contRect = cont.getBoundingClientRect();

        // Current crop box relative to container
        const currentCrop = {
            x: cropRect.left - contRect.left,
            y: cropRect.top - contRect.top,
            w: cropRect.width,
            h: cropRect.height
        };

        // Calculate scale to fit container (with margin)
        const margin = 20;
        const availableW = contRect.width - margin * 2;
        const availableH = contRect.height - margin * 2;

        const scaleX = availableW / currentCrop.w;
        const scaleY = availableH / currentCrop.h;
        const scale = Math.min(scaleX, scaleY);

        // New Crop Dimensions
        const newW = currentCrop.w * scale;
        const newH = currentCrop.h * scale;

        // New Crop Position (Centered)
        const newX = (contRect.width - newW) / 2;
        const newY = (contRect.height - newH) / 2;

        // Calculate shift of the crop center
        const oldCenterX = currentCrop.x + currentCrop.w / 2;
        const oldCenterY = currentCrop.y + currentCrop.h / 2;
        
        // Offset from container center
        const oldCenterOffsetX = oldCenterX - contRect.width / 2;
        const oldCenterOffsetY = oldCenterY - contRect.height / 2;

        // Use the latest transform from ref
        const currentTransform = transformRef.current;

        // New Image Offset (relative to New Crop Center which is 0,0):
        const newImageX = (currentTransform.x - oldCenterOffsetX) * scale;
        const newImageY = (currentTransform.y - oldCenterOffsetY) * scale;
        const newImageScale = currentTransform.scale * scale;

        // Apply updates with smooth transition
        if (animate) {
            if (setSmoothTransition) setSmoothTransition(true);
            crop.style.transition = "all 0.4s cubic-bezier(0.25, 1, 0.5, 1)";
        } else {
            if (setSmoothTransition) setSmoothTransition(false);
            crop.style.transition = "none";
        }
        
        crop.style.width = `${newW}px`;
        crop.style.height = `${newH}px`;
        crop.style.left = `${newX}px`;
        crop.style.top = `${newY}px`;
        
        setTransform(prev => ({
            ...prev,
            x: newImageX,
            y: newImageY,
            scale: newImageScale
        }));
        
        // Reset transition after animation
        if (animate) {
            setTimeout(() => {
                if (setSmoothTransition) setSmoothTransition(false);
                crop.style.transition = "";
            }, 400);
        } else {
            crop.style.transition = "";
        }
    };

    // --- 1. Image Gestures (Pan, Pinch, Wheel) ---
    const bindGestures = useGesture(
        {
            onDrag: ({ offset: [dx, dy] }) => {
                if (resizingRef.current) return;
                setTransform((p) => ({ ...p, x: dx, y: dy }));
                setShowConfirm(true);
            },
            onPinch: ({ offset: [s], event }) => {
                event.preventDefault(); // Prevent browser zoom
                if (resizingRef.current) return;
                setTransform((prev) => ({ ...prev, scale: s }));
                setShowConfirm(true);
            },
            onWheel: ({ event }) => {
                event.preventDefault(); // Prevent browser zoom
                if (resizingRef.current) return;
                setTransform((prev) => {
                    const factor = event.deltaY > 0 ? 0.94 : 1.06;
                    let scale = Math.max(0.05, Math.min(4, prev.scale * factor));
                    return { ...prev, scale };
                });
                setShowConfirm(true);
            },
        },
        {
            drag: { from: () => [transform.x, transform.y] },
            wheel: { eventOptions: { passive: false } },
            pinch: { 
                eventOptions: { passive: false },
                scaleBounds: { min: 0.05, max: 4 },
                from: () => [transform.scale, 0]
            },
        }
    );

    // --- 1.5 Crop Box Movement (Drag the box itself) ---
    // REMOVED: bindCropMove to allow dragging on crop box to pan image instead.

    // --- 2. Crop Box Resizing ---
    useEffect(() => {
        const crop = cropBoxRef.current;
        const cont = containerRef.current;

        if (!crop || !cont) return;

        let startX = 0,
            startY = 0,
            startW = 0,
            startH = 0,
            startL = 0,
            startT = 0;

        const MIN = 40;

        const onMove = (e) => {
            if (!resizingRef.current) return;
            e.stopPropagation(); // Stop propagation to avoid other handlers
            e.preventDefault(); // Prevent default browser actions

            const dx = e.clientX - startX;
            const dy = e.clientY - startY;
            const dir = activeDirRef.current;

            let newW = startW,
                newH = startH,
                newL = startL,
                newT = startT;

            if (aspectRatio === null) {
                if (dir.includes("e")) newW = startW + dx;
                if (dir.includes("w")) {
                    newW = startW - dx;
                    newL = startL + dx;
                }
                if (dir.includes("s")) newH = startH + dy;
                if (dir.includes("n")) {
                    newH = startH - dy;
                    newT = startT + dy;
                }
            } else {
                const r = aspectRatio;
                let delta = Math.abs(dx) > Math.abs(dy) ? dx : dy;

                if (dir.includes("e") || dir.includes("w")) {
                    newW = startW + (dir.includes("w") ? -delta : delta);
                    newH = newW / r;
                } else {
                    newH = startH + (dir.includes("n") ? -delta : delta);
                    newW = newH * r;
                }

                if (dir.includes("w")) newL = startL + (startW - newW);
                if (dir.includes("n")) newT = startT + (startH - newH);
            }

            // Limit crop size to max 2W×2H
            newW = Math.min(newW, maxCrop.w);
            newH = Math.min(newH, maxCrop.h);

            // --- Clamp to container bounds ---
            const contRect = cont.getBoundingClientRect();
            const contW = contRect.width;
            const contH = contRect.height;

            // Clamp Left
            if (newL < 0) {
                if (dir.includes('w')) newW += newL;
                newL = 0;
            }
            // Clamp Top
            if (newT < 0) {
                if (dir.includes('n')) newH += newT;
                newT = 0;
            }
            // Clamp Right
            if (newL + newW > contW) {
                newW = contW - newL;
                if (aspectRatio) newH = newW / aspectRatio;
            }
            // Clamp Bottom
            if (newT + newH > contH) {
                newH = contH - newT;
                if (aspectRatio) newW = newH * aspectRatio;
            }
            
            // Re-check Right/Bottom for aspect ratio side-effects
            if (aspectRatio) {
                if (newL + newW > contW) {
                    newW = contW - newL;
                    newH = newW / aspectRatio;
                }
                if (newT + newH > contH) {
                    newH = contH - newT;
                    newW = newH * aspectRatio;
                }
            }

            if (newW < MIN || newH < MIN) return;

            crop.style.width = `${newW}px`;
            crop.style.height = `${newH}px`;
            crop.style.left = `${newL}px`;
            crop.style.top = `${newT}px`;
            crop.style.transform = "none";

            setShowConfirm(true);
        };

        const stop = () => {
            resizingRef.current = false;
            activeDirRef.current = null;
            setShowGrid(false);
            fitToScreen(); // Zoom in to the cropped region
        };

        const start = (e) => {
            const dir = e.currentTarget.dataset.dir;
            activeDirRef.current = dir;
            resizingRef.current = true;
            setShowGrid(true);
            setShowConfirm(true);

            e.stopPropagation(); // Prevent crop move or image pan

            const rect = crop.getBoundingClientRect();
            const contRect = cont.getBoundingClientRect();

            startX = e.clientX;
            startY = e.clientY;

            startW = rect.width;
            startH = rect.height;
            startL = rect.left - contRect.left;
            startT = rect.top - contRect.top;

            window.addEventListener("pointermove", onMove);
            window.addEventListener("pointerup", stop, { once: true });
            e.preventDefault();
        };

        const handles = crop.querySelectorAll("[data-dir]");
        handles.forEach((h) => h.addEventListener("pointerdown", start));

        return () => {
            handles.forEach((h) => h.removeEventListener("pointerdown", start));
            window.removeEventListener("pointermove", onMove);
        };
    }, [aspectRatio, maxCrop, cropBoxRef, containerRef, setShowGrid, setShowConfirm]);

    // --- 3. Programmatic Crop Box Setting (e.g. Smart Frame) ---
    const setCropBox = ({ x, y, w, h }) => {
        const crop = cropBoxRef.current;
        if (!crop) return;

        crop.style.width = `${w}px`;
        crop.style.height = `${h}px`;
        crop.style.left = `${x}px`;
        crop.style.top = `${y}px`;
        crop.style.transform = "none";

        setShowGrid(true);
        setShowConfirm(true);
    };

    return { bindGestures, resizingRef, setCropBox, fitToScreen };
};
