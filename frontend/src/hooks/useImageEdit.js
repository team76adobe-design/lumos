import { useEffect, useRef, useState } from "react";

export function useImageEditorProcessor(editValues, imageSrc) {
    const canvasRef = useRef(null);
    const [outputUrl, setOutputUrl] = useState(null);
    const workerRef = useRef(null);
    const editValuesRef = useRef(editValues);
    
    // Queue management
    const isWorkerBusy = useRef(false);
    const pendingEditValues = useRef(null);

    // Keep ref updated
    useEffect(() => {
        editValuesRef.current = editValues;
    }, [editValues]);

    // Initialize worker
    useEffect(() => {
        workerRef.current = new Worker(process.env.PUBLIC_URL + '/pixelWorker.js');
        
        workerRef.current.onmessage = (e) => {
            const { pixels } = e.data;
            const canvas = canvasRef.current;
            if (canvas) {
                const ctx = canvas.getContext("2d", { willReadFrequently: true });
                // Create ImageData from the received buffer
                const imageData = new ImageData(new Uint8ClampedArray(pixels), canvas.width, canvas.height);
                ctx.putImageData(imageData, 0, 0);
                setOutputUrl(canvas.toDataURL("image/jpeg", 0.95));
            }

            // Worker is done processing
            isWorkerBusy.current = false;

            // Check if there's a pending update that came in while we were busy
            if (pendingEditValues.current) {
                const nextValues = pendingEditValues.current;
                pendingEditValues.current = null; // Clear pending
                
                isWorkerBusy.current = true;
                workerRef.current.postMessage({ 
                    type: 'update', 
                    editValues: nextValues 
                });
            }
        };

        return () => {
            if (workerRef.current) {
                workerRef.current.terminate();
            }
        };
    }, []);

    // 1. Load image only when imageSrc changes
    useEffect(() => {
        if (!imageSrc) return;

        const img = new Image();
        img.crossOrigin = "Anonymous";
        img.src = imageSrc;

        img.onload = () => {
            const canvas = canvasRef.current || document.createElement("canvas");
            canvasRef.current = canvas;

            canvas.width = img.width;
            canvas.height = img.height;

            const ctx = canvas.getContext("2d", { willReadFrequently: true });
            ctx.drawImage(img, 0, 0);

            const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            
            // Send source image to worker once
            if (workerRef.current) {
                const pixels = new Uint8ClampedArray(imgData.data);
                
                // Mark busy as setImage triggers processing
                isWorkerBusy.current = true;
                
                workerRef.current.postMessage({ 
                    type: 'setImage',
                    imageData: {
                        width: imgData.width,
                        height: imgData.height,
                        data: pixels
                    },
                    editValues: editValuesRef.current
                }, [pixels.buffer]);
            }
        };
    }, [imageSrc]);

    // 2. Process image when editValues changes
    useEffect(() => {
        if (!canvasRef.current || !workerRef.current) return;

        if (isWorkerBusy.current) {
            // Worker is busy, queue this update as the latest pending one
            // This effectively debounces/throttles by dropping intermediate frames
            pendingEditValues.current = editValues;
        } else {
            // Worker is free, send immediately
            isWorkerBusy.current = true;
            workerRef.current.postMessage({ 
                type: 'update', 
                editValues 
            });
        }
    }, [editValues]);


    return { outputUrl, canvasRef };
}