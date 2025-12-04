import { useState, useRef, useEffect } from 'react';
import { segment } from '../api/segmentAPI';
import { resizeTo512 } from '../utils/imageResize';
import { toast } from 'sonner';

export const useImageHandler = () => {
    const [imageSrc, setImageSrc] = useState(null);
    const [fgPoints, setFgPoints] = useState([]);
    const [bgPoints, setBgPoints] = useState([]);
    const [polygons, setPolygons] = useState([]);
    const [maskPath, setMaskPath] = useState(null);
    const [isSegmenting, setIsSegmenting] = useState(false);

    const fileInputRef = useRef(null);
    const canvasExportRef = useRef(null);

    // Image Upload Handler
    const handleImageUpload = (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                setImageSrc(event.target.result);
                setFgPoints([]);
                setBgPoints([]);
                setPolygons([]);
            };
            reader.readAsDataURL(file);
        }
    };

    // Point Removal Handler
    const handlePointRemove = (type, index) => {
        const nextFg = type === 'fg' ? fgPoints.filter((_, i) => i !== index) : fgPoints;
        const nextBg = type === 'bg' ? bgPoints.filter((_, i) => i !== index) : bgPoints;

        setFgPoints(nextFg);
        setBgPoints(nextBg);

        const xs = [...nextFg.map((p) => Math.round(p.imageX)), ...nextBg.map((p) => Math.round(p.imageX))];
        const ys = [...nextFg.map((p) => Math.round(p.imageY)), ...nextBg.map((p) => Math.round(p.imageY))];
        const labels = [...nextFg.map(() => 1), ...nextBg.map(() => 0)];

        // runSegmentation(imageSrc, xs, ys, labels, { setPolygons, setIsSegmenting });
        const runSeg = async () => {
            setIsSegmenting(true);
            try {
                const blob = await fetch(imageSrc).then((r) => r.blob());
                const originalFile = new File([blob], "image.png", { type: blob.type });

                const { file: resizedFile, originalWidth, originalHeight } = await resizeTo512(originalFile);

                const scaledXs = xs.map((x) => Math.round((x / originalWidth) * 512));
                const scaledYs = ys.map((y) => Math.round((y / originalHeight) * 512));

                const data = await segment(resizedFile, scaledXs, scaledYs, labels, true);

                if (data && data.mask_coords) {
                    const newPolygons = data.mask_coords.map((contour) => ({
                        points: contour.map(([x, y]) => ({
                            x: (x / 512) * originalWidth,
                            y: (y / 512) * originalHeight,
                        })),
                        stroke: "lime",
                    }));
                    setPolygons(newPolygons);
                    if (data.mask_base64) {
                        const res = await fetch(`data:image/png;base64,${data.mask_base64}`);
                        const blob = await res.blob();
                        const file = new File([blob], "mask.png", { type: "image/png" });
                        setMaskPath(file);
                    }
                }
            } catch (e) {
                console.error(e);
            } finally {
                setIsSegmenting(false);
            }
        };
        runSeg();
    };

    // Reset Handler
    const handleReset = () => {
        setFgPoints([]);
        setBgPoints([]);
        setPolygons([]);
        setMaskPath(null);
    };

    // Export Handler
    const handleExport = () => {
        if (!imageSrc) {
            toast.error('No image to export');
            return;
        }

        const canvas = canvasExportRef.current;
        if (canvas) {
            canvas.toBlob((blob) => {
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'edited-image.png';
                a.click();
                URL.revokeObjectURL(url);
            });
        }
    };

    // Export Canvas Setup
    useEffect(() => {
        if (imageSrc) {
            const img = new Image();
            img.onload = () => {
                const canvas = canvasExportRef.current;
                if (canvas) {
                    canvas.width = img.naturalWidth;
                    canvas.height = img.naturalHeight;
                    const ctx = canvas.getContext('2d', { willReadFrequently: true });
                    ctx.drawImage(img, 0, 0);
                }
            };
            img.src = imageSrc;
        }
    }, [imageSrc, polygons]);

    return {
        imageSrc,
        setImageSrc,
        fgPoints,
        setFgPoints,
        bgPoints,
        setBgPoints,
        polygons,
        setPolygons,
        maskPath,
        setMaskPath,
        isSegmenting,
        setIsSegmenting,
        fileInputRef,
        canvasExportRef,
        handleImageUpload,
        handlePointRemove,
        handleReset,
        handleExport,
    };
};