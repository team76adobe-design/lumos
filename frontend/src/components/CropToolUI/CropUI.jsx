// FINAL PATCHED index.jsx (Photoshop‑style crop behavior)
// ------------------------------------------------------
// ✔ Crop box free movement
// ✔ Max crop size = 2W × 2H (image rendered size)
// ✔ Image moves freely with no clamping
// ✔ Infinite black canvas behavior
// ------------------------------------------------------

import React, {
  useRef,
  useState,
  useEffect,
  forwardRef,
  useImperativeHandle,
} from "react";
import { RotationRuler } from "../RotationRuler";
import {
  CROP_TOOL_STYLES,
  INITIAL_TRANSFORM,
  ASPECT_RATIOS,
} from "../../constants/cropToolConstants";
import {
  calculateImageFit,
  computeRequiredScale,
} from "../../utils/cropToolUtils";
import { useCanvasExport } from "./canvasExport";
import {
  ConfirmButtons,
  CropHandles,
  CropBoxGrid,
} from "./CropElements";
import { SmartFrameButton } from "./ToolbarMenu";
import { runSmartFrame } from "../../api/smartframeAPI";
import { useCropInteraction } from "../../hooks/useCropInteraction";


const TwinklingStars = () => {
  const STAR_COUNT = 100;

  return (
    <div className="absolute inset-0 z-[9999] overflow-hidden bg-black/80 backdrop-blur-sm flex items-center justify-center">
      {Array.from({ length: STAR_COUNT }).map((_, i) => (
        <span
          key={i}
          className="twinkle-star"
          style={{
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
            width: `${1 + Math.random() * 3}px`,
            height: `${1 + Math.random() * 3}px`,
            animationDelay: `${Math.random() * 2}s`,
            animationDuration: `${1.5 + Math.random() * 2}s`,
            opacity: 0,
          }}
        />
      ))}

    </div>
  );
};
const CropUI = forwardRef(({ imageSrc, onBackToMain, onStateChange }, ref) => {
  const containerRef = useRef(null);
  const imgRef = useRef(null);
  const cropBoxRef = useRef(null);

  const [transform, setTransform] = useState(INITIAL_TRANSFORM);
  const [imageWidth, setImageWidth] = useState(null);
  const [imageHeight, setImageHeight] = useState(null);
  const [aspectRatio, setAspectRatio] = useState(null);
  const [showGrid, setShowGrid] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [smoothTransition, setSmoothTransition] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentImageSrc, setCurrentImageSrc] = useState(imageSrc);

  // NEW — limit crop size to 2W × 2H (Photoshop behavior)
  const [maxCrop, setMaxCrop] = useState({ w: 0, h: 0 });

  const { exportCropToCanvas } = useCanvasExport();

  /* Sync state with prop */
  useEffect(() => {
    setCurrentImageSrc(imageSrc);
  }, [imageSrc]);

  /* Notify parent of confirm state */
  useEffect(() => {
    if (onStateChange) {
      onStateChange(showConfirm && !isProcessing);
    }
  }, [showConfirm, isProcessing, onStateChange]);

  /* Load natural image size */
  useEffect(() => {
    if (!currentImageSrc) return;
    const img = new Image();
    img.src = currentImageSrc;
    img.onload = () => {
      setImageWidth(img.naturalWidth);
      setImageHeight(img.naturalHeight);
    };
  }, [currentImageSrc]);

  /* Reset transform on image change */
  useEffect(() => {
    setTransform(INITIAL_TRANSFORM);
  }, [currentImageSrc]);

  /* Inject tool CSS */
  useEffect(() => {
    const style = document.createElement("style");
    style.innerHTML = CROP_TOOL_STYLES + `
      .crop-box {
        box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.5);
      }
      @keyframes twinkle {
        0% { opacity: 0; transform: scale(0.5); }
        50% { opacity: 1; transform: scale(1.2); }
        100% { opacity: 0; transform: scale(0.5); }
      }
      .twinkle-star {
        position: absolute;
        background: white;
        border-radius: 50%;
        animation-name: twinkle;
        animation-iteration-count: infinite;
        animation-timing-function: ease-in-out;
      }
    `;
    document.head.appendChild(style);
    return () => style.remove();
  }, []);

  // --- Use the new interaction hook ---
  const { bindGestures, setCropBox, fitToScreen } = useCropInteraction({
    cropBoxRef,
    containerRef,
    transform,
    setTransform,
    maxCrop,
    aspectRatio,
    setShowGrid,
    setShowConfirm,
    setSmoothTransition,
  });

  const setCropToImage = () => {
    const crop = cropBoxRef.current;
    const img = imgRef.current;
    const cont = containerRef.current;
    if (!crop || !img || !cont) return;

    const contRect = cont.getBoundingClientRect();
    const imgRect = img.getBoundingClientRect();
    const left = imgRect.left - contRect.left;
    const top = imgRect.top - contRect.top;
    crop.style.left = `${left}px`;
    crop.style.top = `${top}px`;
    crop.style.width = `${imgRect.width}px`;
    crop.style.height = `${imgRect.height}px`;
    crop.style.transform = 'none';
  };

  /* Aspect ratio apply */
  const applyAspectRatio = (ratio) => {
    const crop = cropBoxRef.current;
    const cont = containerRef.current;
    if (!crop || !cont) return;

    const rect = cont.getBoundingClientRect();
    let width = rect.width * 0.8;
    let height = ratio ? width / ratio : rect.height * 0.8;

    if (height > rect.height * 0.8) {
      height = rect.height * 0.8;
      width = ratio ? height * ratio : width;
    }

    crop.style.width = `${width}px`;
    crop.style.height = `${height}px`;
    crop.style.left = `50%`;
    crop.style.top = `50%`;
    crop.style.transform = `translate(-50%, -50%)`;

    setAspectRatio(ratio);
    setShowConfirm(true);
  };

  /* Rotation updates */
  const onRotationChange = (deg) => {
    setTransform((p) => ({ ...p, rotation: deg }));
    setShowConfirm(true);
  };

  /* Cancel */
  const handleCancelCrop = () => {
    setCropToImage();
    setShowGrid(false);
    setShowConfirm(false);
  };

  /* Photoshop behavior — Ensure image covers crop on export */
  const handleConfirmCrop = async () => {
    // Removed auto-scaling logic to allow outpainting (black areas)
    // instead of forcing the image to cover the crop box.

    setIsProcessing(true);
    await exportCropToCanvas(cropBoxRef, imgRef, containerRef, transform, (newImage) => {
      setCurrentImageSrc(newImage);
    });
    setIsProcessing(false);

    setShowConfirm(false);
    setShowGrid(false);
  };

  const onImgLoad = (e) => {
    const box = containerRef.current;
    const img = e.currentTarget;
    if (!box || !img) return;

    const { w, h } = calculateImageFit(img, box.clientWidth, box.clientHeight);
    img.style.width = `${w}px`;
    img.style.height = `${h}px`;

    setMaxCrop({ w: w * 2, h: h * 2 });

    setCropToImage();

    // Initialize view to match "readjusted" state (zoomed in)
    // Use setTimeout to ensure DOM updates from setCropToImage are ready
    setTimeout(() => fitToScreen(false), 0);
  };

  const handleAspectRatioSelect = (opt) => {
    if (opt === "Original" && imageWidth && imageHeight) {
      applyAspectRatio(imageWidth / imageHeight);
    } else if (opt in ASPECT_RATIOS) {
      applyAspectRatio(ASPECT_RATIOS[opt]);
    }
  };

  const handleSmartFrame = async () => {
    setIsProcessing(true);
    try {
      if (!imgRef.current || !imageWidth || !imageHeight) {
        console.warn("SmartFrame: image not ready yet.");
        return;
      }

      // Convert currentImageSrc (URL) to Blob for the API
      const response = await fetch(currentImageSrc);
      const blob = await response.blob();
      const file = new File([blob], "image.png", { type: blob.type });

      const apiResponse = await runSmartFrame(file);
      console.log("SmartFrame response:", apiResponse);

      if (!apiResponse || apiResponse.status !== "Success" || !apiResponse.bbox) {
        console.warn("SmartFrame: Invalid response from backend", apiResponse);
        return;
      }

      const [xmin, ymin, xmax, ymax] = apiResponse.bbox;

      // The model returns coordinates relative to the original image size.
      // We need to scale them to the currently displayed image size.
      const imgRect = imgRef.current.getBoundingClientRect();
      const scaleX = imgRect.width / imageWidth;
      const scaleY = imgRect.height / imageHeight;

      // Calculate new crop box dimensions relative to the container
      const contRect = containerRef.current.getBoundingClientRect();

      // Image top-left relative to container
      const imgLeft = imgRect.left - contRect.left;
      const imgTop = imgRect.top - contRect.top;

      const newX = imgLeft + xmin * scaleX;
      const newY = imgTop + ymin * scaleY;
      const newW = (xmax - xmin) * scaleX;
      const newH = (ymax - ymin) * scaleY;

      setCropBox({
        x: newX,
        y: newY,
        w: newW,
        h: newH,
      });
    } catch (e) {
      console.error("Smart Frame failed:", e);
    } finally {
      setIsProcessing(false);
    }
  };

  useImperativeHandle(ref, () => ({
    rotate: () => {
      setTransform((prev) => ({ ...prev, rotation: (prev.rotation - 90) % 360 }));
      setShowConfirm(true);
    },
    flipX: () => {
      setTransform((prev) => ({ ...prev, flipX: prev.flipX * -1 }));
      setShowConfirm(true);
    },
    flipY: () => {
      setTransform((prev) => ({ ...prev, flipY: prev.flipY * -1 }));
      setShowConfirm(true);
    },
    setAspectRatio: handleAspectRatioSelect,
    cancel: () => onBackToMain(currentImageSrc !== imageSrc ? currentImageSrc : null),
    confirm: handleConfirmCrop,
    reset: handleCancelCrop,
  }));

  return (
    <div className="relative w-full h-full flex flex-col items-center">
      <SmartFrameButton onSmartFrame={handleSmartFrame} />

      <div
        ref={containerRef}
        className="relative flex-1 w-full h-full bg-black flex items-center justify-center overflow-hidden touch-none"
        {...bindGestures()}
      >
        <div
          ref={cropBoxRef}
          className={`crop-box ${showGrid ? "show-grid" : ""}`}
        >
          {/* Removed GridOverlay */}
          <CropBoxGrid />
          <CropHandles />
        </div>

        <img
          ref={imgRef}
          src={currentImageSrc}
          alt="crop"
          draggable={false} // Fixed: movegable -> draggable
          className="crop-image select-none touch-none"
          style={{
            transform: `translate(${transform.x}px, ${transform.y}px) rotate(${transform.rotation}deg) scale(${transform.scale * transform.flipX}, ${transform.scale * transform.flipY})`,
            transition: smoothTransition ? "transform 0.4s cubic-bezier(0.25, 1, 0.5, 1)" : "transform 0.05s linear",
            maxWidth: "none",
            height: "auto",
            objectFit: "contain",
          }}
          onLoad={onImgLoad}
        />
        {isProcessing && (
          <TwinklingStars />
        )}
      </div>

      <p className="text-white/80 text-sm mt-2 mb-6">Move or resize to crop</p>
      {/* Push confirm buttons downward (same trick as other code) */}
      {/* <div className="h-24 w-full" /> */}



      <div className="w-full px-4">
        <RotationRuler rotation={transform.rotation} onRotationChange={onRotationChange} />
      </div>
    </div>
  );
});

export default CropUI;

