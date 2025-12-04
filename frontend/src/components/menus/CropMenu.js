import React, { useState, useRef } from "react";
import { ChevronDown } from "lucide-react";
import { IconAspectRatio, IconRotate, IconFlipHorizontal, IconFlipVertical } from "../../constants";
import Toast from "../Toast";

// Icon button with optional label
const IconButton = ({ icon: Icon, label, onClick, showLabel, isLucide = false }) => (
  <button
    onClick={onClick}
    className="flex flex-col items-center gap-1 text-gray-400 hover:text-white transition-all p-2 min-w-[48px]"
  >
    {isLucide ? <Icon size={20} /> : <Icon />}
    <span
      className={`
        text-[10px] font-medium transition-all duration-200
        ${showLabel ? 'opacity-100 max-h-4' : 'opacity-0 max-h-0 overflow-hidden'}
      `}
    >
      {label}
    </span>
  </button>
);

export default function CropMenu({
  onClose,
  onAspectRatioClick,
  onRotate,
  onFlipX,
  onFlipY,
}) {
  // --- TOAST STATE ---
  const [toastMessage, setToastMessage] = useState("");
  const [showToast, setShowToast] = useState(false);
  const [toastKey, setToastKey] = useState(0);

  // --- SWIPE TO SHOW LABELS ---
  const [showLabels, setShowLabels] = useState(false);
  const touchStartY = useRef(0);
  const hideTimeout = useRef(null);

  const handleTouchStart = (e) => {
    touchStartY.current = e.touches[0].clientY;
  };

  const handleTouchEnd = (e) => {
    const touchEndY = e.changedTouches[0].clientY;
    const swipeDistance = touchStartY.current - touchEndY;

    if (swipeDistance > 30) {
      setShowLabels(true);
      if (hideTimeout.current) clearTimeout(hideTimeout.current);
      hideTimeout.current = setTimeout(() => setShowLabels(false), 3000);
    } else if (swipeDistance < -30) {
      setShowLabels(false);
      if (hideTimeout.current) clearTimeout(hideTimeout.current);
    }
  };

  // --- TOAST HELPER ---
  const triggerToast = (message) => {
    setShowToast(false);
    setToastMessage(message);
    setToastKey((prev) => prev + 1);
    setTimeout(() => {
      setShowToast(true);
    }, 0);
  };

  return (
    <div
      className="flex items-center justify-center w-full h-full px-4"
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
      onMouseEnter={() => setShowLabels(true)}
      onMouseLeave={() => setShowLabels(false)}
    >
      {/* --- RENDER TOAST --- */}
      <Toast
        key={toastKey}
        message={toastMessage}
        isVisible={showToast}
        onComplete={() => setShowToast(false)}
      />

      {/* Centered Tool Options & Close Button */}
      <div className="flex items-center gap-[1rem]">
        <IconButton
          icon={IconAspectRatio}
          label="Ratio"
          showLabel={showLabels}
          onClick={() => {
            triggerToast("Aspect Ratio");
            if (onAspectRatioClick) onAspectRatioClick();
          }}
        />

        <IconButton
          icon={IconRotate}
          label="Rotate"
          showLabel={showLabels}
          onClick={() => {
            triggerToast("Rotate 90°");
            if (onRotate) onRotate();
          }}
        />

        <IconButton
          icon={IconFlipHorizontal}
          label="Flip H"
          showLabel={showLabels}
          onClick={() => {
            triggerToast("Flip Horizontally");
            if (onFlipX) onFlipX();
          }}
        />

        <IconButton
          icon={IconFlipVertical}
          label="Flip V"
          showLabel={showLabels}
          onClick={() => {
            triggerToast("Flip Vertically");
            if (onFlipY) onFlipY();
          }}
        />

        <IconButton
          icon={ChevronDown}
          label="Close"
          showLabel={showLabels}
          onClick={onClose}
          isLucide={true}
        />
      </div>
    </div>
  );
}
