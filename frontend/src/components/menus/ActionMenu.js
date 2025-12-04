import React, { useState, useRef } from "react";
import { ChevronDown } from "lucide-react";
import { IconDrag, IconMove, IconErase } from "../../constants";
import Toast from "../Toast";
import { loadInpaintModel } from "../../api/inpaintAPI";

// Reusable icon button with optional label
const IconButton = ({ icon: Icon, label, onClick, isLucide = false, showLabel = false }) => (
  <button
    onClick={onClick}
    className="flex flex-col items-center gap-1 text-gray-400 hover:text-white transition-all p-2 min-w-[48px]"
  >
    {isLucide ? <Icon size={22} /> : <Icon />}
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

export default function ActionMenu({ onClose, onerase, onmove, ondrag }) {
  const [showLabels, setShowLabels] = useState(false);
  const touchStartY = useRef(0);
  const hideTimeout = useRef(null);

  // --- TOAST STATE ---
  const [toastMessage, setToastMessage] = React.useState("");
  const [showToast, setShowToast] = React.useState(false);
  const [toastKey, setToastKey] = React.useState(0);

  // --- TOAST HELPER ---
  const triggerToast = (message) => {
    setShowToast(false);
    setToastMessage(message);
    setToastKey((prev) => prev + 1);
    setTimeout(() => {
      setShowToast(true);
    }, 0);
  };

  // --- SWIPE HANDLERS ---
  const handleTouchStart = (e) => {
    touchStartY.current = e.touches[0].clientY;
  };

  const handleTouchEnd = (e) => {
    const touchEndY = e.changedTouches[0].clientY;
    const swipeDistance = touchStartY.current - touchEndY;

    // Swipe up (at least 30px) shows labels
    if (swipeDistance > 30) {
      setShowLabels(true);

      // Clear any existing timeout
      if (hideTimeout.current) {
        clearTimeout(hideTimeout.current);
      }

      // Auto-hide after 3 seconds
      hideTimeout.current = setTimeout(() => {
        setShowLabels(false);
      }, 3000);
    }
    // Swipe down hides labels
    else if (swipeDistance < -30) {
      setShowLabels(false);
      if (hideTimeout.current) {
        clearTimeout(hideTimeout.current);
      }
    }
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

      <div className="flex items-center gap-8">
        {/* ERASE TOOL */}
        <IconButton
          icon={IconErase}
          label="Erase"
          showLabel={showLabels}
          onClick={() => {
            loadInpaintModel();
            triggerToast("Erase");
            if (onerase) onerase();
          }}
        />

        {/* MOVE TOOL */}
        <IconButton
          icon={IconMove}
          label="Move"
          showLabel={showLabels}
          onClick={() => {
            triggerToast("Move");
            if (onmove) onmove();
          }}
        />

        {/* DRAG TOOL */}
        <IconButton
          icon={IconDrag}
          label="Drag"
          showLabel={showLabels}
          onClick={() => {
            triggerToast("Drag");
            if (ondrag) ondrag();
          }}
        />

        {/* CLOSE BUTTON */}
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