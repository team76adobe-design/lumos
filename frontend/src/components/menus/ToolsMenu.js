import React, { useState, useRef } from "react";
import {
  Crop,
  PaintbrushVertical,
  ImageUpscale,
  // Eraser,
  // Move,
  // Pipette,
  // Type,
  // Layers,
  // Copy,
  // Scissors,
  // RotateCw,
  Feather,
} from "lucide-react";
import { IconBlend } from "../../constants.js";
import Toast from "../Toast.jsx"; // Import Toast
import { loadSmartFrameModel } from "../../api/smartframeAPI";
import { loadOutpaintingModel } from "../../api/outpaintAPI";

const ToolsMenu = ({
  onBackToMain,
  selectedTool,
  setSelectedTool,
  onBrushClick,
  onCropClick,
  onQuillClick,
  onUpscale,
  onSelectionClick,
  onToolSelect,
}) => {
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

  const tools = [
    { id: "crop", icon: Crop, label: "Crop" },
    { id: "select", icon: IconBlend, label: "Blend" },
    { id: "brush", icon: PaintbrushVertical, label: "Style Transfer" },
    { id: "magic", icon: Feather, label: "Magic Quill" },
    { id: "upscale", icon: ImageUpscale, label: "Upscale" },
    // { id: "face", icon: ScanFace, label: "Face Detection" },
    // TODO: Not implemented yet
    // { id: "eraser", icon: Eraser, label: "Eraser" },
    // { id: "eyedropper", icon: Pipette, label: "Eyedropper" },
    // { id: "text", icon: Type, label: "Text" },
    // { id: "layers", icon: Layers, label: "Layers" },
    // { id: "duplicate", icon: Copy, label: "Duplicate" },
    // { id: "cut", icon: Scissors, label: "Cut" },
    // { id: "rotate", icon: RotateCw, label: "Rotate" },
  ];

  return (
    <div
      className="flex items-center w-full py-2"
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

      {/* Scrollable Tool Icons */}
      <div className="flex items-center gap-3 overflow-x-auto scrollbar-hide px-4 flex-1">
        {tools.map((tool) => {
          const IconComponent = tool.icon;
          const isActive = selectedTool === tool.id;

          return (
            <button
              key={tool.id}
              onClick={() => {
                // Trigger Toast for every tool click
                triggerToast(tool.label);

                // Handle Tool Logic
                if (tool.id === "brush" && onBrushClick) {
                  onBrushClick();
                } else if (tool.id === "crop") {
                  if (onToolSelect) {
                    onToolSelect(tool.label);
                  }

                  // Load models when crop tool is selected
                  loadSmartFrameModel();
                  loadOutpaintingModel();

                  onCropClick(); // show crop layout
                  return;
                } else if (tool.id === "select" && onSelectionClick) {
                  if (onToolSelect) {
                    onToolSelect(tool.label);
                  }
                  onSelectionClick();
                } else if (tool.id === "magic") {
                  if (onToolSelect) {
                    onToolSelect(tool.label);
                  }

                  onQuillClick();
                } else if (tool.id === "upscale" && onUpscale) {
                  if (onToolSelect) {
                    onToolSelect(tool.label);
                  }
                  onUpscale();
                } else {
                  if (onToolSelect) {
                    onToolSelect(tool.label);
                  }
                  setSelectedTool(tool.id);
                }
              }}
              className="flex-shrink-0 flex flex-col items-center justify-center gap-1 transition-all py-2 min-w-[50px]"
            >
              <div
                className={`flex items-center justify-center ${
                  isActive ? "text-[#FE5959]" : "text-white"
                } transition-colors`}
              >
                <IconComponent size={24} strokeWidth={2} />
              </div>
              <span
                className={`
                  text-[9px] font-medium transition-all duration-200 text-center
                  ${isActive ? "text-[#FE5959]" : "text-gray-400"}
                  ${
                    showLabels
                      ? "opacity-100 max-h-4"
                      : "opacity-0 max-h-0 overflow-hidden"
                  }
                `}
              >
                {tool.label}
              </span>
            </button>
          );
        })}
      </div>

      {/* Back Button - Fixed on right */}
      <button
        onClick={onBackToMain}
        className="flex-shrink-0 flex flex-col items-center justify-center gap-1 w-11 rounded-full bg-[rgba(40,40,40,0.8)] border border-white/10 text-white hover:text-[#FE5959] hover:border-[#FE5959]/30 transition-all mr-6 py-2"
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className="w-5 h-5"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M19 9l-7 7-7-7"
          />
        </svg>
        <span
          className={`
            text-[9px] font-medium transition-all duration-200
            ${
              showLabels
                ? "opacity-100 max-h-4"
                : "opacity-0 max-h-0 overflow-hidden"
            }
          `}
        >
          Back
        </span>
      </button>
    </div>
  );
};

export default ToolsMenu;
