import React, { useRef, useState } from "react";
import { Feather, Check, ChevronDown, Plus, Minus } from "lucide-react";
import Toast from "../Toast";

export default function QuillMenu({
  onCancel,
  onOpenColorPanel,
  penColor,
  quillMode,
  setQuillMode,
  onDone,
  // alternate prop names (MenuBar passes these in some places)
  activeTool,
  setActiveTool,
  brushSize,
  setBrushSize,
  brushColor,
  setBrushColor,
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

  const triggerToast = (message) => {
    setShowToast(false);
    setToastMessage(message);
    setToastKey((prev) => prev + 1);
    setTimeout(() => setShowToast(true), 0);
  };

  // prefer direct props, fall back to alternate names
  const currentPenColor = penColor || brushColor || "#ffffff";
  const currentMode = quillMode || activeTool;
  const setMode = setQuillMode || setActiveTool || (() => {});
  const handleDone = onDone || (() => triggerToast("Done"));
  return (
    <div
      className="flex items-center justify-center w-full h-full px-4"
      onTouchStart={handleTouchStart}
      onTouchEnd={handleTouchEnd}
      onMouseEnter={() => setShowLabels(true)}
      onMouseLeave={() => setShowLabels(false)}
    >
      <div className="flex items-center gap-9 relative">

       {/* --- TOAST --- */}
       <Toast
         key={toastKey}
         message={toastMessage}
         isVisible={showToast}
         onComplete={() => setShowToast(false)}
       />

       {/* ADD BUTTON */}
      <button
        className="p-2 transition relative flex flex-col items-center"
        onClick={() => setMode(currentMode === 'add' ? null : 'add')}
        aria-label="Add"
      >
        <div className="relative">
          <Feather
            className={
              currentMode === 'add'
                ? "w-5 text-[#FE5959] drop-shadow-[0px_6px_30px_rgba(254,89,89,0.35)]"
                : "w-5 text-white hover:text-[#FE5959]"
            }
          />
          {/* small plus icon positioned bottom-right (like color preview) */}
          <Plus
            size={14}
            className={`absolute bottom-[-2px] -right-2 transition ${currentMode === 'add' ? 'text-[#FE5959]' : 'text-white/80 hover:text-[#FE5959]'}`}
          />
        </div>
        <span className={`text-[9px] font-medium mt-1 transition-all duration-200 ${currentMode === 'add' ? 'text-[#FE5959]' : 'text-gray-400'} ${showLabels ? 'opacity-100 max-h-4' : 'opacity-0 max-h-0 overflow-hidden'}`}>
          Add
        </span>
      </button>

      {/* SUBTRACT BUTTON */}
      <button
        className="p-2 transition relative flex flex-col items-center"
        onClick={() => setMode(currentMode === 'subtract' ? null : 'subtract')}
        aria-label="Subtract"
      >
        <div className="relative">
          <Feather
            className={
              currentMode === 'subtract'
                ? "w-5 text-[#FE5959] drop-shadow-[0px_6px_30px_rgba(254,89,89,0.35)]"
                : "w-5 text-white hover:text-[#FE5959]"
            }
          />
          {/* small minus icon positioned bottom-right (like color preview) */}
          <Minus
            size={14}
            className={`absolute bottom-[-2px] -right-2 transition ${currentMode === 'subtract' ? 'text-[#FE5959]' : 'text-white/80 hover:text-[#FE5959]'}`}
          />
        </div>
        <span className={`text-[9px] font-medium mt-1 transition-all duration-200 ${currentMode === 'subtract' ? 'text-[#FE5959]' : 'text-gray-400'} ${showLabels ? 'opacity-100 max-h-4' : 'opacity-0 max-h-0 overflow-hidden'}`}>
          Subtract
        </span>
      </button>


        {/* COLOR BUTTON — opens bottom panel */}
        <button
          className="p-2 transition relative flex flex-col items-center"
          onClick={() => {
            if (setMode) setMode('color');
            if (onOpenColorPanel) onOpenColorPanel();
            if (!onOpenColorPanel) triggerToast("Open Color Panel");
          }}
          title="Color Picker"
        >
          <div className="relative">
            <Feather className={
              currentMode === 'color'
                ? "w-5 text-[#FE5959] drop-shadow-[0px_6px_30px_rgba(254,89,89,0.35)]"
                : "w-5 text-white hover:text-[#FE5959]"
            } />

            {/* LIVE COLOR PREVIEW DOT */}
            <div
              className="absolute bottom-[-2px] right-[-4px] w-3 h-3 rounded-full"
              style={{ background: currentPenColor }}
            />
          </div>
          <span className={`text-[9px] font-medium mt-1 transition-all duration-200 ${currentMode === 'color' ? 'text-[#FE5959]' : 'text-gray-400'} ${showLabels ? 'opacity-100 max-h-4' : 'opacity-0 max-h-0 overflow-hidden'}`}>
            Color
          </span>
        </button>

        {/* DONE/TICK BUTTON */}
        <button
          onClick={() => {
            handleDone();
            triggerToast("Done");
          }}
          className="flex flex-col items-center justify-center text-green-400 p-2"
          title="Done"
        >
          <Check className="w-5 h-5" />
          <span className={`text-[9px] font-medium mt-1 transition-all duration-200 text-green-400 ${showLabels ? 'opacity-100 max-h-4' : 'opacity-0 max-h-0 overflow-hidden'}`}>
            Done
          </span>
        </button>

        {/* CLOSE */}
        <button
          onClick={() => {
            if (onCancel) onCancel();
            triggerToast("Close");
          }}
          className="text-gray-400 hover:text-white transition-colors p-2 flex flex-col items-center"
          title="Close"
        >
          <ChevronDown size={24} />
          <span className={`text-[9px] font-medium transition-all duration-200 text-gray-400 ${showLabels ? 'opacity-100 max-h-4' : 'opacity-0 max-h-0 overflow-hidden'}`}>
            Close
          </span>
        </button>

      </div>
    </div>
  );
        }
