import React from "react";
import frameButton from "../../icons/FrameButton.svg";
import rotateIcon from "../../icons/90 degree rotate.svg";
import xSymmetry from "../../icons/xSymmetry.svg";
import ySymmetry from "../../icons/ySymmetry.svg";

/**
 * Bottom toolbar menu with transform controls
 */
export const ToolbarMenu = ({
  onAspectRatioClick,
  onRotate,
  onFlipX,
  onFlipY,
  onCancel,
}) => {
  return (
    <div className="w-full px-6 pb-8 flex items-center justify-center">
      <div className="flex items-center gap-6 px-6 py-3 bg-[#1c1c1c] rounded-full">
        <button
          className="p-2 bg-white/10 rounded-full hover:bg-white/20 transition"
          onClick={onAspectRatioClick}
          title="Aspect Ratio"
        >
          <img src={frameButton} className="w-5" alt="aspect ratio" />
        </button>

        <button
          className="p-2 bg-white/10 rounded-full hover:bg-white/20 transition"
          onClick={onRotate}
          title="Rotate 90°"
        >
          <img src={rotateIcon} className="w-5" alt="rotate" />
        </button>

        <button
          className="p-2 bg-white/10 rounded-full hover:bg-white/20 transition"
          onClick={onFlipX}
          title="Flip Horizontally"
        >
          <img src={xSymmetry} className="w-5" alt="flip x" />
        </button>

        <button
          className="p-2 bg-white/10 rounded-full hover:bg-white/20 transition"
          onClick={onFlipY}
          title="Flip Vertically"
        >
          <img src={ySymmetry} className="w-5" alt="flip y" />
        </button>

        <button
          onClick={onCancel}
          className="w-8 h-8 rounded-full bg-[rgba(40,40,40,0.8)] text-white border border-white/10 hover:bg-[rgba(60,60,60,0.8)] transition"
          title="Close"
        >
          <svg viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
            <path
              d="M19 9l-7 7-7-7"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      </div>
    </div>
  );
};

/**
 * Smart Frame button
 */
// The button now receives onSmartFrame as a prop
export const SmartFrameButton = ({ onSmartFrame }) => (
  <div className="pb-2">
    <button
      onClick={onSmartFrame}   // <--- triggers SmartFrame process
      className="px-4 py-2 mb-[10px] rounded-full text-red-400 bg-white/10 border border-red-400 text-sm flex gap-2 hover:bg-red-400/10 transition"
    >
      <img src={frameButton} className="w-5" alt="smart frame" />
      Smart Frame
    </button>
  </div>
);

