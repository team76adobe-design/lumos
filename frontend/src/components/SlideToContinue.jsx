import { useState } from "react";
import { motion, useMotionValue, useTransform } from "framer-motion";

export default function SlideToContinue({ onComplete }) {
  const x = useMotionValue(0);

  // fade text as slider moves
  const opacity = useTransform(x, [0, 200], [1, 0]);

  const handleDragEnd = (_, info) => {
    if (x.get() > 200) {
      onComplete && onComplete();
    }
  };

  return (
    <div className="w-[330px] h-[65px] rounded-full bg-[#2A2A2A]/70 backdrop-blur-xl relative flex items-center overflow-hidden">
      
      {/* Text (fades out as you slide) */}
      <motion.span
        style={{ opacity }}
        className="absolute left-1/2 -translate-x-1/2 text-gray-300 text-[14px] tracking-wide"
      >
        Slide to Continue to Editing
      </motion.span>

      {/* Draggable Knob (like iPhone) */}
      <motion.div
        drag="x"
        dragConstraints={{ left: 0, right: 250 }}
        onDragEnd={handleDragEnd}
        style={{ x }}
        className="w-[60px] h-[60px] rounded-full bg-black flex items-center justify-center shadow-[0_0_25px_rgba(255,255,255,0.2)] cursor-pointer ml-2"
      >
        <svg
          width="28"
          height="28"
          viewBox="0 0 24 24"
          fill="none"
          className="text-red-400"
        >
          <path
            d="M8 5l8 7-8 7"
            stroke="red"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </motion.div>
    </div>
  );
}
