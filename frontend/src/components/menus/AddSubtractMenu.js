import React from "react";
import { Plus, Minus, ChevronDown, Paintbrush } from "lucide-react";

export default function AddSubtractMenu({
  onBackToMain,
  subSelection,
  setSubSelection,
  isBrushActive,
  setIsBrushActive,
}) {
  return (
    <div className="flex items-center justify-center w-full h-full px-6">
      <div className="flex items-center gap-3 flex-nowrap">

        {/* --- Add Button --- */}
        <button
          onClick={() => setSubSelection("add")}
          className={`flex-shrink-0 flex items-center justify-center px-4 py-2.5 rounded-full gap-2 transition-all duration-200 min-w-[80px] whitespace-nowrap ${
            subSelection === "add"
              ? "bg-[#FE5959] text-white"
              : "bg-[#2a2a2a] text-gray-400 hover:bg-[#353535]"
          }`}
        >
          <div className={`flex items-center justify-center w-5 h-5 rounded-full ${
            subSelection === "add" ? "bg-white/25" : "bg-white/10"
          }`}>
            <Plus size={12} strokeWidth={2.5} />
          </div>
          <span className="text-[16px] lg:text-xs">Add</span>
        </button>

        {/* --- Subtract Button --- */}
        <button
          onClick={() => setSubSelection("subtract")}
          className={`flex-shrink-0 flex items-center justify-center px-4 py-2.5 rounded-full gap-2 transition-all duration-200 min-w-[80px] whitespace-nowrap ${
            subSelection === "subtract"
              ? "bg-[#FE5959] text-white"
              : "bg-[#2a2a2a] text-gray-400 hover:bg-[#353535]"
          }`}
        >
          <div className={`flex items-center justify-center w-5 h-5 rounded-full ${
            subSelection === "subtract" ? "bg-white/25" : "bg-white/10"
          }`}>
            <Minus size={12} strokeWidth={2.5} />
          </div>
          <span className="text-[16px] lg:text-xs">Subtract</span>
        </button>

        {/* --- Brush Button --- */}
        <button
          onClick={() => setIsBrushActive(!isBrushActive)}
          className={`flex-shrink-0 flex items-center justify-center w-10 h-10 rounded-full transition-all duration-200 ${
            isBrushActive
              ? "bg-[#FE5959] text-white"
              : "bg-[#2a2a2a] text-gray-400 hover:bg-[#353535]"
          }`}
        >
          <Paintbrush size={18} strokeWidth={2} />
        </button>

        {/* --- Back Button --- */}
        <button
          onClick={onBackToMain}
          className="flex-shrink-0 flex items-center justify-center w-10 h-10 rounded-full
                     bg-[#1a1a1a] border border-white/10 text-gray-400
                     hover:text-white hover:border-white/20 transition-all duration-200"
        >
          <ChevronDown size={20} />
        </button>
      </div>
    </div>
  );
} 