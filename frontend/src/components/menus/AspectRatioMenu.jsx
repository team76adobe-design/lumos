import React, { useRef, useState, useEffect } from "react";
import { ChevronDown } from "lucide-react";

export default function AspectRatioMenu({ onSelect, onClose }) {
  const [selected, setSelected] = useState("Custom");
  const scrollRef = useRef(null);

  const options = [
    "Original",
    "Freeform",
    "Wallpaper",
    "9:16",
    "1:1",
    "3:4",
    "4:3",
    "16:9",
    "5:4",
  ];

  const handleSelect = (name, idx) => {
    setSelected(name);
    onSelect?.(name);

    const container = scrollRef.current;
    if (!container) return;
    
    const item = container.children[idx];
    if (!item) return;

    // center selected item
    const center =
      item.offsetLeft - container.clientWidth / 2 + item.clientWidth / 2;

    container.scrollTo({
      left: center,
      behavior: "smooth",
    });
  };

  useEffect(() => {
    // Center initially selected value
    const idx = options.indexOf(selected);
    if (idx !== -1) handleSelect(selected, idx);
  }, []);

  return (
    <div className="w-full h-full flex items-center justify-between px-2 select-none">
      
      {/* SCROLLABLE CENTERED SLIDER */}
      <div
        ref={scrollRef}
        className="
          flex-1 flex gap-3 overflow-x-auto no-scrollbar 
          items-center px-2
        "
        style={{ scrollSnapType: "x mandatory" }}
      >
        {options.map((name, idx) => (
          <button
            key={name}
            onClick={() => handleSelect(name, idx)}
            className={`
              px-4 py-2 rounded-full text-base whitespace-nowrap transition-all
              scroll-snap-align-center
              ${selected === name 
                ? "bg-white/20 text-white font-semibold" 
                : "bg-transparent text-gray-400 hover:bg-white/5"}
            `}
          >
            {name}
          </button>
        ))}
      </div>

      {/* CLOSE BUTTON */}
      <button
        onClick={() => onClose()}
        className="ml-4 flex-shrink-0 w-10 h-10 rounded-full bg-[rgba(40,40,40,0.8)] border border-white/10 text-white flex items-center justify-center hover:bg-white/10 transition-colors"
      >
        <ChevronDown size={24} />
      </button>

      <style>
        {`
          .no-scrollbar::-webkit-scrollbar { display: none; }
          .no-scrollbar { scrollbar-width: none; }
        `}
      </style>
    </div>
  );
}
