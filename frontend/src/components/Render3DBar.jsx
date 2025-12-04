import React from "react";
import { ChevronDown, X, Check } from "lucide-react";

const TwinklingStars = () => {
  const STAR_COUNT = 100;
  return (
    <div className="absolute inset-0 pointer-events-none z-[95] overflow-hidden bg-black/50">
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

export default function Render3DBar({
  onModeChange,
  onConfirm,
  onCancel,
  isRelighting = false,
}) {
  const [selectedMode, setSelectedMode] = React.useState("");
  const [showConfirmBar, setShowConfirmBar] = React.useState(true);

  const [isAnimating, setIsAnimating] = React.useState(false);
  const [showActiveContent, setShowActiveContent] = React.useState(true);
  const [isCollapsed, setIsCollapsed] = React.useState(false);

  // Animation (unchanged)
  const animateBarTransition = (callback) => {
    if (isAnimating) return;

    setShowActiveContent(false);
    setIsAnimating(true);

    setTimeout(() => {
      setIsAnimating(false);

      setTimeout(() => {
        callback();
        setShowActiveContent(true);
      }, 300);
    }, 400);
  };

  const handleCancel = () => {
    animateBarTransition(() => {
      if (onCancel) onCancel(selectedMode);
    });
  };

  const handleConfirm = () => {
    if (onConfirm) onConfirm(selectedMode);
  };

  const handleToggleBar = () => {
    if (isAnimating) return;

    setShowActiveContent(false);
    setIsAnimating(true);

    setTimeout(() => {
      setIsCollapsed((prev) => !prev);
      setIsAnimating(false);

      setTimeout(() => {
        setShowActiveContent(true);
      }, 300);
    }, 400);
  };

  const sizeClasses = isAnimating
    ? "animate-compress w-16 h-16"
    : !showActiveContent
      ? "animate-expand"
      : isCollapsed
        ? "w-16 h-16"
        : "w-full h-16";

  const radiusClasses = "rounded-full";

  const ConfirmMenu = ({ onCancel, onConfirm, isLoading }) => (
    <div className="flex items-center justify-between w-full px-3 gap-2">
      {/* Cancel */}
      <button
        onClick={onCancel}
        disabled={isLoading}
        className={`flex items-center justify-center w-12 h-12 rounded-full bg-[rgba(40,40,40,0.95)] text-[#FE5959] ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        <X size={24} />
      </button>

      {/* Confirm */}
      <button
        onClick={onConfirm}
        disabled={isLoading}
        className={`flex items-center justify-center w-12 h-12 rounded-full bg-[rgba(40,40,40,0.95)] text-emerald-400 ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        {isLoading ? (
          <div className="w-5 h-5 border-2 border-emerald-400/30 border-t-emerald-400 rounded-full animate-spin" />
        ) : (
          <Check size={20} strokeWidth={2} />
        )}
      </button>

      {/* Collapse / Hide */}
      <button
        onClick={onCancel}
        disabled={isLoading}
        className={`flex items-center justify-center w-12 h-12 rounded-full bg-[rgba(40,40,40,0.95)] text-white hover:text-[#FE5959] ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        <ChevronDown size={20} strokeWidth={2} />
      </button>

      {/* Loading text */}
      {isLoading && (
        <span className="absolute -top-8 left-1/2 transform -translate-x-1/2 text-white/70 text-sm whitespace-nowrap">
          Applying relighting...
        </span>
        
      )}
    </div>
  );

  return (
    <div className="flex flex-col items-center justify-center pt-2">
      <div className="w-full px-2 mx-1 py-2 flex flex-col justify-center items-center rounded-full relative">
        <div
          className={`
            flex items-center justify-center
            border border-white/5
            bg-[rgba(18,18,18,0.95)]
            backdrop-blur-md
            transition-all duration-300
            ${sizeClasses}
            ${radiusClasses}
            shadow-xl shadow-black/50
          `}
          style={{ transformOrigin: "center center" }}
        >
          {/* COLLAPSED */}
          {isCollapsed && showActiveContent && (
            <button
              onClick={handleToggleBar}
              className="w-full h-full flex items-center justify-center"
            >
              <ChevronDown size={20} color="white" strokeWidth={2} />
            </button>
          )}

          {/* CONFIRM MENU ONLY */}
          {!isCollapsed && showActiveContent && (
            <ConfirmMenu 
              onCancel={handleCancel} 
              onConfirm={handleConfirm} 
              isLoading={isRelighting}
            />
          )}
        </div>
      </div>
    </div>
  );
}
