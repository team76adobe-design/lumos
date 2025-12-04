import React, { useState, useRef, useEffect } from 'react';

const Tooltip = ({ children, text, position = 'top' }) => {
  const [isVisible, setIsVisible] = useState(false);
  const timeoutRef = useRef(null);

  const positionClasses = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2',
  };

  const arrowClasses = {
    top: 'top-full left-1/2 -translate-x-1/2 border-t-[#333] border-x-transparent border-b-transparent',
    bottom: 'bottom-full left-1/2 -translate-x-1/2 border-b-[#333] border-x-transparent border-t-transparent',
    left: 'left-full top-1/2 -translate-y-1/2 border-l-[#333] border-y-transparent border-r-transparent',
    right: 'right-full top-1/2 -translate-y-1/2 border-r-[#333] border-y-transparent border-l-transparent',
  };

  // Handle touch for mobile devices
  const handleTouch = (e) => {
    e.stopPropagation();
    setIsVisible(true);

    // Clear any existing timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }

    // Auto-hide after 1.5 seconds on mobile
    timeoutRef.current = setTimeout(() => {
      setIsVisible(false);
    }, 1500);
  };

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  return (
    <div
      className="relative group inline-block"
      onTouchStart={handleTouch}
    >
      {children}
      <div
        className={`
          absolute ${positionClasses[position]} z-[9999]
          px-2 py-1 text-xs font-medium text-white bg-[#333] rounded-md
          whitespace-nowrap
          transition-all duration-200 ease-out
          pointer-events-none
          ${isVisible ? 'opacity-100 visible' : 'opacity-0 invisible group-hover:opacity-100 group-hover:visible'}
        `}
      >
        {text}
        <div
          className={`
            absolute ${arrowClasses[position]}
            border-4
          `}
        />
      </div>
    </div>
  );
};

export default Tooltip;
