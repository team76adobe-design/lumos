import React, { useEffect } from 'react';

const Toast = ({ message, isVisible, onComplete, variant }) => {
  useEffect(() => {
    if (isVisible) {
      const timer = setTimeout(() => {
        onComplete();
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [isVisible, onComplete]);

  if (!isVisible) return null;

  return (
    <div
      className={`absolute pointer-events-none z-50 left-1/2 transform -translate-x-1/2 ${
        variant === 'actions' ? '-top-28' : '-top-14'
      }`}
    >
      <div className="animate-toastPopIn">
        <div className="bg-white/10 backdrop-blur-md rounded-full px-4 py-2 text-white text-sm font-small shadow-lg">
          {message}
        </div>
      </div>
    </div>
  );
};

export default Toast;
