import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { CloudOff, CloudCheck } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion"; // Import Framer Motion

export default function Header({ 
  onBack, 
  onExport, 
  onMenu, 
  showExport = true 
}) {
  const navigate = useNavigate();

  // --- Cloud / Save Logic ---
  const [isSaved, setIsSaved] = useState(false);
  const [showCloudPopup, setShowCloudPopup] = useState(false);

  const handleToggleSave = () => {
    if (isSaved) {
      setIsSaved(false);
    } else {
      setShowCloudPopup(true);
    }
  };
  
  const handleAllow = () => {
    setIsSaved(true);
    setShowCloudPopup(false);
  };
  
  const handleDontAllow = () => {
    setIsSaved(false);
    setShowCloudPopup(false);
  };

  // --- Navigation Logic ---
  const handleBackClick = () => {
    if (onBack) {
      onBack();
    } else {
      setTimeout(() => {
        navigate("/");
      }, 400);
    }
  };

  // --- Animation Variants ---
  const iconVariants = {
    initial: { scale: 0, rotate: -90, opacity: 0 }, 
    animate: { scale: 1, rotate: 0, opacity: 1 },
    exit: { scale: 0, rotate: 90, opacity: 0 } 
  };

  return (
    <>
      <header className="flex-none bg-black w-full px-4 pt-safe flex items-center justify-between !items-center relative z-10 h-h-[2.5rem]">

        {/* Back Button */}
        <button className="w-8 h-8 flex items-center justify-center" onClick={handleBackClick}>
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="28" viewBox="0 0 20 28" fill="none">
            <path fillRule="evenodd" clipRule="evenodd" d="M5.32634 13.7944L18 24.2114L15.4656 26.2944L1.52477 14.836C1.18876 14.5597 1 14.1851 1 13.7944C1 13.4038 1.18876 13.0292 1.52477 12.7529L15.4656 1.29444L18 3.37753L5.32634 13.7944Z" fill="#A59F9F"/>
            <path fillRule="evenodd" clipRule="evenodd" d="M5.32634 13.7944L18 24.2114L15.4656 26.2944L1.52477 14.836C1.18876 14.5597 1 14.1851 1 13.7944C1 13.4038 1.18876 13.0292 1.52477 12.7529L15.4656 1.29444L18 3.37753L5.32634 13.7944Z" stroke="black" strokeWidth="2"/>
            <path fillRule="evenodd" clipRule="evenodd" d="M5.32634 13.7944L18 24.2114L15.4656 26.2944L1.52477 14.836C1.18876 14.5597 1 14.1851 1 13.7944C1 13.4038 1.18876 13.0292 1.52477 12.7529L15.4656 1.29444L18 3.37753L5.32634 13.7944Z" stroke="black" strokeOpacity="0.32" strokeWidth="2"/>
          </svg>
        </button>

        {/* Right Side Buttons */}
        <div className="flex items-center gap-4">

          {/* 1. Cloud Icon (Animated) */}
          <motion.button // Removed `layout` from here
            onClick={handleToggleSave}
            whileTap={{ scale: 0.85 }} // Subtle press effect
            // Ensure this button container has a fixed size and flex properties for centering
            className="w-8 h-8 flex items-center justify-center rounded-full bg-transparent transition hover:bg-white/10 relative overflow-hidden" 
          >
            <AnimatePresence mode="sync" initial={false}> 
              {isSaved ? (
                <motion.div
                  key="cloud-check"
                  variants={iconVariants}
                  initial="initial"
                  animate="animate"
                  exit="exit"
                  transition={{ type: "spring", stiffness: 400, damping: 17 }}
                  // Add position absolute to prevent layout shifts during exit/entry
                  className="absolute inset-0 flex items-center justify-center" 
                >
                  <CloudCheck className="w-6 h-6 text-green-500" />
                </motion.div>
              ) : (
                <motion.div
                  key="cloud-off"
                  variants={iconVariants}
                  initial="initial"
                  animate="animate"
                  exit="exit"
                  transition={{ type: "spring", stiffness: 400, damping: 17 }}
                  // Add position absolute to prevent layout shifts during exit/entry
                  className="absolute inset-0 flex items-center justify-center"
                >
                  <CloudOff className="w-6 h-6 text-red-500" />
                </motion.div>
              )}
            </AnimatePresence>
          </motion.button>

          {/* 2. Three Dots Menu */}
          <button className="w-8 h-6 hover:opacity-80 flex items-center justify-center" onClick={onMenu}>
            <svg xmlns="http://www.w3.org/2000/svg" width="6" height="21" viewBox="0 0 6 21" fill="none">
              <g clipPath="url(#clip0_1_137)">
                <path d="M3 15.25C4.51878 15.25 5.75 16.4812 5.75 18C5.75 19.5188 4.51878 20.75 3 20.75C1.48122 20.75 0.25 19.5188 0.25 18C0.25 16.4812 1.48122 15.25 3 15.25Z" fill="#A59F9F" stroke="black" strokeWidth="0.5"/>
                <path d="M3 7.75C4.51878 7.75 5.75 8.98122 5.75 10.5C5.75 12.0188 4.51878 13.25 3 13.25C1.48122 13.25 0.25 12.0188 0.25 10.5C0.25 8.98122 1.48122 7.75 3 7.75Z" fill="#A59F9F" stroke="black" strokeWidth="0.5"/>
                <path d="M3 0.25C4.51878 0.25 5.75 1.48122 5.75 3C5.75 4.51878 4.51878 5.75 3 5.75C1.48122 5.75 0.25 4.51878 0.25 3C0.25 1.48122 1.48122 0.25 3 0.25Z" fill="#A59F9F" stroke="black" strokeWidth="0.5"/>
              </g>
              <defs>
                <clipPath id="clip0_1_137">
                  <rect width="6" height="21" fill="white" />
                </clipPath>
              </defs>
            </svg>
          </button>

          {/* 3. Export Button */}
          {showExport && (
            <button className="w-7 h-7 hover:opacity-80 flex items-center justify-center" onClick={onExport}>
              <svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 30 30" fill="none">
                <path d="M7.25 20.25V25.25H22.75V20.25H24.75V25C24.75 26.2369 23.7369 27.25 22.5 27.25H7.5C6.26811 27.25 5.25 26.1633 5.25 25V20.25H7.25ZM21.7178 9.57129L20.3037 10.9854L16 6.68164V21H14V6.68164L9.69629 10.9854L8.28223 9.57129L15 2.85352L21.7178 9.57129Z" fill="#A59F9F" stroke="black" strokeWidth="0.5" />
              </svg>
            </button>
          )}

        </div>
      </header>

      {/* --- CLOUD POPUP OVERLAY --- */}
      {showCloudPopup && (
        <div className="fixed inset-0 backdrop-blur-md bg-black/40 flex items-center justify-center z-[1000]">
          <div className="bg-white w-[85%] max-w-[320px] rounded-2xl pt-5 pb-2 text-center shadow-xl animate-fadeIn">
            <h2 className="font-semibold text-black text-xl mb-2 px-2">
              You are going on Cloud Mode
            </h2>
            <p className="text-sm text-gray-600 mb-5 px-4">
              We'll securely process your image on our servers, only for your benefit.
            </p>
            <div className="flex flex-col gap-2">
              <button
                onClick={handleAllow}
                className="text-blue-600 text-lg font-semibold py-2"
              >
                Allow
              </button>
              <button
                onClick={handleDontAllow}
                className="text-blue-600 text-lg py-1"
              >
                Don't Allow
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}