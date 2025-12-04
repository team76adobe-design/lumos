import React, { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { loadBackgroundRemovalModel, removeBackground, unloadBackgroundRemovalModel } from '../api/backgroundRemovalAPI';
import { processImageTo3D } from '../api/sf3dAPI';

const BLEND_MODES = [
  { id: "3d", label: "3D Blend" },
  { id: "standard", label: "Standard Blend" },
];

// --- Helper Hook for Safe Image URLs ---
const useObjectUrl = (fileOrUrl) => {
  return useMemo(() => {
    if (!fileOrUrl) return null;
    if (typeof fileOrUrl === 'string') return fileOrUrl;
    if (fileOrUrl instanceof File || fileOrUrl instanceof Blob) {
      return URL.createObjectURL(fileOrUrl);
    }
    return null;
  }, [fileOrUrl]);
};

// --- Vertical Wipe Transition Component ---
const VerticalWipeTransition = ({ imageFile, processedImageFile, isActive, isProcessing, isCompleted }) => {
  const imageUrl = useObjectUrl(imageFile);
  const bgUrl = useObjectUrl(processedImageFile);

  // --- 1. Timing Configuration ---
  const scanDuration = 2; // 1s Down + 1s Up

  const transitionConfig = {
    duration: scanDuration,
    ease: "easeInOut",
    repeat: isProcessing ? Infinity : 0, // Loop while processing
  };

  // Determine the animation state
  const getAnimateState = () => {
    if (isCompleted) {
      // Show final result - top layer hidden, processed image visible
      return { clipPath: "inset(100% 0 0 0)" };
    }
    // Default - show original
    return { clipPath: "inset(0% 0 0 0)" };
  };

  return (
    <div className="relative inline-block w-full">
      
      <div className="relative rounded-lg overflow-hidden">
        {/* Sizing Helper */}
        <img 
          src={imageUrl} 
          alt="Reference" 
          className="block w-full h-auto max-h-[50vh] object-contain opacity-0" 
        />

        {/* Bottom Layer: Result */}
        <div className="absolute inset-0 flex items-center justify-center">
          <img 
            src={bgUrl} 
            alt="Result" 
            className="w-full h-full object-cover" 
          />
        </div>

        {/* Top Layer: Original Image */}
        <motion.div
          className="absolute inset-0 overflow-hidden"
          initial={{ clipPath: "inset(0% 0 0 0)" }}
          animate={getAnimateState()}
          transition={{ duration: 0.5 }}
        >
          {imageUrl && (
            <img 
              src={imageUrl} 
              alt="Original" 
              className="w-full h-full object-cover" 
            />
          )}
        </motion.div>
      </div>

      {/* Wipe Line */}
      {!isCompleted && (
        <motion.div
          className="absolute -left-[5%] w-[110%] h-1 bg-sky-400 shadow-[0_0_15px_rgba(56,189,248,0.8)] z-10"
          style={{ transform: 'translateY(-50%)' }}
          initial={{ top: '0%' }}
          animate={isActive
            ? {
                top: [
                  "0%",    // Start
                  "100%",  // Scan Down
                  "0%"     // Scan Up
                ]
              }
            : { top: "0%" }
          }
          transition={transitionConfig}
        />
      )}
    </div>
  );
};

// --- Main Popup Component ---
const ImagePreviewPopup = ({ isVisible, imageFile, backgroundImage, onClose }) => {
  const navigate = useNavigate();
  const [selectedMode, setSelectedMode] = useState(null);
  const [showWipeAnimation, setShowWipeAnimation] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingError, setProcessingError] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [isCompleted, setIsCompleted] = useState(false);

  // 3D model generation state
  const [isGenerating3D, setIsGenerating3D] = useState(false);
  const [glbModelBlob, setGlbModelBlob] = useState(null);

  const imageUrl = useObjectUrl(imageFile);

  // Process image when popup opens
  useEffect(() => {
    if (isVisible && imageFile && !processedImage) {
      const processImage = async () => {
        setIsProcessing(true);
        setProcessingError(null);

        // Start wipe animation when /load API starts
        setShowWipeAnimation(true);

        try {
          // Step 1: Load the model
          console.log('Loading model...');
          await loadBackgroundRemovalModel();

          // Step 2: Process the image (streaming happens here)
          console.log('Removing background...');
          const processedBlob = await removeBackground(imageFile);

          // Update the processed image
          const processedFile = new File(
            [processedBlob],
            'background-removed.png',
            { type: 'image/png' }
          );
          setProcessedImage(processedFile);

          // Step 3: Unload the model - STOP ANIMATION AND SHOW RESULT
          console.log('Unloading model...');
          setIsProcessing(false); // Stop looping
          setIsCompleted(true); // Show final result immediately

          await unloadBackgroundRemovalModel();
          console.log('Background removal complete!');

        } catch (error) {
          console.error('Background removal failed:', error);
          setProcessingError(error.message || 'Failed to process image');
          setIsProcessing(false);
          setShowWipeAnimation(false); // On error, stop animation completely

          // Try to cleanup
          try {
            await unloadBackgroundRemovalModel();
          } catch (cleanupError) {
            console.error('Cleanup error:', cleanupError);
          }
        }
      };

      processImage();
    }
  }, [isVisible, imageFile, processedImage]);

  // Reset state when popup closes
  useEffect(() => {
    if (!isVisible) {
      setProcessedImage(null);
      setShowWipeAnimation(false);
      setProcessingError(null);
      setSelectedMode(null);
      setIsCompleted(false);
      setIsGenerating3D(false);
      setGlbModelBlob(null);
    }
  }, [isVisible]);

  const handleSelection = async (modeId) => {
    if (!processedImage) {
      console.error('No processed image available');
      return;
    }

    setSelectedMode(modeId);

    if (modeId === '3d') {
      // Generate 3D model from the background-removed image
      setIsGenerating3D(true);
      setProcessingError(null);

      try {
        console.log('Generating 3D model from processed image...');
        const glbBlob = await processImageTo3D(processedImage);
        setGlbModelBlob(glbBlob);
        console.log('3D model generated successfully!');

        // Navigate with the 3D model
        setTimeout(() => {
          onClose();
          navigate('/3d', {
            state: {
              overlayImage: processedImage,
              backgroundImage: backgroundImage,
              blendMode: modeId,
              glbModel: glbBlob
            }
          });
        }, 200);
      } catch (error) {
        console.error('Failed to generate 3D model:', error);
        setProcessingError(error.message || 'Failed to generate 3D model');
        setIsGenerating3D(false);
        setSelectedMode(null);
      }
    } else {
      // Standard blend - navigate directly
      setTimeout(() => {
        onClose();
        navigate('/3d', {
          state: {
            overlayImage: processedImage,
            backgroundImage: backgroundImage,
            blendMode: modeId
          }
        });
      }, 200);
    }
  };

  if (isVisible && imageFile) {
    return createPortal(
      <AnimatePresence>
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.5 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="fixed inset-0 bg-black z-[9999]"
            onClick={onClose}
          />

          {/* Popup Container - centers the popup */}
          <div className="fixed bottom-0 left-0 right-0 z-[10000] flex justify-center pointer-events-none">
            <motion.div
              initial={{ y: '100%', opacity: 0 }}
              animate={{ y: '0%', opacity: 1 }}
              exit={{ y: '100%', opacity: 0 }}
              transition={{ type: 'spring', damping: 25, stiffness: 300, duration: 0.5 }}
              className="bg-[#2e2e2e] rounded-t-3xl shadow-2xl pointer-events-auto"
              style={{ height: 'auto', maxHeight: '85vh', width: '100%', maxWidth: '480px' }}
            >
            <div className="flex flex-col items-center justify-start pt-12 p-6 space-y-6 pb-12">
              
              {/* Image Section - Show VerticalWipeTransition during and after processing */}
              {imageUrl && (
                <motion.div
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ delay: 0.2, duration: 0.3 }}
                  className="relative w-full max-w-md"
                >
                  <VerticalWipeTransition
                    imageFile={imageFile}
                    processedImageFile={processedImage || imageFile}
                    isActive={showWipeAnimation}
                    isProcessing={isProcessing}
                    isCompleted={isCompleted}
                  />
                </motion.div>
              )}

              {/* Blend Mode Buttons */}
              <motion.div
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.4, duration: 0.3 }}
                className="flex items-center justify-center gap-3 w-full max-w-md px-4"
              >
                {BLEND_MODES.map((mode) => {
                  const isActive = selectedMode === mode.id;
                  const isDisabled = isProcessing || !processedImage || isGenerating3D;
                  const isLoading = isGenerating3D && mode.id === '3d' && isActive;
                  return (
                    <button
                      key={mode.id}
                      onClick={() => handleSelection(mode.id)}
                      disabled={isDisabled}
                      className={`
                        flex-1 px-6 py-3 text-sm font-medium rounded-full
                        transition-all duration-200 flex items-center justify-center gap-2
                        ${isActive
                          ? "bg-[rgba(254,89,89,0.9)] border border-[#FE5959] text-white shadow-lg shadow-[#FE5959]/30"
                          : "bg-[rgba(40,40,40,0.8)] text-gray-300 hover:bg-[rgba(40,40,40,0.95)] border border-white/10"
                        }
                        ${isDisabled ? "opacity-50 cursor-not-allowed" : ""}
                      `}
                    >
                      {isLoading && (
                        <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      )}
                      {mode.label}
                    </button>
                  );
                })}
              </motion.div>

              {/* Helper Text */}
              {!isProcessing && !processingError && processedImage && !isGenerating3D && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.6, duration: 0.3 }}
                  className="flex items-center justify-center space-x-2"
                >
                  <div className="w-2 h-2 bg-[#FE5959] rounded-full animate-pulse"></div>
                  <span className="text-white/70 text-sm">
                    Background removed! Select a blend mode to continue
                  </span>
                </motion.div>
              )}

              {/* 3D Generation State */}
              {isGenerating3D && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex flex-col items-center justify-center space-y-3"
                >
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                    <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                    <div className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                  </div>
                  <span className="text-white/70 text-sm">
                    Generating 3D model...
                  </span>
                </motion.div>
              )}

              {/* Loading State */}
              {isProcessing && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex flex-col items-center justify-center space-y-3"
                >
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-sky-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                    <div className="w-2 h-2 bg-sky-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                    <div className="w-2 h-2 bg-sky-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                  </div>
                  <span className="text-white/70 text-sm">
                    Removing background...
                  </span>
                </motion.div>
              )}

              {/* Error State */}
              {processingError && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex flex-col items-center justify-center space-y-3 px-4"
                >
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                    <span className="text-red-400 text-sm font-medium">
                      {processingError}
                    </span>
                  </div>
                  <button
                    onClick={() => {
                      setProcessingError(null);
                      setProcessedImage(null);
                    }}
                    className="px-4 py-2 text-xs bg-[rgba(40,40,40,0.8)] text-gray-300 rounded-lg hover:bg-[rgba(40,40,40,0.95)] transition-all border border-white/10"
                  >
                    Try Again
                  </button>
                </motion.div>
              )}

            </div>
            </motion.div>
          </div>
        </>
      </AnimatePresence>,
      document.body
    );
  }

  return null;
};

export default ImagePreviewPopup;