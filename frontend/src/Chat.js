import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import SlideToContinue from "./components/SlideToContinue";
import Header from "./components/Header";
import { toast } from 'sonner';
import { Loader2 } from 'lucide-react';

// 1. Import the Hook and the Component
import GeminiInputBar from "./components/GeminiInputBar";
import { useGeminiRecorder } from "./hooks/useGeminiRecorder";

// API
import { loadTextToImageModel, generateImage, unloadTextToImageModel } from "./api/textToImageAPI";

const cards = [
  { title: "Futuristic Cityscape", image: "https://thumbs.dreamstime.com/b/futuristic-cityscape-night-neon-lights-skyscrapers-under-full-moon-346064543.jpg" },
  { title: "Vintage Art Style", image: "https://www.shutterstock.com/shutterstock/photos/2315486613/display_1500/stock-vector-pop-art-pinup-girl-retro-color-style-comic-glamour-gorgeous-abstract-colorful-lady-wow-2315486613.jpg" },
  { title: "Cinematic Portrait", image: "https://i.pinimg.com/236x/9e/fa/fe/9efafe36651293008fcc4251d5621101.jpg" },
  { title: "Dreamy Landscape", image: "https://img.freepik.com/premium-photo/dreamy-landscape-colorful-expansive-vista_1106493-56874.jpg" },
];

export default function Chat() {
  const {
    isRecording,
    waveData,
    transcribedText,
    setTranscribedText,
    startRecording,
    confirmRecording,
    cancelRecording,
  } = useGeminiRecorder();

  const navigate = useNavigate();

  // UI flow
  const [showPromptMenu, setShowPromptMenu] = useState(false);
  const [showImageResult, setShowImageResult] = useState(false);
  const [generatedImage, setGeneratedImage] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const modelLoadAttempted = useRef(false);

  // Load model on mount (with guard against StrictMode double-call)
  useEffect(() => {
    if (modelLoadAttempted.current) return;
    modelLoadAttempted.current = true;

    setIsModelLoading(true);
    loadTextToImageModel()
      .then(() => {
        setIsModelLoaded(true);
        console.log('Text-to-image model ready');
      })
      .catch((error) => {
        console.error('Failed to load model:', error);
        toast.error('Failed to load image generation model');
      })
      .finally(() => {
        setIsModelLoading(false);
      });
  }, []);

  // Handle image generation with optional prompt parameter
  const handleGenerateWithPrompt = async (prompt) => {
    const textToUse = prompt || transcribedText;

    if (!textToUse.trim()) {
      toast.error('Please enter a prompt');
      return;
    }

    // Wait for model to be loaded
    if (!isModelLoaded) {
      if (isModelLoading) {
        toast.error('Model is still loading, please wait...');
      } else {
        toast.error('Model failed to load. Please refresh the page.');
      }
      return;
    }

    setIsGenerating(true);
    setShowPromptMenu(false);

    try {
      const result = await generateImage(textToUse, {
        height: 1024,
        width: 1024,
        guidance_scale: 4.5,
        steps: 20,
      });

      // Unload model after generation to free up resources
      unloadTextToImageModel().catch(err => console.warn('Failed to unload text-to-image model:', err));

      let imageUrl = null;

      if (result.blob) {
        imageUrl = URL.createObjectURL(result.blob);
      } else if (result.image_base64 || result.image_b64 || result.image) {
        const b64 = result.image_base64 || result.image_b64 || result.image;
        imageUrl = b64.startsWith('data:') ? b64 : `data:image/png;base64,${b64}`;
      } else if (result.image_url) {
        imageUrl = result.image_url;
      }

      if (imageUrl) {
        setGeneratedImage(imageUrl);
        setShowImageResult(true);
        toast.success('Image generated!');
      } else {
        toast.error('No image returned');
      }
    } catch (error) {
      console.error('Generation failed:', error);
      toast.error('Failed to generate image: ' + (error.message || 'Unknown error'));
      // Also unload on error
      unloadTextToImageModel().catch(err => console.warn('Failed to unload text-to-image model:', err));
    } finally {
      setIsGenerating(false);
      setIsModelLoaded(false); // Reset so it reloads next time
    }
  };

  // Wrapper for menu button (uses current transcribedText)
  const handleGenerateImage = () => handleGenerateWithPrompt();

  // container width (centered)
  const containerClass = "w-full max-w-2xl mx-auto";

  return (
    <div className="chat-root">
      {/* compact, single-file style block to enforce 100vh behavior */}
      <style>{`
        :root {
          --header-h: 48px;
          --footer-h: 78px; /* space reserved for input + safe area */
        }
        /* root viewport */
        .chat-root {
          height: 100vh;
          min-height: 100vh;
          max-height: 100vh;
          display: flex;
          flex-direction: column;
          background: #000;
          color: #fff;
          overflow: hidden; /* prevent page scroll */
        }
        /* fixed compact header */
        .chat-header {
          height: var(--header-h);
          min-height: var(--header-h);
          flex: 0 0 var(--header-h);
          z-index: 40;
          backdrop-filter: blur(4px);
        }
        /* main area uses remaining viewport (no scroll) */
        .chat-main {
          height: calc(100vh - var(--header-h) - var(--footer-h));
          min-height: calc(100vh - var(--header-h) - var(--footer-h));
          max-height: calc(100vh - var(--header-h) - var(--footer-h));
          overflow: hidden;
          padding: 8px 12px;
          display: flex;
          flex-direction: column;
          align-items: stretch;
          justify-content: flex-start;
        }
        /* the fixed input overlay at bottom */
        .chat-input-overlay {
          position: fixed;
          left: 50%;
          transform: translateX(-50%);
          bottom: env(safe-area-inset-bottom, 12px);
          width: min(100%, 720px);
          padding: 6px;
          z-index: 120;
          box-sizing: border-box;
          pointer-events: auto;
        }
        /* ensure it sits above canvas/content */
        .chat-input-inner {
          width: 100%;
        }
        /* prompt menu animation */
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(8px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fadeIn {
          animation: fadeInUp 0.2s ease-out;
        }
        /* recording overlays */
        .recording-controls {
          position: absolute;
          right: 8px;
          top: 50%;
          transform: translateY(-50%);
          z-index: 160;
        }
        .wave-overlay {
          position: absolute;
          left: 0;
          right: 0;
          top: 0;
          bottom: 0;
          display:flex;
          align-items:center;
          justify-content:center;
          pointer-events: none;
        }
        /* image result overlay: fill but keep content inside viewport */
        .image-result-overlay {
          position: fixed;
          inset: 0;
          z-index: 2000;
          background: rgba(0,0,0,0.96);
          display:flex;
          flex-direction:column;
          align-items:center;
          justify-content:flex-start;
          padding-top: calc(var(--header-h) + 8px);
          box-sizing: border-box;
        }
        .image-result-inner {
          width: min(100%, 720px);
          padding: 12px;
          display:flex;
          flex-direction:column;
          align-items:center;
          gap: 12px;
          box-sizing: border-box;
        }
        .image-result-img {
          width: 100%;
          max-height: calc(100vh - var(--header-h) - var(--footer-h) - 140px);
          object-fit: cover;
          border-radius: 12px;
        }

        /* responsive grid tweaks */
        @media (max-width:420px) {
          .chat-main { padding-left: 10px; padding-right: 10px; }
        }
      `}</style>

      {/* Header (fixed at top) */}
      <div className="chat-header">
        <div className={containerClass}>
          <Header showExport={false} />
        </div>
      </div>

      {/* Main area (no vertical scrolling) */}
      <main className="chat-main">
        {/* Show model loading state */}
        {/* {isModelLoading && (
          <div className="absolute top-16 left-1/2 -translate-x-1/2 bg-[#181818] px-4 py-2 rounded-full flex items-center gap-2 z-50">
            <Loader2 size={16} className="text-[#ff4b5c] animate-spin" />
            <span className="text-xs text-gray-400">Loading model...</span>
          </div>
        )} */}

        {/* Show loading state when generating */}
        {isGenerating ? (
          <div className={`${containerClass} px-0 pt-8 flex flex-col items-center`}>
            <Loader2 size={48} className="text-[#ff4b5c] animate-spin" />
            <p className="mt-4 text-gray-300 text-center">
              Generating your image...
            </p>
            <p className="mt-2 text-gray-500 text-sm text-center">
              "{transcribedText}"
            </p>
          </div>
        ) : (
          /* Show prompt cards when not generating */
          <div className={`${containerClass} px-0 pt-2`}>
            <h1 className="text-center text-xl font-semibold leading-tight">
              What can I make for
              <br />
              you today?
            </h1>

            <div className="mt-2 flex justify-center">
              <div className="w-20 h-[3px] rounded-full bg-[#ff4b5c]" />
            </div>

            <div className="mt-6 grid grid-cols-2 gap-3">
              {cards.map((card) => (
                <button
                  key={card.title}
                  className="bg-[#181818] rounded-2xl p-2 shadow-md flex flex-col items-center transition-transform duration-150 active:scale-95"
                  onClick={() => {
                    // Set the prompt text and immediately generate
                    setTranscribedText(card.title);
                    // Trigger generation with the card title
                    handleGenerateWithPrompt(card.title);
                  }}
                >
                  <div className="w-full rounded-lg overflow-hidden aspect-[4/3]">
                    <img src={card.image} alt={card.title} className="w-full h-full object-cover" />
                  </div>
                  <p className="mt-2 text-[12px] font-medium text-gray-100 text-center px-1">
                    {card.title}
                  </p>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Spacer to ensure some breathing room above the fixed input */}
        <div style={{ flex: 1 }} />
      </main>

      {/* Fixed input overlay (always visible, above content) */}
      <div className="chat-input-overlay" aria-hidden={false}>
        <div className="chat-input-inner">
          <GeminiInputBar
            value={transcribedText}
            onChange={(val) => setTranscribedText(val)}
            onMicClick={() => {
              startRecording();
              setShowPromptMenu(false);
            }}
            onSendClick={() => setShowPromptMenu((s) => !s)}
            showBackIcon={false}
            disabled={isRecording}
            hideActions={isRecording}
            placeholder={isRecording ? "" : "Type your idea..."}
            className={isRecording ? "opacity-60" : ""}
          />

          {/* Prompt menu popup */}
          {showPromptMenu && !isRecording && !isGenerating && (
            <>
              {/* Backdrop to close menu */}
              <div
                className="fixed inset-0 z-[125]"
                onClick={() => setShowPromptMenu(false)}
              />
              {/* Menu panel */}
              <div className="absolute bottom-full right-2 mb-3 z-[130] animate-fadeIn">
                <div className="bg-[#1a1a1a] border border-white/10 rounded-2xl p-2 shadow-xl min-w-[200px]">
                  {/* Generate Image option */}
                  <button
                    onClick={handleGenerateImage}
                    className="w-full flex items-center gap-3 px-3 py-3 rounded-xl hover:bg-white/10 transition-colors"
                  >
                    <div className="w-10 h-10 rounded-full bg-[#ff4b5c] flex items-center justify-center flex-shrink-0">
                      <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <div className="text-left">
                      <p className="text-sm font-medium text-white">Generate Image</p>
                      <p className="text-xs text-gray-500">Create from your prompt</p>
                    </div>
                  </button>

                  {/* Divider */}
                  <div className="h-px bg-white/10 mx-2 my-1" />

                  {/* Enhance prompt option */}
                  <button
                    onClick={() => {
                      // TODO: Implement enhance prompt
                      setShowPromptMenu(false);
                    }}
                    className="w-full flex items-center gap-3 px-3 py-3 rounded-xl hover:bg-white/10 transition-colors"
                  >
                    <div className="w-10 h-10 rounded-full bg-[#2a2a2a] border border-white/20 flex items-center justify-center flex-shrink-0">
                      <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
                      </svg>
                    </div>
                    <div className="text-left">
                      <p className="text-sm font-medium text-white">Enhance Prompt</p>
                      <p className="text-xs text-gray-500">Improve with AI</p>
                    </div>
                  </button>
                </div>
              </div>
            </>
          )}

          {/* Loading overlay when generating
          {isGenerating && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/60 rounded-full">
              <Loader2 size={24} className="text-[#ff4b5c] animate-spin" />
            </div>
          )} */}

          {/* Recording controls (right side of input) */}
          {/* --- RECORDING CONTROLS (Correct UI Style) --- */}
          {isRecording && (
            <div className="absolute right-4 top-1/2 -translate-y-1/2 flex flex-row gap-3 z-[999]">

              {/* Cancel Button */}
              <button
                type="button"
                onClick={cancelRecording}
                className="
        w-9 h-9 
        flex items-center justify-center 
        rounded-full 
        bg-[#2a2a2a]
        text-white text-sm
        shadow-[0_4px_12px_rgba(0,0,0,0.5)]
        hover:bg-[#3a3a3a]
        active:scale-95
        transition
      "
              >
                ✕
              </button>

              {/* Confirm Button */}
              <button
                type="button"
                onClick={confirmRecording}
                className="
        w-9 h-9 
        flex items-center justify-center 
        rounded-full 
        bg-[#ff4b5c]
        text-white text-sm
        shadow-[0_4px_12px_rgba(0,0,0,0.5)]
        hover:bg-[#ff616a]
        active:scale-95
        transition
      "
              >
                ✓
              </button>
            </div>
          )}

          {/* --- WAVEFORM OVERLAY (Correct UI Style) --- */}
          {isRecording && (
            <div
              className="
      absolute inset-0 pointer-events-none 
      px-7 flex items-center
      z-[500]
    "
            >
              <div className="flex w-[70%] items-center justify-center gap-[3px] h-full pl-2">
                {waveData.map((v, i) => (
                  <div
                    key={i}
                    className="rounded-full transition-all duration-150 ease-out"
                    style={{
                      width: "2px",
                      height: v === null ? "3px" : `${4 + v * 26}px`,
                      backgroundColor: v === null ? "#333" : "#fff",
                      opacity: v === null ? 0.25 : 0.7 + v * 0.3,
                    }}
                  />
                ))}
              </div>
            </div>
          )}


        </div>
      </div>

      {/* Image result overlay (fits the viewport without page scroll) */}
      {showImageResult && (
        <div className="image-result-overlay" role="dialog" aria-modal="true">
          {/* header area with back */}
          <div style={{ width: "100%" }}>
            <div className={containerClass} style={{ paddingLeft: 12, paddingRight: 12 }}>
              <Header showExport={false} onBack={() => setShowImageResult(false)} />
            </div>
          </div>

          <div className="image-result-inner" style={{ marginTop: 8 }}>
            <img src={generatedImage} alt="Generated" className="image-result-img" />
            <div style={{ width: "100%", display: "flex", gap: 10 }}>
              <button
                className="bg-[#181818] flex-1 py-3 rounded-full text-sm text-gray-300 flex items-center justify-center gap-2 disabled:opacity-50"
                onClick={() => {
                  setShowImageResult(false);
                  handleGenerateImage();
                }}
                disabled={isGenerating}
              >
                {isGenerating ? <Loader2 size={16} className="animate-spin" /> : null}
                Regenerate
              </button>
              <button
                className="bg-[#181818] flex-1 py-3 rounded-full text-sm text-gray-300"
                onClick={() => {
                  setShowImageResult(false);
                }}
              >
                Edit Prompt
              </button>
            </div>

            <div style={{ width: "100%", display: "flex", justifyContent: "center", paddingTop: 8 }}>
              {/* SlideToContinue is placed so it's visible and accessible */}
              <div style={{ width: "90%", maxWidth: 400 }}>
                <SlideToContinue onComplete={() => navigate("/photoroom", { state: { compositedImage: generatedImage } })} />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
