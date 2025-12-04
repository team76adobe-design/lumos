import React, { useEffect } from "react";
import GeminiInputBar from "../GeminiInputBar";
import { useGeminiRecorder } from "../../hooks/useGeminiRecorder";

/* --------------------------------------------
   GEMINI MENU COMPONENT
   - Manages Audio Logic via Custom Hook
   - Uses GeminiInputBar for the base UI
   - Overlays Waveform/Controls when recording
--------------------------------------------- */

export default function GeminiMenu({
    value,
    onChange,
    onSend,
    onBackToMain,
    isProcessingAnim,
    setIsProcessingAnim,
    onAnalyzeClick,
    isAnalyzing
}) {
  // 1. USE THE CUSTOM HOOK
  const {
    isRecording,
    waveData,
    transcribedText,
    startRecording,
    confirmRecording,
    cancelRecording,
  } = useGeminiRecorder();

  // 2. SYNC TRANSCRIPTION WITH PARENT COMPONENT
  // When the hook gets new text, we pass it up to the parent's onChange handler
  useEffect(() => {
    if (transcribedText && onChange) {
      onChange(transcribedText);
    }
  }, [transcribedText, onChange]);

  return (
    <div className="w-full py-3 px-0 flex justify-center">
      <div className="relative w-full max-w-4xl">
        {/* 3. THE REUSABLE INPUT COMPONENT */}
        <GeminiInputBar
          value={value}
          onChange={onChange}
          onMicClick={startRecording}
          onSendClick={onSend}
          onBackClick={onBackToMain}
          onAnalyzeClick={onAnalyzeClick}
          isAnalyzing={isAnalyzing}
          showBackIcon={true} // Explicitly enabling the 3rd icon
          disabled={isRecording} // Disable input during recording
          hideActions={isRecording} // Hide buttons during recording
          placeholder={isRecording ? "" : "Type your idea..."}
          isProcessingAnim={isProcessingAnim}
          setIsProcessingAnim={setIsProcessingAnim}
          // Dim the bar slightly when recording so the overlay pops
          className={
            isRecording ? "opacity-60 transition-opacity" : "transition-opacity"
          }
        />

        {/* 4. RECORDING CONTROLS OVERLAY 
            (Buttons appear over the right side of the bar) */}
        {isRecording && (
          <div className="absolute right-4 top-1/2 -translate-y-1/2 flex flex-row gap-3 z-50">
            {/* CANCEL BUTTON */}
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

            {/* CONFIRM BUTTON */}
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

        {/* 5. WAVEFORM OVERLAY 
            (Appears over the text area part of the bar) */}
        {isRecording && (
          <div
            className="
              absolute inset-0 pointer-events-none 
              px-7 flex items-center
              z-10 
            "
          >
            <div className="flex w-[70%] items-center justify-center gap-[3px] h-full pl-2">
              {waveData.map((v, i) => (
                <div
                  key={i}
                  className="rounded-full transition-all duration-150 ease-out"
                  style={{
                    width: "3px",
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
  );
} 