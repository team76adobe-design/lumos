import React, { useState, useRef } from "react";
import { IconVoicePrompt, IconPrompt, IconAnalyze } from "../constants";
import { motion } from "framer-motion";

// Compact icon button with optional label
const IconButton = ({ label, onClick, disabled, className = "", showLabel = false, children }) => (
    <button
        type="button"
        onClick={onClick}
        disabled={disabled}
        className={`
            flex flex-col items-center gap-0.5
            hover:bg-white/10 active:scale-95
            transition-all rounded-full px-2 py-1
            ${disabled ? "opacity-50 cursor-not-allowed" : ""}
            ${className}
        `}
    >
        {children}
        <span
            className={`
                text-[9px] text-gray-400 font-medium transition-all duration-200
                ${showLabel ? 'opacity-100 max-h-4' : 'opacity-0 max-h-0 overflow-hidden'}
            `}
        >
            {label}
        </span>
    </button>
);

export default function GeminiInputBar({
    value,
    onChange,
    onMicClick,
    onSendClick,
    onBackClick,
    onAnalyzeClick,
    showBackIcon = true,
    placeholder = "Type your idea...",
    disabled = false,
    isAnalyzing = false,
    isProcessingAnim,
    setIsProcessingAnim,
    className = "",
    hideActions = false,
}) {
    // State to track if the text area has grown beyond a single line
    const [isExpanded, setIsExpanded] = useState(false);
    const [showLabels, setShowLabels] = useState(false);
    const touchStartY = useRef(0);
    const hideTimeout = useRef(null);
    const textareaRef = useRef(null);

    // Auto-resize when value changes
React.useEffect(() => {
    if (!textareaRef.current) return;

    const el = textareaRef.current;

    // Reset height so scrollHeight is correct
    el.style.height = "auto";

    const newHeight = Math.min(el.scrollHeight, 144);
    el.style.height = `${newHeight}px`;

    // Hysteresis thresholds
    const expandThreshold = 60;   // must exceed this to expand
    const collapseThreshold = 44; // must go below this to collapse

    setIsExpanded((prev) => {
        if (prev) {
            // Already expanded → only collapse if clearly below collapseThreshold
            return newHeight > collapseThreshold;
        } else {
            // Not expanded yet → only expand if clearly above expandThreshold
            return newHeight > expandThreshold;
        }
    });
}, [value]);



    // --- SWIPE HANDLERS ---
    const handleTouchStart = (e) => {
        touchStartY.current = e.touches[0].clientY;
    };

    const handleTouchEnd = (e) => {
        const touchEndY = e.changedTouches[0].clientY;
        const swipeDistance = touchStartY.current - touchEndY;

        // Swipe up (at least 30px) shows labels
        if (swipeDistance > 30) {
            setShowLabels(true);

            // Clear any existing timeout
            if (hideTimeout.current) {
                clearTimeout(hideTimeout.current);
            }

            // Auto-hide after 3 seconds
            hideTimeout.current = setTimeout(() => {
                setShowLabels(false);
            }, 3000);
        }
        // Swipe down hides labels
        else if (swipeDistance < -30) {
            setShowLabels(false);
            if (hideTimeout.current) {
                clearTimeout(hideTimeout.current);
            }
        }
    };

    return (
        <motion.div
            layout
            transition={{ type: "spring", bounce: 0, duration: 0.4 }}
            className={`
        flex
        ${isExpanded ? "flex-col items-stretch gap-2 py-3 px-4 rounded-3xl"
                    : "flex-row items-center py-3 px-5 rounded-full"}
        bg-[#181818]
        shadow-[0_10px_30px_rgba(0,0,0,0.7)]
        w-full max-w-4xl
        ${disabled ? "opacity-60 pointer-events-none" : ""}
        ${className}
    `}

            onTouchStart={handleTouchStart}
            onTouchEnd={handleTouchEnd}
            onMouseEnter={() => setShowLabels(true)}
            onMouseLeave={() => setShowLabels(false)}
        >
            {/* TEXTAREA */}
            <motion.textarea
                layout
                ref={textareaRef}
                value={value}
                placeholder={placeholder}
                disabled={disabled}
                // TODO: Fix this
                onChange={(e) => {
                    // IMPORTANT: send only the text value
                    if (onChange) onChange(e.target.value);
                }}
                className={`
                    bg-transparent
                    text-[16px] font-normal
                    tracking-wide
                    text-white
                    placeholder:text-gray-500
                    outline-none
                    resize-none
                    max-h-[144px]
                    leading-relaxed
                    ${isExpanded ? "w-full mb-2" : "flex-1 pr-2"}
                `}
                // If expanded, take full width (w-full). If not, share space (flex-1).
                rows={1}
            />

            {/* ACTION BUTTONS */}
            <motion.div
                layout
                className={`
                flex items-center gap-2
                ${isExpanded ? "w-full justify-end mt-1" : "ml-0"}
                ${hideActions ? "opacity-0 pointer-events-none" : ""}
                `}
            >
                {/* 2. MIC BUTTON */}
                <IconButton
                    label="Voice"
                    onClick={onMicClick}
                    showLabel={showLabels}
                >
                    <IconVoicePrompt className="w-5 h-5 opacity-80" />
                </IconButton>

                {/* 3. SEND BUTTON */}
                <button
                    onClick={() => {
                        if (setIsProcessingAnim) {
                            setIsProcessingAnim(true);
                            setTimeout(() => setIsProcessingAnim(false), 5000);
                        }
                        if (onSendClick) {
                            onSendClick();
                        }
                    }}
                    className="
                        flex flex-col items-center gap-0.5
                        px-3 py-1
                        rounded-xl
                        bg-[#FE5959]
                        shadow-[0_10px_20px_rgba(0,0,0,0.6)]
                        hover:bg-[#ff6e6e] active:scale-95
                        transition-all
                    "
                >
                    <IconPrompt className="w-5 h-5" />
                    <span className={`text-[9px] text-white font-medium transition-all duration-200 ${showLabels ? 'opacity-100 max-h-4' : 'opacity-0 max-h-0 overflow-hidden'}`}>Send</span>
                </button>

                {/* 4. BACK / DROPDOWN BUTTON (CONDITIONAL) */}
                {showBackIcon && (
                    <IconButton
                        label="Back"
                        onClick={onBackClick}
                        showLabel={showLabels}
                        className="bg-[rgba(40,40,40,0.85)] border border-white/10 hover:border-[#FE5959]/40"
                    >
                        <svg
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            className="w-5 h-5 text-red-400"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                d="M19 9l-7 7-7-7"
                            />
                        </svg>
                    </IconButton>
                )}
            </motion.div>
        </motion.div>
    );
}