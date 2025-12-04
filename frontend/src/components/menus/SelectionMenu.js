import React, { useRef, useState } from "react";
import {
    Square,
    Circle,
    Lasso,
    Wand2,
    MousePointer2,
    Move3D,
    Sparkles,
    SquareDashed,
    Plus,
    ChevronDown,
} from "lucide-react";
import ImagePreviewPopup from "../ImagePreviewPopup";

const SelectionMenu = ({
    onBackToTools,
    selectedSelectionTool,
    setSelectedSelectionTool,
    onImageSelected, // optional: get selected file in parent
    backgroundImage, // background image from photoroom
}) => {
    const fileInputRef = useRef(null);
    const [showPopup, setShowPopup] = useState(false);
    const [selectedFile, setSelectedFile] = useState(null);

    const selectionTools = [
        { id: "rectangle", icon: Square, label: "Rectangle Select" },
        { id: "ellipse", icon: Circle, label: "Ellipse Select" },
        { id: "lasso", icon: Lasso, label: "Lasso Select" },
        { id: "magic", icon: Wand2, label: "Magic Select" },
        { id: "pointer", icon: MousePointer2, label: "Pointer Select" },
        { id: "marquee", icon: SquareDashed, label: "Marquee" },
        { id: "transform", icon: Move3D, label: "Transform" },
        { id: "smart", icon: Sparkles, label: "Smart Select" },
    ];

    const handlePlusClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        
        // Set file and show popup
        setSelectedFile(file);
        setShowPopup(true);
        
        // Call parent callback if provided
        onImageSelected?.(file);
        
        // Reset file input
        e.target.value = '';
    };

    const handleClosePopup = () => {
        setShowPopup(false);
        setSelectedFile(null);
    };

    return (
        <>
            <div className="flex items-center w-full py-2 px-6">

                {/* LEFT: Plus button */}
                <button
                    type="button"
                    onClick={handlePlusClick}
                    className="flex-shrink-0 flex items-center justify-center w-10 h-10 rounded-full bg-[#FE5959] text-white shadow-md hover:bg-[#fe5959]/90 transition-colors"
                >
                    <Plus size={18} />
                </button>

                {/* MIDDLE: Center text */}
                <div className="flex-1 flex justify-center">
                    <span className="text-sm text-gray-100 whitespace-nowrap">
                        Add image for blending
                    </span>
                </div>

                {/* RIGHT: Back button */}
                <button
                    type="button"
                    onClick={onBackToTools}
                    className="flex-shrink-0 flex items-center justify-center w-11 h-11 rounded-full bg-[rgba(40,40,40,0.8)] border border-white/10 text-white hover:text-[#FE5959] hover:border-[#FE5959]/30 transition-all"
                >
                    <ChevronDown className="w-5 h-5" />
                </button>

                {/* Hidden File Input */}
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleFileChange}
                />
            </div>

            {/* Image Preview Popup */}
            <ImagePreviewPopup
                isVisible={showPopup}
                imageFile={selectedFile}
                backgroundImage={backgroundImage}
                onClose={handleClosePopup}
            />
        </>
    );
};

export default SelectionMenu;
