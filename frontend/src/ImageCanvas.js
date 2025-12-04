import React from "react";
import PolygonHighlighter from "./components/PolygonHighlighter";

const TwinklingStars = () => {
  const STAR_COUNT = 100;
  return (
    <div className="absolute inset-0 pointer-events-none z-[95] overflow-hidden">
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

const ImageCanvas = ({
  imageSrc,
  isComparing = false,
  compareSrc = null,
  polygons,
  setPolygons,
  maskPath,
  setMaskPath,
  fgPoints,
  bgPoints,
  isSegmenting,
  onPointClick,
  onPointRemove,
  clickMode,
  moveStartPoint,
  moveEndPoint,
  activeTool,
  moveStage,
  activeMenu,
  maskSelectionMode,
  isBrushActive,
  brushSize,
  filterStyle = {},
  editValues = {},
  spotlightPolygons = false,
  isProcessingAnim = false,
  quillMode = null,
  quillPoints = [],
  setQuillPoints = null,
  penColor = "#ff0000",
  onDragEnd,
  onImageProcessed,
}) => {
  const displaySrc = isComparing && compareSrc ? compareSrc : imageSrc;

  // NO IMAGE -> upload prompt (constrained)
  if (!imageSrc) {
    return (
      <div className="w-full max-w-2xl h-auto flex-1 flex items-center justify-center overflow-hidden">
        <div className="w-full h-auto flex items-center justify-center px-4">
          <button
            onClick={() => document.querySelector('input[type="file"]').click()}
            className="flex flex-col items-center justify-center gap-4 px-6 py-8 border-2 border-dashed border-gray-600 rounded-lg hover:border-[#FE5959] transition-colors max-w-[90%]"
          >
            <svg
              className="w-16 h-16 text-gray-500"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 4v16m8-8H4"
              />
            </svg>
            <p className="text-white text-lg">Click to upload image</p>
            <p className="text-gray-400 text-sm">PNG, JPG up to 10MB</p>
          </button>
        </div>
      </div>
    );
  }

  // MAIN: Constrain everything to viewport height so mobile never scrolls vertically
  return (
    <div className="w-full max-w-2xl flex-1 flex items-center justify-center overflow-hidden">

      <div className="relative w-full h-full flex items-center justify-center">
        {/* LOADING / SEGMENTING OVERLAY */}
        {isSegmenting && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-50">
            <div className="absolute inset-0 z-[40] pointer-events-none transition-all duration-300 backdrop-blur-[6px]">
              <TwinklingStars />
            </div>
          </div>
        )}

        <div className="relative flex items-center justify-center w-full h-full overflow-hidden">
          <div className="relative flex items-center justify-center w-full h-full max-h-full">

            {/* move start/end markers (unchanged) */}
            {(activeTool === "move" || activeTool === "drag") && moveStartPoint && (
              <div
                className="absolute z-[60] w-4 h-4 rounded-full border-2 border-emerald-400 bg-emerald-300/80 pointer-events-none"
                style={{
                  left: moveStartPoint.viewX - 8,
                  top: moveStartPoint.viewY - 8,
                }}
              />
            )}
            {(activeTool === "move" || activeTool === "drag") && moveEndPoint && (
              <div
                className="absolute z-[60] w-4 h-4 rounded-full border-2 border-red-400 bg-red-300/80 pointer-events-none"
                style={{
                  left: moveEndPoint.viewX - 8,
                  top: moveEndPoint.viewY - 8,
                }}
              />
            )}

            {/* blur overlay when processing */}
            <div
              className={`absolute inset-0 z-[40] pointer-events-none transition-all duration-300 ${isProcessingAnim ? "backdrop-blur-[6px]" : "backdrop-blur-0"}`}
            >
              {isProcessingAnim && <TwinklingStars />}
            </div>

            {/* IMAGE WRAPPER: key constraints here */}
            <div className="relative flex items-center justify-center w-full h-full max-h-[100vh] overflow-hidden">
              {/* PolygonHighlighter should render inner <img> or <canvas> with these classes/styles:
                  className="w-full h-auto max-h-[100vh] object-contain"
              */}
              <PolygonHighlighter
                baseSrc={displaySrc}
                polygons={polygons}
                setPolygons={setPolygons}
                maskPath={maskPath}
                setMaskPath={setMaskPath}
                fgPoints={fgPoints}
                bgPoints={bgPoints}
                onPointClick={onPointClick}
                onPointRemove={onPointRemove}
                clickMode={clickMode}
                filterStyle={filterStyle}
                editValues={editValues}
                spotlightPolygons={spotlightPolygons}
                activeMenu={activeMenu}
                activeTool={activeTool}
                maskSelectionMode={maskSelectionMode}
                quillMode={quillMode}
                quillPoints={quillPoints}
                setQuillPoints={setQuillPoints}
                penColor={penColor}
                isBrushActive={isBrushActive}
                brushSize={brushSize}
                onDragEnd={onDragEnd}
                onImageProcessed={onImageProcessed}
              />
            </div>

          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageCanvas;
