import React from "react";
import { Heart } from "lucide-react";
import { icons } from "../../constants";
import AddSubtractMenu from "./AddSubtractMenu";
// import ActionMenu from "../../ActionMenu";
import ActionMenu from "./ActionMenu";
import CropMenu from "./CropMenu";
import AspectRatioMenu from "./AspectRatioMenu";
import ToolsMenu from "./ToolsMenu";
import SelectionMenu from "./SelectionMenu";
import SliderControl from "../SliderControl";
import ElasticSlider from "../ElasticSlider";
import GeminiMenu from "./GeminiMenu";
import Toast from "../Toast"; // Added Toast import
import { loadSam } from "../../api/segmentAPI";
import QuillMenu from "./QuillMenu";
import ColorSliders from "../MagicQuill/ColorSliders";

const MenuBar = ({
  activeMenu,
  setActiveMenu,
  showActiveMenu,
  setShowActiveMenu,
  isAnimating,
  setIsAnimating,
  selectedMainIcon,
  setSelectedMainIcon,
  handleMainIconClick,
  handleCloseMaskSelection,
  maskSelectionMode,
  setMaskSelectionMode,
  handleCloseActionMenu,
  onerase,
  onmove,
  ondrag,
  editIcons,
  activeEditControl,
  setActiveEditControl,
  getProgressPercentage,
  CircularProgress,
  handleAutoAdjust,
  handleSliderChange,
  selectedTool,
  setSelectedTool,
  onBrushClick,
  onCropClick,
  onQuillClick,
  editValues,
  selectedSelectionTool,
  setSelectedSelectionTool,
  showSelectionMenu,
  handleOpenSelectionMenu,
  handleCloseSelectionMenu,
  backgroundImage,
  onCropCancel,
  onAspectRatioSelect,
  onRotate,
  onFlipX,
  onFlipY,
  isProcessingAnim,
  setIsProcessingAnim,
  onClearSelection,
  isBrushActive,
  setIsBrushActive,
  brushSize,
  setBrushSize,
  brushColor,
  setBrushColor,
  activeTool,
  setActiveTool,
  geminiInput,
  setGeminiInput,
  handleGeminiSubmit,
  isGeminiLoading,
  handleSave,
  handleReset,
  handleDownload,
  quillMode,
  setQuillMode,
  onQuillDone,
  onQuillCancel,
  onUpscale,
  penColor,
  setPenColor,
  onAnalyzeClick,
  isAnalyzing,
  onGeminiPromptSubmit,
  onCommitAdjustments
}) => {
  const [isClosing, setIsClosing] = React.useState(false);
  const [prevEditControl, setPrevEditControl] =
    React.useState(activeEditControl);
  const [showBottomBar, setShowBottomBar] = React.useState(false);
  const [geminiText, setGeminiText] = React.useState("");
  const [showQuillColorPanel, setShowQuillColorPanel] = React.useState(false);
  const [isQuillColorPanelClosing, setIsQuillColorPanelClosing] = React.useState(false);
  const [showBrushSlider, setShowBrushSlider] = React.useState(false);
  const [isBrushSliderClosing, setIsBrushSliderClosing] = React.useState(false);

  const [showAspectMenu, setShowAspectMenu] = React.useState(false);

  // --- SWIPE TO SHOW LABELS ---
  const [showLabels, setShowLabels] = React.useState(false);
  const touchStartY = React.useRef(0);
  const hideTimeout = React.useRef(null);

  const handleTouchStart = (e) => {
    touchStartY.current = e.touches[0].clientY;
  };

  const handleTouchEnd = (e) => {
    const touchEndY = e.changedTouches[0].clientY;
    const swipeDistance = touchStartY.current - touchEndY;

    if (swipeDistance > 30) {
      setShowLabels(true);
      if (hideTimeout.current) clearTimeout(hideTimeout.current);
      hideTimeout.current = setTimeout(() => setShowLabels(false), 3000);
    } else if (swipeDistance < -30) {
      setShowLabels(false);
      if (hideTimeout.current) clearTimeout(hideTimeout.current);
    }
  };

  // --- TOAST STATE ---
  const [toastMessage, setToastMessage] = React.useState("");
  const [showToast, setShowToast] = React.useState(false);
  const [toastKey, setToastKey] = React.useState(0);

  // --- HEARTS STATE ---
  const [hearts, setHearts] = React.useState([]);

  // --- TOAST HELPER ---
  const [toastVariant, setToastVariant] = React.useState(null);
  const triggerToast = (message, variant = null) => {
    setShowToast(false);
    setToastMessage(message);
    setToastVariant(variant);
    setToastKey((prev) => prev + 1);
    setTimeout(() => {
      setShowToast(true);
    }, 0);
  };

  // Track when to show bottom bar
  React.useEffect(() => {
    // Only show bottom bar when a specific edit control is selected (not just when edit mode opens)
    if (activeEditControl && activeEditControl !== "menu_open") {
      setShowBottomBar(true);
    } else {
      setShowBottomBar(false);
    }

    if (activeEditControl) {
      setPrevEditControl(activeEditControl);
    }
  }, [activeEditControl, prevEditControl]);
  /* ------------------------------------------------------------ */
  /* ANIMATION WRAPPER                                            */
  /* ------------------------------------------------------------ */
  const animateMenuTransition = (callback) => {
    if (isAnimating) return;

    setIsAnimating(true);
    setShowActiveMenu(false);

    setTimeout(() => {
      callback();
      // Small delay to ensure content switch happens while collapsed
      setTimeout(() => {
        setShowActiveMenu(true);
        setTimeout(() => {
          setIsAnimating(false);
        }, 500);
      }, 50);
    }, 500);
  };

  /* ------------------------------------------------------------ */
  /* MAIN ICON CLICK HANDLER                                      */
  /* ------------------------------------------------------------ */
  const handleMainIconClick_Internal = (iconId) => {
    // --- Heart Animation Trigger ---
    if (iconId === "heart") {
      const newHeart = {
        id: Date.now(),
        offset: Math.random() * 40 - 20,
      };

      setHearts((prev) => [...prev, newHeart]);

      setTimeout(() => {
        setHearts((prev) => prev.filter((h) => h.id !== newHeart.id));
      }, 1000);

      return;
    }

    if (iconId === "tools") {
      triggerToast("Tools");
      animateMenuTransition(() => {
        setSelectedMainIcon(null);
        setActiveMenu("tools");
      });
      return;
    }

    if (iconId === "star") {
      // triggerToast("Gemini"); // Trigger Toast
      animateMenuTransition(() => {
        setSelectedMainIcon("star");
        setActiveMenu("star");
      });
      return;
    }

    if (iconId === "adjust") {
      if (activeEditControl) {
        // Closing logic
        setIsClosing(true);
        
        // Commit adjustments when closing via toggle
        if (onCommitAdjustments) {
          onCommitAdjustments();
        }

        setTimeout(() => {
          setActiveEditControl(null);
          setIsClosing(false);
          setShowActiveMenu(false);
          setIsAnimating(true);

          setTimeout(() => {
            setIsAnimating(false);
            setShowActiveMenu(true);
          }, 400);
        }, 250);
      } else {
        // Opening logic
        triggerToast("Adjustment"); // Trigger Toast
        setShowActiveMenu(false);
        setIsAnimating(true);

        setTimeout(() => {
          setActiveEditControl("menu_open");
          setIsAnimating(false);

          setTimeout(() => {
            setShowActiveMenu(true);
          }, 200);
        }, 400);
      }
      return;
    }

    if (iconId === "action") {
      triggerToast("Actions", "actions"); // Trigger Toast (actions variant)
      loadSam(); // Load SAM model
      animateMenuTransition(() => {
        setSelectedMainIcon("action");
        setActiveMenu("addSubtract");
      });
      return;
    }

    if (iconId === "magic") {
      // Toggle magic menu
      if (activeEditControl === "magic") {
        setActiveEditControl(null);
      } else {
        setActiveEditControl("magic");
      }
    } else if (iconId === "quill") {
      // Toggle quill menu
      triggerToast("Quill");
      animateMenuTransition(() => {
        setSelectedMainIcon("quill");
        setActiveMenu("magic");
      });
    } else {
      handleMainIconClick(iconId);
    }
  };

  /* ------------------------------------------------------------ */
  /* EDITING CONTROLS (Brightness / Contrast / etc.)              */
  /* ------------------------------------------------------------ */

  const handleCloseEditing = () => {
    setIsClosing(true);
    setShowActiveMenu(false);
    setIsAnimating(true);

    // Commit adjustments when closing
    if (onCommitAdjustments) {
      onCommitAdjustments();
    }

    setTimeout(() => {
      setShowBottomBar(false);
      setActiveEditControl(null);
      setIsClosing(false);
      setPrevEditControl(null);

      setTimeout(() => {
        setIsAnimating(false);
        setShowActiveMenu(true);
      }, 50);
    }, 500);
  };

  const renderEditingControls = () => {
    if (activeEditControl === "magic") {
      return (
        <div className="absolute bottom-20 left-1/2 transform -translate-x-1/2 bg-[#1e1e1e] rounded-xl p-2 flex gap-2 shadow-lg border border-[#333]">
          <ColorSliders
            color={brushColor}
            onChange={(color) => setBrushColor(color.hex)}
          />
        </div>
      );
    }

    if (activeMenu !== "main") {
      return null;
    }

    if (activeEditControl) {
      return (
        <div className="flex items-center w-full py-2 rounded-3xl">
          {/* AUTO BUTTON */}
          <button
            onClick={() => {
              triggerToast("auto"); // Trigger Toast
              handleAutoAdjust();
            }}
            className="flex-shrink-0 flex items-center justify-center w-12 h-12 text-white hover:text-[#FE5959] transition-colors ml-6"
            title="Auto-adjust image"
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              className="w-6 h-6"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M13 10V3L4 14h7v7l9-11h-7z"
              />
            </svg>
          </button>

          {/* SCROLLABLE ICONS */}
          <div className="flex items-center gap-3 overflow-x-auto no-scrollbar px-2 flex-1">
            {editIcons.map((editIcon) => {
              const currentValue = editValues[editIcon.id] ?? editIcon.default;
              const isModified = Number(currentValue) !== editIcon.default;
              return (
                <button
                  key={editIcon.id}
                  onClick={() => {
                    triggerToast(editIcon.label);
                    setActiveEditControl(editIcon.id);
                    setShowBottomBar(true);
                  }}
                  className="flex-shrink-0 flex items-center justify-center py-2"
                >
                  <CircularProgress
                    progress={getProgressPercentage(editIcon.id)}
                    isActive={isModified}
                  >
                    <div className={activeEditControl === editIcon.id ? "text-[#FE5959]" : "text-white"}>
                      {editIcon.icon}
                    </div>
                  </CircularProgress>
                </button>
              );
            })}
          </div>

          {/* BACK BUTTON */}
          <button
            onClick={handleCloseEditing}
            className="flex-shrink-0 flex items-center justify-center w-11 h-11 rounded-full bg-[rgba(40,40,40,0.8)] border border-white/10 text-white hover:text-[#FE5959] transition-all mr-6"
            title="Back to main menu"
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              className="w-5 h-5"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M19 9l-7 7-7-7"
              />
            </svg>
          </button>
        </div>
      );
    }

    // Otherwise, show the main icon bar
    return (
      // center the icon group and keep it uniform with other menus
      <div
        className="flex items-center justify-center gap-4 lg:gap-6 w-full px-6 overflow-x-auto no-scrollbar"
        onTouchStart={handleTouchStart}
        onTouchEnd={handleTouchEnd}
        onMouseEnter={() => setShowLabels(true)}
        onMouseLeave={() => setShowLabels(false)}
      >
        {icons.map((iconConfig) => {
          const IconComponent = iconConfig.icon;
          const isActive =
            (iconConfig.id === "adjust" && activeEditControl) ||
            selectedMainIcon === iconConfig.id;

          return (
            <button
              key={iconConfig.id}
              onClick={() => handleMainIconClick_Internal(iconConfig.id)}
              className="flex flex-col items-center justify-center gap-1 min-w-[56px] hover:opacity-70 transition-all"
            >
              {/* normalize icon size and ensure it's centered visually */}
              <IconComponent
                className="w-6 h-6 block"
                strokeWidth={2}
                color={isActive ? "#FE5959" : "white"}
              />
              <span
                className={`
                  text-[10px] font-medium transition-all duration-200
                  ${isActive ? "text-[#FE5959]" : "text-gray-400"}
                  ${showLabels ? 'opacity-100 max-h-4' : 'opacity-0 max-h-0 overflow-hidden'}
                `}
              >
                {iconConfig.label}
              </span>
            </button>
          );
        })}
      </div>
    );
  };


  /* ------------------------------------------------------------ */
  /* TOOLS MENU                                                   */
  /* ------------------------------------------------------------ */
  const renderToolsMenu = () => {
    if (activeMenu !== "tools") return null;

    return (
      <ToolsMenu
        onBackToMain={() =>
          animateMenuTransition(() => {
            setSelectedMainIcon(null);
            setActiveMenu("main");
          })
        }
        selectedTool={selectedTool}
        setSelectedTool={setSelectedTool}
        onBrushClick={onBrushClick}
        onCropClick={() => {
          animateMenuTransition(() => {
            onCropClick();
          });
        }}
        onQuillClick={() => {
          animateMenuTransition(() => {
            onQuillClick();
          });
        }}
        onUpscale={onUpscale}
        onSelectionClick={handleOpenSelectionMenu}
      />
    );
  };

  /* ------------------------------------------------------------ */
  /* SELECTION MENU                                               */
  /* ------------------------------------------------------------ */
  const renderSelectionMenu = () => {
    if (activeMenu !== "selection") return null;

    return (
      <SelectionMenu
        onBackToTools={() =>
          animateMenuTransition(() => {
            setActiveMenu("tools");
          })
        }
        selectedSelectionTool={selectedSelectionTool}
        setSelectedSelectionTool={setSelectedSelectionTool}
        backgroundImage={backgroundImage}
        onImageSelected={(file) => {
          // console.log("file");
        }}
      />
    );
  };

  /* ------------------------------------------------------------ */
  /* GEMINI MENU (AI Text Input)                                  */
  /* ------------------------------------------------------------ */
  const renderGeminiMenu = () => {
    if (
      !(activeMenu === "star" && selectedMainIcon === "star")
    )
      return null;

    return (
      <GeminiMenu
        value={geminiText}
        onChange={(text) => setGeminiText(text)}
        onSend={() => {
          // console.log("Gemini prompt sent:", geminiText);
          if (onGeminiPromptSubmit && geminiText.trim()) {
            onGeminiPromptSubmit(geminiText);
          }
          setGeminiText("");
        }}
        onBackToMain={() =>
          animateMenuTransition(() => {
            if (onClearSelection) onClearSelection();
            setSelectedMainIcon(null);
            setActiveMenu("main");
          })
        }
        placeholder="Type your idea..."
        isProcessingAnim={isProcessingAnim}
        setIsProcessingAnim={setIsProcessingAnim}
        onAnalyzeClick={onAnalyzeClick}
        isAnalyzing={isAnalyzing}
      />
    );
  };

  /* ------------------------------------------------------------ */
  /* ADD / SUBTRACT MENU                                          */
  /* ------------------------------------------------------------ */
  const renderAddSubtractMenu = () => {
    if (
      !(
        activeMenu === "addSubtract" &&
        selectedMainIcon !== "tools" &&
        selectedMainIcon !== "star"
      )
    )
      return null;

    return (
      <AddSubtractMenu
        onBackToMain={() => {
          if (onClearSelection) onClearSelection();
          setIsBrushActive(false);
          animateMenuTransition(() => {
            setSelectedMainIcon(null);
            setActiveMenu("main");
          });
        }}
        subSelection={maskSelectionMode}
        setSubSelection={setMaskSelectionMode}
        isBrushActive={isBrushActive}
        setIsBrushActive={setIsBrushActive}
      />
    );
  };

  /* ------------------------------------------------------------ */
  /* SECOND MENU (erase / move)                                 */
  /* ------------------------------------------------------------ */
  const renderActionMenu = () => {
    if (activeMenu !== "second") return null;

    return (
      <ActionMenu
        onClose={() => {
          if (onClearSelection) onClearSelection();
          handleCloseActionMenu();
        }}
        onerase={onerase}
        onmove={onmove}
        ondrag={ondrag}
      />
    );
  };

  /* ------------------------------------------------------------ */
  /* CROP MENU                                                    */
  /* ------------------------------------------------------------ */
  const renderCropMenu = () => {
    if (activeMenu !== "crop") return null;

    return (
      <CropMenu
        onClose={() => {
          animateMenuTransition(() => {
            onCropCancel();
            setActiveMenu("tools");
          });
        }}
        onAspectRatioClick={() => {
          animateMenuTransition(() => {
            setActiveMenu("aspectRatio");
          });
        }}
        onRotate={onRotate}
        onFlipX={onFlipX}
        onFlipY={onFlipY}
      />
    );
  };

  const handleCloseQuillColorPanel = () => {
    setIsQuillColorPanelClosing(true);
    setTimeout(() => {
      setShowQuillColorPanel(false);
      setIsQuillColorPanelClosing(false);
    }, 600);
  };

  React.useEffect(() => {
    if (activeMenu === "magic" && quillMode !== 'color' && showQuillColorPanel && !isQuillColorPanelClosing) {
      handleCloseQuillColorPanel();
    }
  }, [activeMenu, quillMode, showQuillColorPanel, isQuillColorPanelClosing]);

  React.useEffect(() => {
    const shouldShow = (isBrushActive && activeMenu === "addSubtract") || (activeMenu === "magic" && (quillMode === 'add' || quillMode === 'subtract'));

    if (shouldShow) {
      if (!showBrushSlider) {
        setShowBrushSlider(true);
        setIsBrushSliderClosing(false);
      } else if (isBrushSliderClosing) {
        setIsBrushSliderClosing(false);
      }
    } else {
      if (showBrushSlider && !isBrushSliderClosing) {
        setIsBrushSliderClosing(true);
        setTimeout(() => {
          setShowBrushSlider(false);
          setIsBrushSliderClosing(false);
        }, 600);
      }
    }
  }, [isBrushActive, activeMenu, quillMode, showBrushSlider, isBrushSliderClosing]);

  const renderBottomPanel = () => {
    if (activeMenu === "magic" && (showQuillColorPanel || isQuillColorPanelClosing)) {
      return (
        <div className={`w-full flex justify-center -mt-1 ${isQuillColorPanelClosing ? "animate-bottomBarClose" : "animate-slideDown"}`}>
          <div
            className="
            w-full bg-[rgba(18,18,18,0.95)]
            backdrop-blur-md border border-white/5 border-t-0
            rounded-b-3xl shadow-xl shadow-black/50 px-6 py-4
          "
          >
            <ColorSliders
              color={penColor}
              onChange={(color) => setPenColor(color)}
            />
          </div>
        </div>
      );
    }

    // ---------- DEFAULT SLIDER PANEL ----------
    if (!showBottomBar || isClosing) return null;

    return (
      <div
        className={`w-full flex justify-center -mt-1 ${isClosing ? "animate-bottomBarClose" : "animate-slideDown"
          }`}
      >
        <div
          className="w-full bg-[rgba(18,18,18,0.95)] backdrop-blur-md
                    border border-white/5 border-t-0 rounded-b-3xl
                    shadow-xl shadow-black/50 px-6 py-4"
        >
          <SliderControl
            activeEditControl={prevEditControl || activeEditControl}
            editValues={editValues}
            handleSliderChange={handleSliderChange}
            onBack={handleCloseEditing}
          />
        </div>
      </div>
    );
  };

  /* ------------------------------------------------------------ */
  /* MAGIC QUILL MENU                                             */
  /* ------------------------------------------------------------ */
  const renderQuillMenu = () => {
    if (activeMenu !== "magic") return null;

    return (
      <QuillMenu
        onCancel={() => {
          if (showQuillColorPanel) {
            handleCloseQuillColorPanel();
          } else {
            if (onQuillCancel) onQuillCancel();
            animateMenuTransition(() => {
              if (selectedMainIcon === "quill") {
                setSelectedMainIcon(null);
                setActiveMenu("main");
              } else {
                setActiveMenu("tools");
              }
            });
          }
        }}
        onOpenColorPanel={() => setShowQuillColorPanel(true)}
        penColor={penColor}
        activeTool={activeTool}
        setActiveTool={setActiveTool}
        brushSize={brushSize}
        setBrushSize={setBrushSize}
        brushColor={brushColor}
        setBrushColor={setBrushColor}
        quillMode={quillMode}
        setQuillMode={setQuillMode}
        onDone={onQuillDone}
      />
    );
  };

  /* ------------------------------------------------------------ */
  /* ASPECT RATIO MENU                                            */
  /* ------------------------------------------------------------ */
  const renderAspectRatioMenu = () => {
    if (activeMenu !== "aspectRatio") return null;

    return (
      <AspectRatioMenu
        onSelect={(ratio) => {
          onAspectRatioSelect?.(ratio);
        }}
        onClose={() => {
          animateMenuTransition(() => {
            setActiveMenu("crop");
          });
        }}
      />
    );
  };

  /* ------------------------------------------------------------ */
  /* BRUSH SLIDER                                                 */
  /* ------------------------------------------------------------ */
  const renderBrushSlider = () => {
    if (!showBrushSlider && !isBrushSliderClosing) return null;

    const BOTTOM_BAR_HEIGHT = "py-4";
    const BOTTOM_BAR_HORIZONTAL = "px-6";
    const BOTTOM_BAR_MARGIN_TOP = "-mt-1";
    const BOTTOM_BAR_BORDER_RADIUS = "rounded-b-3xl";

    return (
      <div
        className={`w-full flex justify-center ${BOTTOM_BAR_MARGIN_TOP} ${isBrushSliderClosing ? "animate-bottomBarClose" : "animate-slideDown"}`}
      >
        <div
          className={`w-full bg-[rgba(18,18,18,0.95)] backdrop-blur-md border border-white/5 border-t-0 ${BOTTOM_BAR_BORDER_RADIUS} shadow-xl shadow-black/50 ${BOTTOM_BAR_HORIZONTAL} ${BOTTOM_BAR_HEIGHT}`}
        >
          <div className="w-full px-4 py-2">
            <div className="flex justify-between text-xs text-gray-400 mb-2">
              <span>Brush Size</span>
              <span>{Math.round(brushSize)}px</span>
            </div>
            <ElasticSlider
              value={brushSize}
              min={1}
              max={100}
              onChange={setBrushSize}
            />
          </div>
        </div>
      </div>
    );
  };

  /* ------------------------------------------------------------ */
  /* SLIDER CONTROL (shows below menu bar when edit icon clicked) */
  /* ------------------------------------------------------------ */
  const renderSliderControl = () => {
    if (!showBottomBar && !isClosing) {
      return null;
    }

    const BOTTOM_BAR_HEIGHT = "py-4";
    const BOTTOM_BAR_HORIZONTAL = "px-6";
    const BOTTOM_BAR_MARGIN_TOP = "-mt-1";
    const BOTTOM_BAR_BORDER_RADIUS = "rounded-b-3xl";

    return (
      <div
        className={`w-full flex justify-center ${BOTTOM_BAR_MARGIN_TOP} ${isClosing ? "animate-bottomBarClose" : "animate-slideDown"
          }`}
      >
        <div
          className={`w-full bg-[rgba(18,18,18,0.95)] backdrop-blur-md border border-white/5 border-t-0 ${BOTTOM_BAR_BORDER_RADIUS} shadow-xl shadow-black/50 ${BOTTOM_BAR_HORIZONTAL} ${BOTTOM_BAR_HEIGHT}`}
        >
          <SliderControl
            activeEditControl={prevEditControl || activeEditControl}
            editValues={editValues}
            handleSliderChange={handleSliderChange}
            onBack={handleCloseEditing}
          />
        </div>
      </div>
    );
  };

  /* ------------------------------------------------------------ */
  /* MAIN RENDER WRAPPER                                          */
  /* ------------------------------------------------------------ */
  return (
    <div className="flex flex-col items-center justify-center pt-0 pb-5">
      <style>
        {`
            @keyframes flutterUp {
              0% { transform: translateY(0) scale(1) translateX(0); opacity: 1; }
              25% { transform: translateY(-30px) scale(1.2) translateX(-10px); opacity: 0.9; }
              50% { transform: translateY(-60px) scale(1) translateX(10px); opacity: 0.8; }
              75% { transform: translateY(-90px) scale(1.1) translateX(-5px); opacity: 0.5; }
              100% { transform: translateY(-120px) scale(0) translateX(0); opacity: 0; }
            }
            .animate-float-heart {
              animation: flutterUp 1s ease-out forwards;
            }
            .transition-bouncy {
              transition-timing-function: cubic-bezier(0.34, 1.56, 0.64, 1);
            }
            .no-scrollbar::-webkit-scrollbar { display: none; }
            .no-scrollbar { scrollbar-width: none; }
          `}
      </style>

      <div className="w-full px-2 flex flex-col justify-center items-center rounded-3xl relative">
        {/* --- RENDER TOAST --- */}
        <Toast
          key={toastKey}
          message={toastMessage}
          isVisible={showToast}
          variant={toastVariant}
          onComplete={() => {
            setShowToast(false);
            setToastVariant(null);
          }}
        />

        {/* --- RENDER FLOATING HEARTS --- */}
        {hearts.map((heart) => (
          <div
            key={heart.id}
            className="absolute z-50 pointer-events-none text-[#FE5959] animate-float-heart"
            style={{
              bottom: "80px",
              left: `calc(50% + ${heart.offset}px)`,
            }}
          >
            <Heart fill="currentColor" className="w-6 h-6" />
          </div>
        ))}

        {/* Main Menu Bar */}
        {/* Main Menu Bar */}
        <div
          className={`
              flex items-center justify-center
              transition-[width,height,border-radius,transform,background-color,border-color] duration-700 transition-bouncy
              ${activeMenu === "star" ? "overflow-visible bg-transparent border-none shadow-none" : "overflow-hidden border border-white/5 bg-[rgba(18,18,18,0.95)] backdrop-blur-md shadow-xl shadow-black/50"}
              ${!showActiveMenu ? "w-16 scale-y-75" : "w-full scale-y-100"}
              ${activeMenu === "star" ? "h-auto min-h-[4rem]" : "h-16"}
              ${(showBottomBar && !isClosing) || isBrushActive ? "rounded-t-3xl" : "rounded-full"}
            `}
          style={{ transformOrigin: "center center" }}
        >
          <div
            className={`
      min-w-[340px] w-full flex justify-center items-center h-full 
      transition-opacity duration-200 
      ${!showActiveMenu ? "opacity-0" : "opacity-100"}
    `}
          >
            {renderEditingControls()}
            {renderToolsMenu()}
            {renderSelectionMenu()}
            {renderGeminiMenu()}
            {renderAddSubtractMenu()}
            {renderActionMenu()}
            {renderCropMenu()}
            {renderAspectRatioMenu()}
            {renderQuillMenu()}
          </div>
        </div>

        {renderBottomPanel()}
        {renderBrushSlider()}
        {/* {renderSliderControl()} */}
      </div>
    </div>
  );
};

export default MenuBar;
