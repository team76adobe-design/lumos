import { useState } from "react";

/**
 * Centralized state + logic for managing the animated bottom menu system:
 * - Main Menu
 * - Add/Subtract Menu
 * - Second Menu
 * - Image Editing Controls (brightness, contrast, etc.)
 * - erase/move tools
 * - Animation (compress → switch → expand)
 */
export const useMenuState = () => {
  /* -------------------------------------------------------
   *  UI + Animation State
   * ----------------------------------------------------- */

  // Is the menu currently compressing into a circle?
  const [isAnimating, setIsAnimating] = useState(false);

  // Is the menu currently expanding after switching?
  const [isExpanding, setIsExpanding] = useState(false);

  // Which menu is currently displayed? ("main" | "addSubtract" | "second")
  const [activeMenu, setActiveMenu] = useState("main");

  // Controls visibility of the content inside the menu
  const [showActiveMenu, setShowActiveMenu] = useState(true);

  /* -------------------------------------------------------
   *  Main Menu (Icon Bar)
   * ----------------------------------------------------- */

  // Which of the main toolbar icons is selected (add, brush, adjust…)
  const [selectedMainIcon, setSelectedMainIcon] = useState(null);

  // Controls the "add" or "subtract" mode inside the Add/Subtract menu
  const [maskSelectionMode, setMaskSelectionMode] = useState("add");
  // const [activeEditControl, setActiveEditControl] = useState(false);

  // Brush toggle state
  const [isBrushActive, setIsBrushActive] = useState(false);
  const [brushSize, setBrushSize] = useState(20);

  /* -------------------------------------------------------
   *  Style & Tool States
   * ----------------------------------------------------- */

  // Tools like eraser, brush, transform, etc.
  const [selectedTool, setSelectedTool] = useState(null);

  // Selection tools state
  const [selectedSelectionTool, setSelectedSelectionTool] = useState(null);
  const [showSelectionMenu, setShowSelectionMenu] = useState(false);

  // Style sidebar (filters, presets, etc.)
  const [showStylesMenu, setShowStylesMenu] = useState(false);
  const [showQuillMenu, setShowQuillMenu] = useState(false);

  const [showCropUI, setShowCropUI] = useState(false);

  /* -------------------------------------------------------
   *  Second Menu: erase / move Tools
   * ----------------------------------------------------- */

  // Which tool is active in the second menu? ("erase" | "move")
  const [activeTool, setActiveTool] = useState(null);

  // move tool workflow ("idle" → "start" → "end")
  const [moveStage, setmoveStage] = useState("idle");

  // Start and end points for move region
  const [moveStartPoint, setmoveStartPoint] = useState(null);
  const [moveEndPoint, setmoveEndPoint] = useState(null);

  /* -------------------------------------------------------
   *  Image Editing Controls (Brightness, Contrast, etc.)
   * ----------------------------------------------------- */

  // Active editing control (e.g. "brightness", "contrast")
  const [activeEditControl, setActiveEditControl] = useState(null);

  /* -------------------------------------------------------
   *  Menu Transition Handler
   *  - Compress → Switch Menu → Expand
   * ----------------------------------------------------- */

  const runMenuTransition = (nextMenu) => {
    if (isAnimating) return;

    setShowActiveMenu(false); // fade out inner content
    setIsAnimating(true); // compress animation start

    // After compress finishes (~400ms)
    setTimeout(() => {
      setIsAnimating(false);
      setIsExpanding(true); // start expansion

      // After expand finishes (~300ms)
      setTimeout(() => {
        setActiveMenu(nextMenu);
        setShowActiveMenu(true);
        setIsExpanding(false);
      }, 300);
    }, 400);
  };

  /* -------------------------------------------------------
   *  Menu Navigation Functions
   * ----------------------------------------------------- */

  // Click main toolbar icon → go to appropriate menu
  const handleMainIconClick = (iconId) => {
    if (isAnimating) return;
    setActiveTool(null);
    setSelectedMainIcon(iconId);
    if (iconId === "tools") {
      setSelectedMainIcon("star");
      setActiveMenu("main");
      runMenuTransition("tools");
    } else if (iconId === "star") {
      setSelectedMainIcon("star");
      setActiveMenu("main");
      runMenuTransition("star");
      return;
    } else if (iconId === "adjust") {
      if (isAnimating) return;
      setShowActiveMenu(false);
      setIsAnimating(true);

      setTimeout(() => {
        setIsAnimating(false);
        setIsExpanding(true);

        setTimeout(() => {
          setActiveMenu("main");
          setSelectedMainIcon("adjust");
          setActiveEditControl("adjust");
          setShowActiveMenu(true);
          setIsExpanding(false);
        }, 300);
      }, 400);
      return;
    } else if (iconId === "styles") {
      return;
    } else if (iconId === "action") {
      setSelectedMainIcon("action");
      setActiveMenu("main");
      runMenuTransition("addSubtract");
      return;
    } else if (iconId === "crop") {
      setSelectedMainIcon("crop");
      setActiveMenu("crop");
      runMenuTransition("crop");
      return;
    } else if (iconId === "select") {
      setSelectedMainIcon("main");
      setActiveMenu("selection");
      runMenuTransition("selection");
      return;
    } else if (iconId === "brush") {
      setSelectedMainIcon("brush");
            setShowStylesMenu(true);

      return;
    } else if (iconId === "magic") {
      setSelectedMainIcon("magic");
      setActiveMenu("magic");
      runMenuTransition("magic");
      return;
    // } else if (iconId === "upscale") {
    //   setSelectedMainIcon("upscale");
    //   setActiveMenu("upscale");
    //   runMenuTransition("upscale");
   //   return;
    } else {
      runMenuTransition("addSubtract");
    }
  };

  // Return from Add/Subtract → back to main
  const handleCloseMaskSelection = () => {
    if (isAnimating) return;
    setSelectedMainIcon(null); // Reset selected icon when returning to main
    runMenuTransition("main");
  };

  // Open Second Menu (Tools)
  const handleOpenActionMenu = () => {
    if (isAnimating) return;
    runMenuTransition("second");
  };

  // confirm mask selection → open Second Menu
  const handleConfirmMenu = () => {
    if (isAnimating) return;
    setIsBrushActive(false);
    runMenuTransition("second");
  };

  // Close second menu → return to main
  const handleCloseActionMenu = () => {
    if (isAnimating) return;
    setActiveTool(null);
    runMenuTransition("main");
  };

  // Open Selection Menu from Tools
  const handleOpenSelectionMenu = () => {
    if (isAnimating) return;
    runMenuTransition("selection");
  };

  // Close Selection Menu → return to tools
  const handleCloseSelectionMenu = () => {
    if (isAnimating) return;
    runMenuTransition("tools");
  };

  const handleOpenCropMenu = () => {
    if (isAnimating) return;
    runMenuTransition("crop");
  };
  const handleCloseCropMenu = () => {
    if (isAnimating) return;
    runMenuTransition("tools"); // or main
  };

  const handleOpenQuillMenu = () => {
    if (isAnimating) return;
    runMenuTransition("magic");
  };
  const handleCloseQuillMenu = () => {
    if (isAnimating) return;
    runMenuTransition("tools");
  };

  /* -------------------------------------------------------
   *  Expose API to components
   * ----------------------------------------------------- */

  return {
    // Animation
    isAnimating,
    isExpanding,
    setIsAnimating,
    setIsExpanding,

    // Menu
    activeMenu,
    setActiveMenu,
    showActiveMenu,
    setShowActiveMenu,

    // Main menu icon selection
    selectedMainIcon,
    handleMainIconClick,
    handleMainIconClick_Internal: handleMainIconClick,

    // Mask modes
    maskSelectionMode,
    setMaskSelectionMode,
    handleCloseMaskSelection,
    isBrushActive,
    setIsBrushActive,
    brushSize,
    setBrushSize,

    // Style menu
    selectedTool,
    setSelectedTool,
    showStylesMenu,
    setShowStylesMenu,

    //Crop UI
    setShowCropUI,
    showCropUI,
    handleCloseCropMenu,
    handleOpenCropMenu,

    // Quill Menu
    setShowQuillMenu,
    showQuillMenu,
    handleCloseQuillMenu,
    handleOpenQuillMenu,

    // Selection menu
    selectedSelectionTool,
    setSelectedSelectionTool,
    showSelectionMenu,
    setShowSelectionMenu,
    handleOpenSelectionMenu,
    handleCloseSelectionMenu,

    // Second menu tools
    activeTool,
    setActiveTool,
    moveStage,
    setmoveStage,
    moveStartPoint,
    setmoveStartPoint,
    moveEndPoint,
    setmoveEndPoint,
    handleOpenActionMenu,
    handleConfirmMenu,
    handleCloseActionMenu,

    // Image editing controls
    activeEditControl,
    setActiveEditControl,
    setSelectedMainIcon,

    // Animation transition function
    runMenuTransition,
  };
};
