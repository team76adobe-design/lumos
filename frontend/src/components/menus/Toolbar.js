import React, { useEffect, useRef, useState } from "react";
import {
  IconUndo,
  IconRedo,
  IconSearchFooter,
  IconSplit,
  IconClose,
  IconCheck,
  IconAnalyze,
} from "../../constants";
import {
  Wrench,
  SquareDashedMousePointer,
  Heart,
  SlidersHorizontal,
  Sparkles,
  Crop,
  PaintbrushVertical,
  ImageUpscale,
  Feather,
} from "lucide-react";
import { IconBlend } from "../../constants.js";

const DEFAULT_TOOLS = [
  { id: "tools", icon: Wrench, label: "Tools" },
  { id: "adjust", icon: SlidersHorizontal, label: "Adjust" },
  { id: "star", icon: Sparkles, label: "AI" },
  { id: "action", icon: SquareDashedMousePointer, label: "Select" },
  { id: "crop", icon: Crop, label: "Crop" },
  { id: "select", icon: IconBlend, label: "Blend" },
  { id: "brush", icon: PaintbrushVertical, label: "Style Transfer" },
  { id: "magic", icon: Feather, label: "Magic Quill" },
  { id: "upscale", icon: ImageUpscale, label: "Upscale" },
];

const Toolbar = ({
  activeMenu,
  handleConfirmMenu,
  activeTool, // optional controlled prop from parent (can pass selectedMainIcon here)
  currentOperation,
  moveStage,
  onUndo,
  onRedo,
  canUndo,
  canRedo,
  onCompareStart,
  onCompareEnd,
  onToolSelect,
  handleMainIconClick,
  tools = DEFAULT_TOOLS,
  onAnalyzeClick,
  isAnalyzing,
  showCropConfirm,
  onCropConfirm,
  onCropReset,
}) => {
  const [searchOpen, setSearchOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [filtered, setFiltered] = useState(tools);
  const [highlightIdx, setHighlightIdx] = useState(0);
  const inputRef = useRef(null);
  const panelRef = useRef(null);

  // explicit "no active" start
  const [internalActive, setInternalActive] = useState(null);

  // sync from parent if provided (null is valid meaning "no active")
  useEffect(() => {
    if (typeof activeTool !== "undefined") {
      setInternalActive(activeTool === null ? null : activeTool);
    }
  }, [activeTool]);

  const currentActive = activeTool ?? internalActive;

  useEffect(() => {
    const q = query.trim().toLowerCase();
    setFiltered((tools || []).filter((t) => t.label.toLowerCase().includes(q)));
    setHighlightIdx(0);
  }, [query, tools]);

  // highlight current active when opening search (only if non-null)
  // run after filtered updates so highlight index matches what the user sees
  useEffect(() => {
    if (!searchOpen) return;
    if (currentActive != null) {
      const idx = filtered.findIndex((t) => t.id === currentActive);
      if (idx >= 0) {
        setHighlightIdx(idx);
        return;
      }
    }
    // fallback to 0 (or -1 if you prefer none highlighted)
    setHighlightIdx(0);
  }, [filtered, searchOpen, currentActive]);

  // close on outside click
  useEffect(() => {
    const onDoc = (e) => {
      if (
        searchOpen &&
        panelRef.current &&
        !panelRef.current.contains(e.target)
      ) {
        setSearchOpen(false);
      }
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [searchOpen]);

  const openSearch = () => setSearchOpen(true);
  const closeSearch = () => setSearchOpen(false);

  const handleKeyDown = (e) => {
    if (!searchOpen) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlightIdx((i) => Math.min(i + 1, Math.max(0, filtered.length - 1)));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlightIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      const item = filtered[highlightIdx];
      if (item) selectTool(item.id);
    } else if (e.key === "Escape") {
      e.preventDefault();
      closeSearch();
    }
  };

  const selectTool = (id) => {
    // Prefer calling the MenuBar handler you provided (handleMainIconClick_Internal).
    // This keeps selection behavior identical to clicking the main icon.
    if (typeof handleMainIconClick === "function") {
      handleMainIconClick(id);
    } else if (typeof onToolSelect === "function") {
      // backward compatible callback
      onToolSelect(id);
    } else {
      // fallback to internal UI-only active state
      setInternalActive(id);
    }
    closeSearch();
  };

  //
  // Robust hold-to-compare logic:
  // - Capture pointer when possible (setPointerCapture)
  // - Add document-level pointerup/pointercancel fallback so release always fires
  // - Clean up listeners on unmount
  //
  const pointerIdRef = useRef(null);
  const activeRef = useRef(false);

  useEffect(() => {
    // cleanup on unmount
    return () => {
      if (activeRef.current) {
        // ensure callback ends
        try {
          if (typeof onCompareEnd === "function") onCompareEnd();
        } catch (e) {}
        activeRef.current = false;
        pointerIdRef.current = null;
      }
      // remove any global listeners (defensive)
      document.removeEventListener("pointerup", handleGlobalPointerUp);
      document.removeEventListener("pointercancel", handleGlobalPointerCancel);
      document.removeEventListener("touchend", handleGlobalTouchEnd);
      document.removeEventListener("touchcancel", handleGlobalTouchEnd);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // we declare handlers as function declarations so they can be referenced for add/remove
  function handleGlobalPointerUp(e) {
    // only end if we previously started
    if (!activeRef.current) return;
    // if pointerIdRef is set, ensure it matches; otherwise accept any
    if (pointerIdRef.current == null || pointerIdRef.current === e.pointerId) {
      activeRef.current = false;
      pointerIdRef.current = null;
      if (typeof onCompareEnd === "function") onCompareEnd();
      // remove global listeners
      document.removeEventListener("pointerup", handleGlobalPointerUp);
      document.removeEventListener("pointercancel", handleGlobalPointerCancel);
    }
  }

  function handleGlobalPointerCancel(e) {
    if (!activeRef.current) return;
    activeRef.current = false;
    pointerIdRef.current = null;
    if (typeof onCompareEnd === "function") onCompareEnd();
    document.removeEventListener("pointerup", handleGlobalPointerUp);
    document.removeEventListener("pointercancel", handleGlobalPointerCancel);
  }

  function handleGlobalTouchEnd() {
    if (!activeRef.current) return;
    activeRef.current = false;
    pointerIdRef.current = null;
    if (typeof onCompareEnd === "function") onCompareEnd();
    document.removeEventListener("touchend", handleGlobalTouchEnd);
    document.removeEventListener("touchcancel", handleGlobalTouchEnd);
  }

  const onSplitPointerDown = (e) => {
    // prevent text selection / default behavior
    try {
      e.preventDefault();
    } catch (err) {}

    // Start comparing
    if (typeof onCompareStart === "function") onCompareStart();
    activeRef.current = true;

    // attempt to capture pointer on the target element
    try {
      if (typeof e.pointerId !== "undefined" && e.currentTarget?.setPointerCapture) {
        pointerIdRef.current = e.pointerId;
        e.currentTarget.setPointerCapture(e.pointerId);
        // ensure we release listeners on pointerup on document as fallback
        document.addEventListener("pointerup", handleGlobalPointerUp);
        document.addEventListener("pointercancel", handleGlobalPointerCancel);
      } else {
        // fallback: add document-level touch/mouse listeners
        document.addEventListener("pointerup", handleGlobalPointerUp);
        document.addEventListener("pointercancel", handleGlobalPointerCancel);
        document.addEventListener("touchend", handleGlobalTouchEnd);
        document.addEventListener("touchcancel", handleGlobalTouchEnd);
      }
    } catch (err) {
      // If setPointerCapture fails, use doc-level listeners
      document.addEventListener("pointerup", handleGlobalPointerUp);
      document.addEventListener("pointercancel", handleGlobalPointerCancel);
      document.addEventListener("touchend", handleGlobalTouchEnd);
      document.addEventListener("touchcancel", handleGlobalTouchEnd);
    }
  };

  const onSplitPointerUp = (e) => {
    // release pointer capture if previously captured on the element
    try {
      if (typeof e.pointerId !== "undefined" && e.currentTarget?.releasePointerCapture) {
        try {
          e.currentTarget.releasePointerCapture(e.pointerId);
        } catch (_) {}
      }
    } catch (_) {}

    // End comparing if we started
    if (!activeRef.current) return;
    activeRef.current = false;
    pointerIdRef.current = null;

    if (typeof onCompareEnd === "function") onCompareEnd();

    // cleanup listeners (defensive)
    document.removeEventListener("pointerup", handleGlobalPointerUp);
    document.removeEventListener("pointercancel", handleGlobalPointerCancel);
    document.removeEventListener("touchend", handleGlobalTouchEnd);
    document.removeEventListener("touchcancel", handleGlobalTouchEnd);
  };

  const onSplitPointerCancel = (e) => {
    if (!activeRef.current) return;
    activeRef.current = false;
    pointerIdRef.current = null;
    if (typeof onCompareEnd === "function") onCompareEnd();
    document.removeEventListener("pointerup", handleGlobalPointerUp);
    document.removeEventListener("pointercancel", handleGlobalPointerCancel);
    document.removeEventListener("touchend", handleGlobalTouchEnd);
    document.removeEventListener("touchcancel", handleGlobalTouchEnd);
  };

  // keyboard support: space / enter to toggle compare while key is held
  const onSplitKeyDown = (e) => {
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      if (!activeRef.current) {
        if (typeof onCompareStart === "function") onCompareStart();
        activeRef.current = true;
      }
    }
  };
  const onSplitKeyUp = (e) => {
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      if (activeRef.current) {
        activeRef.current = false;
        if (typeof onCompareEnd === "function") onCompareEnd();
      }
    }
  };

  return (
    <div className="flex flex-col items-center mb-1 gap-1 w-full px-4">
      {/* Instruction text */}
      <div className="text-center min-h-[20px]">
        {currentOperation === "drag" && (
          <p className="text-white/70 text-xs">Drag arrow to change perspective</p>
        )}
        {currentOperation === "move" && (
          <p className="text-white/70 text-xs">Move the selected region</p>
        )}
        {!currentOperation && activeMenu === "addSubtract" && (
          <p className="text-white/50 text-[10px]">
            Click icon again to remove
          </p>
        )}
      </div>

      <div className="flex items-center justify-between w-full mb-2 gap-2">
        {/* Left group: undo / redo */}
        <div className="flex items-center gap-4 sm:gap-6">
          <button
            className={`w-8 h-8 sm:w-9 sm:h-9 flex items-center justify-center hover:opacity-80 ${
              !canUndo ? "opacity-40 pointer-events-none" : ""
            }`}
            onClick={onUndo}
            title="Undo"
            aria-label="Undo"
          >
            <IconUndo />
          </button>
          <button
            className={`w-8 h-8 sm:w-9 sm:h-9 flex items-center justify-center hover:opacity-80 ${
              !canRedo ? "opacity-40 pointer-events-none" : ""
            }`}
            onClick={onRedo}
            title="Redo"
            aria-label="Redo"
          >
            <IconRedo />
          </button>
        </div>

        {/* Center group: X / ✓ only in AddSubtract mode */}
        <div className="flex items-center gap-4 sm:gap-6">
          {activeMenu === "addSubtract" && (
            <>
              <button className="w-8 h-8 sm:w-9 sm:h-9 flex items-center justify-center hover:opacity-80">
                <IconClose />
              </button>
              <button
                className="w-8 h-8 sm:w-9 sm:h-9 flex items-center justify-center hover:opacity-80"
                onClick={handleConfirmMenu}
              >
                <IconCheck />
              </button>
            </>
          )}
          {activeMenu === "crop" && showCropConfirm && (
            <>
              <button
                className="w-8 h-8 sm:w-9 sm:h-9 flex items-center justify-center hover:opacity-80"
                onClick={onCropReset}
              >
                <IconClose />
              </button>
              <button
                className="w-8 h-8 sm:w-9 sm:h-9 flex items-center justify-center hover:opacity-80"
                onClick={onCropConfirm}
              >
                <IconCheck />
              </button>
            </>
          )}
          {activeMenu === "star" && onAnalyzeClick && (
            <button
              className={`w-8 h-8 sm:w-9 sm:h-9 flex items-center justify-center hover:opacity-80 ${
                isAnalyzing ? "opacity-50 cursor-not-allowed" : ""
              }`}
              onClick={onAnalyzeClick}
              disabled={isAnalyzing}
              title="Analyze"
            >
              {isAnalyzing ? (
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <IconAnalyze className="w-5 h-5 opacity-80" />
              )}
            </button>
          )}
        </div>

        {/* Right group: search / split */}
        <div className="flex items-center gap-4 sm:gap-6 relative">
          {/* SEARCH / TOOLS POPUP */}
          <div className="relative">
            <button
              className="w-8 h-8 sm:w-9 sm:h-9 flex items-center justify-center hover:opacity-80"
              onClick={() => (searchOpen ? closeSearch() : openSearch())}
              aria-haspopup="dialog"
              aria-expanded={searchOpen}
              title="Search tools"
            >
              <IconSearchFooter />
            </button>

            {searchOpen && (
              <div
                ref={panelRef}
                onKeyDown={handleKeyDown}
                className="absolute right-0 bottom-full mb-2 w-64 max-h-72 overflow-hidden rounded-lg bg-[rgba(18,18,18,0.98)] border border-white/5 shadow-xl z-50"
                role="dialog"
                aria-label="Tool search"
              >
                <div className="p-3">
                  <input
                    ref={inputRef}
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Search tools..."
                    className="w-full px-3 py-2 rounded-md bg-black/60 text-white placeholder-white/40 outline-none border border-white/6"
                    aria-label="Search tools"
                    autoComplete="off"
                  />
                </div>

                <div className="max-h-56 overflow-y-auto divide-y divide-white/5">
                  {filtered.length === 0 && (
                    <div className="px-3 py-4 text-sm text-white/60">
                      No tools found
                    </div>
                  )}

                  {filtered.map((t, idx) => {
                    const IconComp = t.icon;
                    const isHighlighted = idx === highlightIdx;
                    const isActive =
                      currentActive !== null && currentActive === t.id;
                    return (
                      <button
                        key={t.id}
                        onMouseEnter={() => setHighlightIdx(idx)}
                        onClick={() => selectTool(t.id)}
                        className={`w-full text-left flex items-center gap-3 px-3 py-3 hover:bg-white/5 transition ${
                          isHighlighted ? "bg-white/6" : ""
                        }`}
                        type="button"
                      >
                        <div className="w-8 h-8 flex items-center justify-center rounded-md bg-white/5">
                          <IconComp />
                        </div>
                        <div className="flex-1">
                          <div className="text-white text-sm">{t.label}</div>
                          <div className="text-white/40 text-xs">{t.id}</div>
                        </div>
                        {isActive && (
                          <div className="text-[#FE5959] text-sm">Active</div>
                        )}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </div>

          {/* SPLIT (hold to compare) */}
          <button
            className="w-8 h-8 sm:w-9 sm:h-9 flex items-center justify-center hover:opacity-80"
            title="Hold to compare original"
            // pointer handlers — robust via pointer capture / document fallback
            onPointerDown={onSplitPointerDown}
            onPointerUp={onSplitPointerUp}
            onPointerCancel={onSplitPointerCancel}
            // touch fallbacks (some browsers dispatch touch events)
            onTouchStart={(e) => {
              // forward to pointerdown handler-like behavior
              onSplitPointerDown(e);
            }}
            onTouchEnd={(e) => {
              onSplitPointerUp(e);
            }}
            onTouchCancel={(e) => {
              onSplitPointerCancel(e);
            }}
            // keyboard
            onKeyDown={onSplitKeyDown}
            onKeyUp={onSplitKeyUp}
            aria-pressed={activeRef.current ? "true" : "false"}
            aria-label="Hold to compare"
          >
            <IconSplit />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Toolbar;
