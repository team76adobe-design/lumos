import React, { useCallback } from "react";
import { IconCheck, IconClose } from "../../constants";

/**
 * Confirmation buttons for crop accept/cancel
 */
export const ConfirmButtons = ({ onConfirm, onCancel, show }) => {
  if (!show) return null;

  return (
    <div className="confirm-row">
      <button
        className="w-8 h-8 sm:w-9 sm:h-9 flex items-center justify-center hover:opacity-80"
        onClick={onCancel}
        aria-label="Cancel crop"
      >
        <IconClose />
      </button>
      <button
        className="w-8 h-8 sm:w-9 sm:h-9 flex items-center justify-center hover:opacity-80"
        onClick={onConfirm}
        aria-label="Confirm crop"
      >
        <IconCheck />
      </button>
    </div>
  );
};

/**
 * Overlay pieces that dim areas outside crop box
 */
export const DimOverlay = ({ overlayPieces }) => {
  if (!overlayPieces) return null;

  return (
    <>
      <div
        className="outside-dim piece"
        style={{
          left: overlayPieces.topRect.left,
          top: overlayPieces.topRect.top,
          width: overlayPieces.topRect.width,
          height: overlayPieces.topRect.height,
        }}
      />
      <div
        className="outside-dim piece"
        style={{
          left: overlayPieces.bottomRect.left,
          top: overlayPieces.bottomRect.top,
          width: overlayPieces.bottomRect.width,
          height: overlayPieces.bottomRect.height,
        }}
      />
      <div
        className="outside-dim piece"
        style={{
          left: overlayPieces.leftRect.left,
          top: overlayPieces.leftRect.top,
          width: overlayPieces.leftRect.width,
          height: overlayPieces.leftRect.height,
        }}
      />
      <div
        className="outside-dim piece"
        style={{
          left: overlayPieces.rightRect.left,
          top: overlayPieces.rightRect.top,
          width: overlayPieces.rightRect.width,
          height: overlayPieces.rightRect.height,
        }}
      />
    </>
  );
};

/**
 * Crop box handles (corners and edges)
 *
 * Behavior: when a handle is pressed (pointerdown/touch), we add the class
 * "handle-active" to the nearest parent .crop-box. That class can be used
 * to darken the border (see CSS snippet below).
 */
export const CropHandles = () => {
  // helper to toggle active class on nearest .crop-box
  const setParentActive = useCallback((el, active) => {
    if (!el || !el.closest) return;
    const parent = el.closest(".crop-box");
    if (!parent) return;
    if (active) parent.classList.add("handle-active");
    else parent.classList.remove("handle-active");
  }, []);

  // pointerdown -> set active and attach a window pointerup handler for safety
  const handlePointerDown = (e) => {
    // prevent default so text selection/drag doesn't interfere
    e.preventDefault();

    const target = e.currentTarget;
    setParentActive(target, true);

    // if user releases pointer outside the handle, ensure we clear active
    const onWindowPointerUp = () => {
      setParentActive(target, false);
      window.removeEventListener("pointerup", onWindowPointerUp);
      window.removeEventListener("touchend", onWindowPointerUp);
    };

    window.addEventListener("pointerup", onWindowPointerUp, { passive: true });
    window.addEventListener("touchend", onWindowPointerUp, { passive: true });
  };

  // pointerup on the handle itself -> clear active
  const handlePointerUp = (e) => {
    const target = e.currentTarget;
    setParentActive(target, false);
  };

  const corners = [
    { class: "nw", dir: "nw" },
    { class: "ne", dir: "ne" },
    { class: "sw", dir: "sw" },
    { class: "se", dir: "se" },
  ];

  const edges = [
    { class: "n", dir: "n" },
    { class: "s", dir: "s" },
    { class: "e", dir: "e" },
    { class: "w", dir: "w" },
  ];

  return (
    <>
      {corners.map((corner) => (
        <div
          key={corner.dir}
          className={`corner-handle ${corner.class}`}
          data-dir={corner.dir}
          role="button"
          tabIndex={0}
          onPointerDown={handlePointerDown}
          onPointerUp={handlePointerUp}
          onTouchStart={handlePointerDown}
          onTouchEnd={handlePointerUp}
        />
      ))}
      {edges.map((edge) => (
        <div
          key={edge.dir}
          className={`edge-handle ${edge.class}`}
          data-dir={edge.dir}
          role="button"
          tabIndex={0}
          onPointerDown={handlePointerDown}
          onPointerUp={handlePointerUp}
          onTouchStart={handlePointerDown}
          onTouchEnd={handlePointerUp}
        />
      ))}
    </>
  );
};

/**
 * Crop box background grid (9-cell grid)
 */
export const CropBoxGrid = () => {
  return (
    <div className="grid grid-cols-3 grid-rows-3 w-full h-full pointer-events-none">
      {Array.from({ length: 9 }).map((_, i) => (
        <div key={i} className="border border-white/10" />
      ))}
    </div>
  );
};
