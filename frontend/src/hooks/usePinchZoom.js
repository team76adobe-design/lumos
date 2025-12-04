// hooks/usePinchZoom.js
import { useEffect, useRef } from "react";

/**
 * usePinchZoom
 * - containerRef: ref to element listening for pointer/wheel events
 * - transform: current transform object { x, y, scale, rotation, flipX, flipY }
 * - setTransform: setter (setState) for transform
 *
 * This hook uses Pointer Events for robust multi-touch pinch handling.
 */
export function usePinchZoom(containerRef, transform, setTransform) {
  const pointersRef = useRef(new Map());
  const pinchInfoRef = useRef(null); // { startDist, startScale, midpoint, startTransform }

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    // Ensure we can prevent default on touchmove:
    // Tailwind `.touch-none` sets touch-action: none already but double-check
    el.style.touchAction = el.style.touchAction || "none";

    // Pointer handlers
    const onPointerDown = (ev) => {
      // Capture the pointer to receive move/cancel events reliably
      try { ev.target.setPointerCapture?.(ev.pointerId); } catch (e) {}
      pointersRef.current.set(ev.pointerId, { x: ev.clientX, y: ev.clientY, id: ev.pointerId });
      if (pointersRef.current.size === 2) {
        const pts = Array.from(pointersRef.current.values());
        const dx = pts[1].x - pts[0].x;
        const dy = pts[1].y - pts[0].y;
        const startDist = Math.hypot(dx, dy);
        const midpoint = { x: (pts[0].x + pts[1].x) / 2, y: (pts[0].y + pts[1].y) / 2 };

        pinchInfoRef.current = {
          startDist,
          startScale: transform.scale,
          midpoint,
          startTransform: { ...transform },
        };
      } else if (pointersRef.current.size === 1) {
        // store start transform for single pointer drag
        pinchInfoRef.current = {
          startTransform: { ...transform },
          singleStart: { x: ev.clientX, y: ev.clientY },
        };
      }
    };

    const onPointerMove = (ev) => {
      if (!pointersRef.current.has(ev.pointerId)) return;
      ev.preventDefault(); // important on mobile to stop scrolling
      pointersRef.current.set(ev.pointerId, { x: ev.clientX, y: ev.clientY, id: ev.pointerId });

      const count = pointersRef.current.size;
      if (count === 2) {
        const pts = Array.from(pointersRef.current.values());
        const dx = pts[1].x - pts[0].x;
        const dy = pts[1].y - pts[0].y;
        const dist = Math.hypot(dx, dy);
        const midpoint = { x: (pts[0].x + pts[1].x) / 2, y: (pts[0].y + pts[1].y) / 2 };

        const info = pinchInfoRef.current;
        if (!info) return;
        const scaleFactor = dist / info.startDist;
        const newScale = Math.max(0.05, info.startScale * scaleFactor);

        // Adjust translate so that midpoint remains visually stable:
        // formula: newPos = midpoint + (oldPos - midpoint) * (newScale/oldScale)
        const start = info.startTransform;
        const oldScale = info.startScale;
        const mx = midpoint.x;
        const my = midpoint.y;

        const deltaX = (start.x - mx) * (newScale / oldScale) - (start.x - mx);
        const deltaY = (start.y - my) * (newScale / oldScale) - (start.y - my);

        setTransform((prev) => ({
          ...prev,
          scale: newScale,
          x: start.x - deltaX,
          y: start.y - deltaY,
        }));
      } else if (count === 1) {
        // single finger drag to move image
        const info = pinchInfoRef.current;
        if (!info || !info.singleStart) return;
        const ptr = Array.from(pointersRef.current.values())[0];
        const dx = ptr.x - info.singleStart.x;
        const dy = ptr.y - info.singleStart.y;
        const s = info.startTransform;
        setTransform((prev) => ({ ...prev, x: s.x + dx, y: s.y + dy }));
      }
    };

    const onPointerUpOrCancel = (ev) => {
      pointersRef.current.delete(ev.pointerId);
      pinchInfoRef.current = null;
      try { ev.target.releasePointerCapture?.(ev.pointerId); } catch (e) {}
    };

    // Wheel handler for desktop zoom (centered at mouse)
    const onWheel = (ev) => {
      // zoom factor
      const delta = -ev.deltaY;
      const zoomIntensity = 0.0015; // tune to taste
      const factor = 1 + delta * zoomIntensity;
      if (factor === 1) return;

      ev.preventDefault();

      const rect = el.getBoundingClientRect();
      const mx = ev.clientX;
      const my = ev.clientY;

      setTransform((prev) => {
        const oldScale = prev.scale;
        const newScale = Math.max(0.05, oldScale * factor);

        // Keep mouse point stable
        const deltaX = (prev.x - mx) * (newScale / oldScale) - (prev.x - mx);
        const deltaY = (prev.y - my) * (newScale / oldScale) - (prev.y - my);

        return {
          ...prev,
          scale: newScale,
          x: prev.x - deltaX,
          y: prev.y - deltaY,
        };
      });
    };

    // Add listeners with passive: false where needed
    el.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("pointermove", onPointerMove, { passive: false });
    window.addEventListener("pointerup", onPointerUpOrCancel);
    window.addEventListener("pointercancel", onPointerUpOrCancel);
    el.addEventListener("wheel", onWheel, { passive: false });

    return () => {
      el.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUpOrCancel);
      window.removeEventListener("pointercancel", onPointerUpOrCancel);
      el.removeEventListener("wheel", onWheel);
      pointersRef.current.clear();
      pinchInfoRef.current = null;
    };
  }, [containerRef, setTransform, transform]); // transform included so startScale uses latest

}
