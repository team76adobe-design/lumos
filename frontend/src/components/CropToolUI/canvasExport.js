import React from "react";
import { getRects } from "../../utils/cropToolUtils";
import { inferOutpaint } from "../../api/outpaintAPI";

/**
 * Photoshop-style canvas export for crop tool
 */
export const useCanvasExport = () => {
  const exportCropToCanvas = async (
    cropBoxRef,
    imgRef,
    containerRef,
    transform,
    callback
  ) => {
    const cropEl = cropBoxRef.current;
    const imgEl = imgRef.current;
    const contEl = containerRef.current;

    if (!cropEl || !imgEl || !contEl) return;

    const cropRect = cropEl.getBoundingClientRect();
    const contRect = contEl.getBoundingClientRect();
    const imgRect = imgEl.getBoundingClientRect();

    // Final output size (in device pixels)
    const outW = Math.round(cropRect.width);
    const outH = Math.round(cropRect.height);

    const canvas = document.createElement("canvas");
    canvas.width = outW;
    canvas.height = outH;

    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    // ctx.fillStyle = "black";
    // ctx.fillRect(0, 0, outW, outH);

    // ----------------------------------------
    // STEP 1: Compute where the image center is relative to crop box
    // ----------------------------------------

    // Image center in container coords
    const imgCenterX_cont =
      imgRect.left - contRect.left + imgRect.width / 2;
    const imgCenterY_cont =
      imgRect.top - contRect.top + imgRect.height / 2;

    // Image center relative to cropBox (canvas origin)
    const imgCenterX_canvas = imgCenterX_cont - (cropRect.left - contRect.left);
    const imgCenterY_canvas = imgCenterY_cont - (cropRect.top - contRect.top);

    // ----------------------------------------
    // STEP 2: Map CSS size → natural pixel size
    // ----------------------------------------
    const naturalW = imgEl.naturalWidth;
    const naturalH = imgEl.naturalHeight;

    // ----------------------------------------
    // STEP 3: Apply Photoshop-style transform pipeline
    // ----------------------------------------
    ctx.save();

    // Move to where image center should be on canvas
    ctx.translate(imgCenterX_canvas, imgCenterY_canvas);

    // Apply rotation
    ctx.rotate((transform.rotation * Math.PI) / 180);

    // Apply flip & zoom (scale)
    // We must account for the fact that the image is displayed at a different size
    // than its natural size (e.g. fitted to screen).
    const renderW = imgEl.offsetWidth || naturalW;
    const renderH = imgEl.offsetHeight || naturalH;
    
    const baseScaleX = renderW / naturalW;
    const baseScaleY = renderH / naturalH;

    ctx.scale(
      transform.scale * transform.flipX * baseScaleX,
      transform.scale * transform.flipY * baseScaleY
    );

    // Finally draw image centered at origin
    ctx.drawImage(
      imgEl,
      -naturalW / 2,
      -naturalH / 2,
      naturalW,
      naturalH
    );

    ctx.restore();

    // ----------------------------------------
    // STEP 4: Output
    // ----------------------------------------
    let finalURL = canvas.toDataURL("image/png");

    // Always check for outpainting (margin detection handled in API)
    try {
      const result = await inferOutpaint(finalURL, 1.0, "high quality, seamless", "blur, artifacts");
      if (result && result.image_base64) {
        finalURL = "data:image/png;base64," + result.image_base64;
      }
    } catch (e) {
      console.error("Outpaint failed", e);
    }

    if (typeof callback === "function") {
      callback(finalURL);
    } else {
      const w = window.open();
      w?.document.write(`<img src="${finalURL}" alt="cropped" />`);
    }
  };

  return { exportCropToCanvas };
};
