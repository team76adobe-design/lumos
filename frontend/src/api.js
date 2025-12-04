// API Configuration
import { resizeTo512, rescaleFrom512 } from "./utils/imageResize.js";
import { toast } from 'sonner';

export const API_URL = 'https://colby-imprecatory-jimply.ngrok-free.dev';

// ---- erase API (uses first polygon as mask) ----
export const runeraseFromMask = async (imageSrc, polygons, setters) => {
  console.log("runeraseFromMask input polygons:", polygons);
  const { setIsSegmenting, setImageSrc: setImage, setFgPoints, setBgPoints, setPolygons, setActiveMenu, setShowActiveMenu, setActiveTool } = setters;

  if (!imageSrc) {
    toast.error("Upload an image first.");
    return;
  }
  if (!polygons.length) {
    toast.error("No mask polygon found. Run segmentation first.");
    return;
  }

  try {
    setIsSegmenting(true);
    console.log("segment");
    const blob = await fetch(imageSrc).then((r) => r.blob());
    const file = new File([blob], "image.png", { type: blob.type });

    // Use all polygons instead of just the first one
    const biggestPolygon = polygons.reduce((maxPoly, current) => {
      return current.points.length > maxPoly.points.length ? current : maxPoly;
    });

    // Step B — Convert to coordinate array
    const allCoords = [
      biggestPolygon.points.map((p) => [p.x, p.y])
    ];
    // const allCoords = polygons.map(poly => poly.points.map((p) => [p.x, p.y]));

    console.log("erase All Polygon Coords:", allCoords);

    const formData = new FormData();
    formData.append("input_image", file);
    formData.append("polygon", JSON.stringify(allCoords));

    const res = await fetch(`${API_URL}/erase`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errText = await res.text();
      console.error("erase failed:", errText);
      toast.error(`Erase failed: ${errText}`);
      return;
    }

    const outBlob = await res.blob();
    const url = URL.createObjectURL(outBlob);

    // Replace image with erase result
    setImage(url);
    setFgPoints([]);
    setBgPoints([]);
    setPolygons([]);
    setActiveMenu("main");
    setShowActiveMenu(true);
    setActiveTool(null);
  } catch (err) {
    console.error(err);
    toast.error("Erase operation failed.");
  } finally {
    setIsSegmenting(false);
  }
};

export const runmoveFromMask = async (
  imageSrc,
  polygons,
  startPt,
  endPt,
  setters
) => {
  // console.log("runmoveFromMask input polygon:", polygon);
  const {
    setIsSegmenting,
    setImageSrc: setImage,
    setFgPoints,
    setBgPoints,
    setPolygons,
    setActiveTool,
    setmoveStage,
    setmoveStartPoint,
    setmoveEndPoint,
    setActiveMenu,
    setShowActiveMenu,
  } = setters;

  if (!imageSrc) return toast.error("Upload an image first.");
  // if (!polygon) return toast.error("Select a polygon first.");
  if (!startPt || !endPt) return toast.error("Pick start and end points.");

  try {
    setIsSegmenting(true);

    /* -----------------------------------------
     * STEP 1 — Get original file from imageSrc
     * ----------------------------------------- */
    const blob = await fetch(imageSrc).then((r) => r.blob());
    const originalFile = new File([blob], "image.png", { type: blob.type });

    /* -----------------------------------------
     * STEP 2 — FRONTEND resize to 512x512
     * (NO letterboxing, just stretch)
     * ----------------------------------------- */
    const {
      file: resizedFile,
      originalWidth,
      originalHeight,
    } = await resizeTo512(originalFile);

    /* -----------------------------------------
     * STEP 3 — Map original coords → 512x512
     * ----------------------------------------- */
    const scaleTo512 = (x, y) => ({
      x: Math.round((x / originalWidth) * 512),
      y: Math.round((y / originalHeight) * 512),
    });

    // Step A — Find polygon with max points
    const biggestPolygon = polygons.reduce((maxPoly, current) => {
      return current.points.length > maxPoly.points.length ? current : maxPoly;
    });

    // Step B — Convert to coordinate array
    const allCoords = [
      biggestPolygon.points.map((p) => [p.x, p.y])
    ];


    console.log("move All Polygon Coords:", allCoords);

    const scaledAllCoords = allCoords.map(poly => poly.map(([x, y]) => {
      const s = scaleTo512(x, y);
      return [s.x, s.y];
    }));

    console.log("move Scaled All Coords:", scaledAllCoords);

    const scaledStart = scaleTo512(startPt.imageX, startPt.imageY);
    const scaledEnd = scaleTo512(endPt.imageX, endPt.imageY);

    const maskStr = JSON.stringify(scaledAllCoords);
    const handleStr = JSON.stringify([[scaledStart.x, scaledStart.y]]);
    const targetStr = JSON.stringify([[scaledEnd.x, scaledEnd.y]]);

    /* -----------------------------------------
     * STEP 4 — Build form-data
     * ----------------------------------------- */
    const formData = new FormData();
    formData.append("input_image", resizedFile); // already 512x512
    formData.append("mask_coordinate", maskStr);
    formData.append("handle_points", handleStr);
    formData.append("target_points", targetStr);

    /* -----------------------------------------
     * STEP 5 — Call API
     * ----------------------------------------- */
    const res = await fetch(`${API_URL}/move_erase`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.text();
      console.error("move failed:", err);
      toast.error(`Move failed: ${err}`);
      return;
    }

    const outBlob512 = await res.blob(); // 512x512 result from backend

    /* -----------------------------------------
     * STEP 6 — Scale result back to original size
     * ----------------------------------------- */
    const finalBlob = await rescaleFrom512(
      outBlob512,
      originalWidth,
      originalHeight
    );
    const finalUrl = URL.createObjectURL(finalBlob);

    /* -----------------------------------------
     * STEP 7 — Update UI
     * ----------------------------------------- */
    setImage(finalUrl);
    setFgPoints([]);
    setBgPoints([]);
    setPolygons([]);

    setActiveTool(null);
    setmoveStage("idle");
    setmoveStartPoint(null);
    setmoveEndPoint(null);

    setActiveMenu("main");
    setShowActiveMenu(true);
  } catch (err) {
    console.error("Error calling move API:", err);
    toast.error("Move operation failed.");
  } finally {
    setIsSegmenting(false);
  }
};

// ---- SEGMENTATION API ----
export const runSegmentation = async (imageSrc, xs, ys, labels, setters) => {
  const { setPolygons, setIsSegmenting } = setters;

  try {
    if (!imageSrc || !xs.length) return;

    setIsSegmenting(true);

    const blob = await fetch(imageSrc).then(r => r.blob());
    const file = new File([blob], 'image.png', { type: blob.type });

    const formData = new FormData();
    formData.append('input_image', file);
    xs.forEach((v) => formData.append('xs', String(v)));
    ys.forEach((v) => formData.append('ys', String(v)));
    labels.forEach((v) => formData.append('labels', String(v)));

    const res = await fetch(`${API_URL}/segment`, {
      method: 'POST',
      body: formData,
    });

    const data = await res.json();
    if (data.status !== 'success' || !Array.isArray(data.mask_coordinates)) {
      toast.error('Segmentation failed.');
      return;
    }

    const newPolygons = data.mask_coordinates.map((contour) => ({
      points: contour.map(([x, y]) => ({ x, y })),
      stroke: 'lime',
    }));

    setPolygons(newPolygons);
    console.log(newPolygons);
  } catch (err) {
    console.error(err);
    toast.error(`Segmentation failed: ${err.message}`);
  } finally {
    setIsSegmenting(false);
  }
};