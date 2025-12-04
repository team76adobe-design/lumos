import React, { useState, useEffect, useRef, useCallback } from "react";
import Toolbar from "./components/menus/Toolbar";
import MenuBar from "./components/menus/MenuBar";
import ImageCanvas from "./ImageCanvas";
import Header from "./components/Header";
import { useLocation } from "react-router-dom";

// Hooks
import { useImageHandler } from "./hooks/useImageHandler";
import { useMenuState } from "./hooks/useMenuState";
import { useImageEditor } from "./hooks/useImageEditor";

// API
import { runInpaint } from "./api/inpaintAPI";
import { segment } from "./api/segmentAPI";
import { loadInpaintModel, runDragInpaint } from "./api/moveAPI";
import { loadDragModel, runDragModel } from "./api/dragAPI"; // Imported Drag API
import { processImageUpscale } from "./api/upscaleAPI";
import { processImageAnalysis } from "./api/analyzeAPI";
import { loadLeditsModel, runLedits, freeLeditsModel } from "./api/leditsAPI";
import { runMagicQuillFullCycle } from "./api/quillAPI";

// Utils
import { resizeTo512, rescaleFrom512 } from "./utils/imageResize";
import {
  generateBinaryMask,
  saveMaskLocally,
  generateVectorMask,
  generateBinaryMaskFile,
  generateBackendMask,
} from "./utils/maskUtils";

// UI components
import { editIcons, CircularProgress } from "./constants";
import StylesMenu from "./components/menus/StylesMenu";
import QuillMenu from "./components/menus/QuillMenu";
import CropToolUI from "./components/CropToolUI/CropUI";
import AnalysisResultPanel from "./components/AnalysisResultPanel";

export default function Edit() {
  const imageHandler = useImageHandler();
  const menuState = useMenuState();
  const imageEditor = useImageEditor();
  const location = useLocation();
  const cropRef = useRef(null);

  const { showCropUI, setShowCropUI } = menuState;
  const [isProcessingAnim, setIsProcessingAnim] = useState(false);

  // Quill brush state
  const [quillMode, setQuillMode] = useState(null); // 'add' | 'subtract' | 'color' | null
  const [quillPoints, setQuillPoints] = useState([]);
  const [penColor, setPenColor] = useState("#ff0000");

  // Analysis state
  const [showAnalysisPanel, setShowAnalysisPanel] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [analysisError, setAnalysisError] = useState(null);
  const [showCropConfirm, setShowCropConfirm] = useState(false);

  // comparing
  const [isComparing, setIsComparing] = useState(false);
  const [compareSrc, setCompareSrc] = useState(null);
  const [currentProcessedImage, setCurrentProcessedImage] = useState(null);

  useEffect(() => {
    if (imageHandler.imageSrc && !compareSrc) {
      setCompareSrc(imageHandler.imageSrc);
    }
  }, [imageHandler.imageSrc, compareSrc]);

  const handleCompareStart = () => setIsComparing(true);
  const handleCompareEnd = () => setIsComparing(false);

  // --------------- UNDO / REDO ----------------
  const undoStackRef = useRef([]);
  const redoStackRef = useRef([]);

  // Snapshot utility (keeps it small and robust)
  const snapshot = useCallback((obj) => {
    if (typeof structuredClone !== "undefined") return structuredClone(obj);
    return JSON.parse(JSON.stringify(obj));
  }, []);

  // takeSnapshot: currently captures only visual state (image/polygons/mask).
  // If you'd like to also capture menu/UI state, see the commented "menu" entry.
  const takeSnapshot = useCallback(() => {
    return snapshot({
      imageSrc: imageHandler.imageSrc,
      polygons: imageHandler.polygons,
      maskPath: imageHandler.maskPath,
      // OPTIONAL: capture menu UI state too (disabled by default)
      // menu: {
      //   activeMenu: menuState.activeMenu,
      //   selectedMainIcon: menuState.selectedMainIcon,
      //   activeTool: menuState.activeTool,
      //   moveStage: menuState.moveStage,
      // }
    });
  }, [imageHandler, menuState, snapshot]);

  const pushSnapshot = useCallback(() => {
    try {
      undoStackRef.current.push(takeSnapshot());
      // keep reasonable memory
      if (undoStackRef.current.length > 30) undoStackRef.current.shift();
      // clear redo when new action happens
      redoStackRef.current = [];
    } catch (e) {
      // console.warn("Failed to push snapshot", e);
    }
  }, [takeSnapshot]);

  // applySnapshot: restore only image-related state. Do NOT mutate menuState here.
  // If you later decide to restore UI state as part of undo/redo, uncomment the
  // section below and ensure takeSnapshot is capturing the menu fields.
  const applySnapshot = useCallback(
    (snap) => {
      if (!snap) return;
      try {
        // restore image, polygons, mask
        imageHandler.setImageSrc(snap.imageSrc || null);
        imageHandler.setPolygons(snap.polygons || []);
        imageHandler.setMaskPath(snap.maskPath || null);
      } catch (e) {
        // console.error("Failed to apply snapshot data:", e);
      }

      // IMPORTANT: Do not change menuState here — that causes the menubar to jump.
      // If you *do* want to restore menuState, capture it in takeSnapshot and
      // uncomment the block below:

      /*
      if (snap.menu) {
        menuState.setActiveMenu(snap.menu.activeMenu || "main");
        if (typeof snap.menu.selectedMainIcon !== "undefined") {
          menuState.setSelectedMainIcon(snap.menu.selectedMainIcon);
        }
        // restore tool and move state carefully
        menuState.setActiveTool(snap.menu.activeTool || null);
        menuState.setmoveStage(snap.menu.moveStage || "idle");
      }
      */
    },
    [imageHandler, menuState]
  );

  const setImageSrcWithSnapshot = useCallback(
    (newSrc) => {
      pushSnapshot();
      imageHandler.setImageSrc(newSrc);
    },
    [imageHandler, pushSnapshot]
  );

  const setPolygonsWithSnapshot = useCallback(
    (nextPolygons) => {
      pushSnapshot();
      imageHandler.setPolygons(nextPolygons);
    },
    [imageHandler, pushSnapshot]
  );

  const setMaskPathWithSnapshot = useCallback(
    (file) => {
      pushSnapshot();
      imageHandler.setMaskPath(file);
    },
    [imageHandler, pushSnapshot]
  );

  const handleUndo = useCallback(() => {
    if (!undoStackRef.current.length) return;
    try {
      // push current to redo
      redoStackRef.current.push(takeSnapshot());
      const prev = undoStackRef.current.pop();
      applySnapshot(prev);
    } catch (e) {
      // console.warn("Undo failed", e);
    }
  }, [takeSnapshot, applySnapshot]);

  const handleRedo = useCallback(() => {
    if (!redoStackRef.current.length) return;
    try {
      undoStackRef.current.push(takeSnapshot());
      const next = redoStackRef.current.pop();
      applySnapshot(next);
    } catch (e) {
      // console.warn("Redo failed", e);
    }
  }, [takeSnapshot, applySnapshot]);

  const canUndo = () => undoStackRef.current.length > 0;
  const canRedo = () => redoStackRef.current.length > 0;

  /* ------------------------------------------------------------
   * Handle composited image
   * ------------------------------------------------------------ */
  useEffect(() => {
    if (location.state?.compositedImage) {
      setImageSrcWithSnapshot(location.state.compositedImage);
      window.history.replaceState({}, document.title);
    }
  }, [location.state, imageHandler]);

  /* ------------------------------------------------------------
   * ERASE HANDLER
   * ------------------------------------------------------------ */
  const handleActionMenuerase = async () => {
    // console.log("🔧 Tool: Erase");
    menuState.setActiveTool("erase");

    try {
      imageHandler.setIsSegmenting(true);

      // console.log("📥 Original imageSrc:", imageHandler.imageSrc);

      // -------------------------------------------------------
      // 1. Load original image
      // -------------------------------------------------------
      const blob = await fetch(imageHandler.imageSrc).then((r) => r.blob());
      // console.log("📦 Original Blob:", blob);

      const originalFile = new File([blob], "image.png", { type: blob.type });
      // console.log("📄 Original File:", originalFile);

      // -------------------------------------------------------
      // 2. Resize to 512
      // -------------------------------------------------------
      const {
        file: resizedFile,
        originalWidth,
        originalHeight,
      } = await resizeTo512(originalFile);

      // console.log("🖼️ ResizedFile (512x512):", resizedFile);
      // console.log("📏 Original Width/Height:", originalWidth, originalHeight);

      // -------------------------------------------------------
      // 3. Create BINARY MASK FILE
      // -------------------------------------------------------
      // console.log("🧊 maskPath (incoming):", imageHandler.maskPath);

      let maskFileBlob = imageHandler.maskPath;
      // console.log("🧊 Original Mask Blob:", maskFileBlob);

      // Ensure maskFileBlob exists
      if (!maskFileBlob) {
        throw new Error("No mask found. Please create a mask first.");
      }

      // maskPath is already a File object from segmentation or brush, use it directly
      const maskFile = await generateBinaryMaskFile(maskFileBlob);
      // console.log("🧊 Final Binary Mask File:", maskFile);

      // -------------------------------------------------------
      // LOG MASK IMAGE
      // -------------------------------------------------------
      const maskURL = URL.createObjectURL(maskFile);
      // console.log("🖼️ Mask Image Preview URL:", maskURL);

      // Optional: show it in DOM while debugging
      // const img = new Image();
      // img.src = maskURL;
      // document.body.appendChild(img);

      // -------------------------------------------------------
      // 4. Run inpainting
      // -------------------------------------------------------
      // console.log("📤 Sending to runInpaint:", {
      //   resizedFile,
      //   maskFile,
      // });

      const result = await runInpaint(resizedFile, maskFile, "");
      // console.log("📥 Inpaint Result Raw:", result);

      // -------------------------------------------------------
      // 5. Handle output (Blob or Base64)
      // -------------------------------------------------------
      let finalUrl = null;

      if (result instanceof Blob) {
        // console.log("📦 Inpaint returned Blob:", result);

        const finalBlob = await rescaleFrom512(
          result,
          originalWidth,
          originalHeight
        );
        // console.log("📏 Final Blob Rescaled:", finalBlob);

        finalUrl = URL.createObjectURL(finalBlob);
      } else if (result.image_base64) {
        // console.log("🧬 Inpaint returned Base64");

        const res = await fetch(`data:image/png;base64,${result.image_base64}`);
        const b = await res.blob();

        const finalBlob = await rescaleFrom512(
          b,
          originalWidth,
          originalHeight
        );
        // console.log("📏 Final Blob Rescaled:", finalBlob);

        finalUrl = URL.createObjectURL(finalBlob);
      }

      // -------------------------------------------------------
      // 6. Update UI
      // -------------------------------------------------------
      // console.log("🎯 finalUrl:", finalUrl);

      if (finalUrl) {
        setImageSrcWithSnapshot(finalUrl);
        imageHandler.handleReset();
        menuState.setActiveMenu("main");
        menuState.setActiveTool(null);
      }
    } catch (error) {
      // console.error("❌ Erase failed:", error);
      alert("Erase failed.");
    } finally {
      imageHandler.setIsSegmenting(false);
    }
  };

  /* ------------------------------------------------------------
   * MOVE HANDLER
   * ------------------------------------------------------------ */
  const handleActionMenumove = async () => {
    // console.log("Tool: Move - Initializing");
    if (!imageHandler.polygons.length && !imageHandler.maskPath) {
      alert("No mask found. Run segmentation or use the brush first.");
      return;
    }

    menuState.setActiveTool("move");
    menuState.setmoveStage("start");
    menuState.setmoveStartPoint(null);
    menuState.setmoveEndPoint(null);

    // Pre-load model
    loadInpaintModel().catch((e) => 
      console.error("Model Load Error:", e)
  );
  };

  /* ------------------------------------------------------------
   * DRAG HANDLER (UPDATED)
   * ------------------------------------------------------------ */
  const handleActionMenudrag = async () => {
    // console.log("Tool: Drag - Initializing");
    if (!imageHandler.polygons.length && !imageHandler.maskPath) {
      alert("No mask found. Run segmentation or use the brush first.");
      return;
    }

    menuState.setActiveTool("drag");
    menuState.setmoveStage("start");
    menuState.setmoveStartPoint(null);
    menuState.setmoveEndPoint(null);

    // Pre-load Drag model
    loadDragModel().catch((e) => console.error("Drag Model Load Error:", e));
  };

  /* ------------------------------------------------------------
   * ANALYZE HANDLER
   * ------------------------------------------------------------ */
  const handleActionMenuanalyze = async () => {
    // console.log("Tool: Analyze - Starting image analysis");

    if (!imageHandler.imageSrc) {
      alert("No image loaded. Please upload an image first.");
      return;
    }

    // Show the panel and start loading
    setShowAnalysisPanel(true);
    setAnalysisLoading(true);
    setAnalysisError(null);
    setAnalysisResult(null);

    try {
      // Convert current image to File
      const blob = await fetch(imageHandler.imageSrc).then((r) => r.blob());
      const imageFile = new File([blob], "image.png", { type: blob.type });

      // console.log("Analyzing image:", imageFile);

      // Run analysis
      const result = await processImageAnalysis(imageFile, 3);
      // console.log("Analysis result:", result);

      setAnalysisResult(result);
    } catch (error) {
      // console.error("Analysis failed:", error);
      setAnalysisError(error.message || "Failed to analyze image");
    } finally {
      setAnalysisLoading(false);
    }
  };

  /* ------------------------------------------------------------
   * GEMINI PROMPT HANDLER (Inpaint if mask, LEDITS if no mask)
   * ------------------------------------------------------------ */
  const handleGeminiPromptSubmit = async (prompt) => {
    // console.log("Gemini prompt submitted:", prompt);

    if (!imageHandler.imageSrc) {
      // console.error("No image loaded");
      return;
    }

    if (!prompt || prompt.trim() === "") {
      // console.error("Empty prompt");
      return;
    }

    const hasMask =
      imageHandler.maskPath ||
      (imageHandler.polygons && imageHandler.polygons.length > 0);

    try {
      imageHandler.setIsSegmenting(true);

      // Get original image dimensions
      const blob = await fetch(imageHandler.imageSrc).then((r) => r.blob());
      const originalFile = new File([blob], "image.png", { type: blob.type });
      const {
        file: resizedFile,
        originalWidth,
        originalHeight,
      } = await resizeTo512(originalFile);

      if (hasMask) {
        // --- INPAINT PATH (mask exists) ---
        // console.log("Mask detected - using INPAINT API with prompt:", prompt);

        let maskFileBlob = imageHandler.maskPath;
        if (!maskFileBlob) {
          throw new Error("No mask found.");
        }

        const maskFile = await generateBinaryMaskFile(maskFileBlob);
        // console.log("Binary mask generated for inpaint");

        const result = await runInpaint(resizedFile, maskFile, prompt);
        // console.log("Inpaint result:", result);

        let finalUrl = null;
        if (result instanceof Blob) {
          const finalBlob = await rescaleFrom512(
            result,
            originalWidth,
            originalHeight
          );
          finalUrl = URL.createObjectURL(finalBlob);
        } else if (result.image_base64) {
          const res = await fetch(
            `data:image/png;base64,${result.image_base64}`
          );
          const b = await res.blob();
          const finalBlob = await rescaleFrom512(
            b,
            originalWidth,
            originalHeight
          );
          finalUrl = URL.createObjectURL(finalBlob);
        }

        if (finalUrl) {
          setImageSrcWithSnapshot(finalUrl);
          imageHandler.handleReset();
          menuState.setActiveMenu("main");
        }
      } else {
        // --- LEDITS PATH (no mask) ---
        // console.log("No mask detected - using LEDITS API with prompt:", prompt);

        // Load LEDITS model
        await loadLeditsModel();
        // console.log("LEDITS model loaded");

        // Run LEDITS
        const result = await runLedits(imageHandler.imageSrc, prompt);
        // console.log("LEDITS result:", result);

        // Unload LEDITS model immediately after
        await freeLeditsModel();
        // console.log("LEDITS model freed");

        let finalUrl = null;
        if (result.blob) {
          const finalBlob = await rescaleFrom512(
            result.blob,
            originalWidth,
            originalHeight
          );
          finalUrl = URL.createObjectURL(finalBlob);
        } else if (result.image_base64) {
          const res = await fetch(
            `data:image/png;base64,${result.image_base64}`
          );
          const b = await res.blob();
          const finalBlob = await rescaleFrom512(
            b,
            originalWidth,
            originalHeight
          );
          finalUrl = URL.createObjectURL(finalBlob);
        }

        if (finalUrl) {
          setImageSrcWithSnapshot(finalUrl);
          menuState.setActiveMenu("main");
        }
      }
    } catch (error) {
      // console.error("Gemini prompt processing failed:", error);
    } finally {
      imageHandler.setIsSegmenting(false);
    }
  };

  const fileToBase64 = (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = () => {
        const result = reader.result;

        // Always force prefix
        if (result.startsWith("data:image")) {
          resolve(result); // already correct
        } else {
          resolve("data:image/png;base64," + result.split(",")[1]);
        }
      };

      reader.onerror = (err) => reject(err);
      reader.readAsDataURL(file);
    });


  /* ------------------------------------------------------------
   * Quill Brush Handlers
   * ------------------------------------------------------------ */
  const handleQuillDone = async () => {
    // console.log("=== QUILL BRUSH POINTS ===");
    // console.log("Total points:", quillPoints.length);
    // console.log("Points:", quillPoints);
    // console.log("========================");

    if (quillPoints.length === 0) {
      setQuillPoints([]);
      menuState.setShowQuillMenu(false);
      menuState.setActiveMenu("main");
      setQuillMode(null);
      return;
    }

    try {
      imageHandler.setIsSegmenting(true);

      // 1️⃣ Prepare original image
      const blob = await fetch(imageHandler.imageSrc).then((r) => r.blob());
      const originalFile = new File([blob], "image.png", { type: blob.type });
      const { file: resizedImage, originalWidth, originalHeight } = await resizeTo512(originalFile);

      const originalImageB64 = await fileToBase64(resizedImage);

      // 2️⃣ Split points
      const colorPoints = quillPoints.filter((p) => p.type === 2);
      const posPoints = quillPoints.filter((p) => p.type === 1);
      const negPoints = quillPoints.filter((p) => p.type === 0);

      // Helper: Convert DataURL → Blob
      const dataURLToBlob = async (dataURL) => await (await fetch(dataURL)).blob();

      // Helper: Create resized base64 masks
      const createB64Mask = async (points, useColor = false, drawBackground = false) => {
        if (points.length === 0) return "";

        const dataURL = await generateVectorMask(
          points,
          imageHandler.imageSrc,
          useColor,
          drawBackground
        );
        // console.log("Generated Mask DataURL:", dataURL);

        const blob = await dataURLToBlob(dataURL);
        const maskFile = await generateBinaryMaskFile(blob, 512, 512);
        // console.log("Mask File:", maskFile);
        return await fileToBase64(maskFile); // convert to pure base64
      };

      // Helper: Create backend-compatible mask (RGB=0, Alpha=Mask)
      const createBackendMask = async (points) => {
        // Pass original dimensions to draw accurately, then scale to 512x512
        const dataURL = await generateBackendMask(points, originalWidth, originalHeight, 512, 512);
        return await fileToBase64(await dataURLToBlob(dataURL));
      };


      // 3️⃣ Generate masks
      // total_mask: All points? Or just edge points?
      // User request: "total edge mask -> RGB channels 0, alpha channel mask"
      // Usually total mask includes all edits. Let's assume all edge points (pos + neg).
      // Or maybe all points including color?
      // Let's use all points for total mask for now, or just pos+neg if color is separate.
      // "total_mask" usually implies the region to be inpainted.
      // If we are adding color, that region also needs inpainting/generation.
      // So let's use ALL points.
      const totalMaskB64 = await createBackendMask(quillPoints);

      const addColorB64 =
        quillMode === "color" ? await createB64Mask(colorPoints, true, true) : "";

      const posMaskB64 = await createBackendMask(posPoints);
      const negMaskB64 = await createBackendMask(negPoints);

      // Ensure required field (backend expects remove_edge_image always)
      const removeEdge = negMaskB64 || "";
      const addEdge = posMaskB64 || "";

      // 4️⃣ Build FormData
      const formData = new FormData();
      formData.append("original_image", originalImageB64);
      formData.append("total_mask", totalMaskB64);
      formData.append("add_color_image", addColorB64); // This one is RGB+Alpha(1)
      formData.append("add_edge_image", addEdge);
      formData.append("remove_edge_image", removeEdge);
      formData.append("positive_prompt", "");
      formData.append("negative_prompt", "");

      // console.log("=== API PAYLOAD ===");
      // console.log("original_image:", originalImageB64 ? originalImageB64.slice(0, 50) + "..." : "null");
      // console.log("total_mask:", totalMaskB64 ? totalMaskB64.slice(0, 50) + "..." : "null");
      // console.log("add_color_image:", addColorB64 ? addColorB64.slice(0, 50) + "..." : "null");
      // console.log("add_edge_image:", addEdge ? addEdge.slice(0, 50) + "..." : "null");
      // console.log("remove_edge_image:", removeEdge ? removeEdge.slice(0, 50) + "..." : "null");
      // console.log("===================");

      // 5️⃣ Call MagicQuill full cycle
      const resultUrl = await runMagicQuillFullCycle({
        mode: "formdata",
        formData,
      });

      if (resultUrl) {
        // Rescale result back to original dimensions
        const resultBlob = await (await fetch(resultUrl)).blob();
        const finalBlob = await rescaleFrom512(resultBlob, originalWidth, originalHeight);
        const finalUrl = URL.createObjectURL(finalBlob);
        setImageSrcWithSnapshot(finalUrl);
      }
    } catch (e) {
      // console.error("Quill processing failed", e);
      alert("Magic Quill failed.");
    } finally {
      imageHandler.setIsSegmenting(false);

      setQuillPoints([]);
      menuState.setShowQuillMenu(false);
      menuState.setActiveMenu("main");
      setQuillMode(null);
    }
  };


  const handleQuillCancel = () => {
    setQuillPoints([]);
    menuState.setShowQuillMenu(false);
    setQuillMode(null);
  };
  const ensureRGBMask = async (maskBlob) => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = 512;
        canvas.height = 512;

        const ctx = canvas.getContext("2d");

        // Fill BLACK background
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, 512, 512);

        // Draw the mask (auto-expanded to 3 channels)
        ctx.drawImage(img, 0, 0, 512, 512);

        // Export as RGB PNG
        canvas.toBlob((blob) => {
          resolve(new File([blob], "mask_rgb.png", { type: "image/png" }));
        }, "image/png");
      };

      img.onerror = reject;
      img.src = URL.createObjectURL(maskBlob);
    });
  };

  /* ------------------------------------------------------------
   * DRAG END HANDLER (New Interaction)
   * ------------------------------------------------------------ */
  const handleDragEnd = async (startPoint, endPoint) => {
    // console.log("Drag End:", startPoint, endPoint);

    try {
      imageHandler.setIsSegmenting(true);

      // 1. Prepare Image
      const blob = await fetch(imageHandler.imageSrc).then((r) => r.blob());
      const originalFile = new File([blob], "source.png", { type: blob.type });
      const {
        file: resizedImage,
        originalWidth,
        originalHeight,
      } = await resizeTo512(originalFile);

      // 2. Prepare Coordinates (Mapped to 512x512)
      const points = {
        x1: (startPoint.x / originalWidth) * 512,
        y1: (startPoint.y / originalHeight) * 512,
        x2: (endPoint.x / originalWidth) * 512,
        y2: (endPoint.y / originalHeight) * 512,
      };

      // console.log("🚀 Triggering API with points:", points);

      // 3. Call API (Conditional based on tool)

      // Generate binary mask for API
      let maskFile = imageHandler.maskPath;
      if (maskFile) {
        let maskBinary = await generateBinaryMaskFile(maskFile, 512, 512);
        maskFile = await ensureRGBMask(maskBinary);
      }

      let response;
      if (menuState.activeTool === "move") {
        // Use Move API
        response = await runDragInpaint(resizedImage, maskFile, points);
      } else {
        // Use Drag API (Expects handle/target structure)
        const dragPoints = {
          handle: { x: points.x1, y: points.y1 },
          target: { x: points.x2, y: points.y2 },
        };
        response = await runDragModel(resizedImage, maskFile, dragPoints);
      }

      // 4. Handle Response
      let finalUrl = null;
      const b64 = response.image_b64 || response.image_base64;

      if (b64) {
        const fetchSrc = b64.startsWith("data:")
          ? b64
          : `data:image/png;base64,${b64}`;
        const res = await fetch(fetchSrc);
        const b = await res.blob();
        const finalBlob = await rescaleFrom512(
          b,
          originalWidth,
          originalHeight
        );
        finalUrl = URL.createObjectURL(finalBlob);
      } else if (response instanceof Blob) {
        const finalBlob = await rescaleFrom512(
          response,
          originalWidth,
          originalHeight
        );
        finalUrl = URL.createObjectURL(finalBlob);
      }

      if (finalUrl) {
        // console.log("Operation Successful. Updating Image.");
        setImageSrcWithSnapshot(finalUrl);
        imageHandler.handleReset();
        menuState.setActiveMenu("main");
        menuState.setActiveTool(null);
      } else {
        throw new Error("No valid image data returned from API");
      }
    } catch (error) {
      // console.error("Move/Drag Failed:", error);
      alert(`Operation failed: ${error.message}`);
    } finally {
      imageHandler.setIsSegmenting(false);
    }
  };

  /* ------------------------------------------------------------
   * CANVAS CLICK HANDLER (Unified Logic)
   * ------------------------------------------------------------ */
  const handleCanvasClick = async (xCanvas, yCanvas, xDisplay, yDisplay) => {
    const clickPoint = {
      imageX: xCanvas,
      imageY: yCanvas,
      viewX: xDisplay,
      viewY: yDisplay,
    };

    // console.log(
    //   `Canvas Click! Tool: ${menuState.activeTool}, Stage: ${menuState.moveStage}`
    // );

    // === LOGIC FOR BOTH MOVE AND DRAG ===
    if (menuState.activeTool === "move" || menuState.activeTool === "drag") {
      // STEP 1: Set Start Point
      if (menuState.moveStage === "start") {
        // console.log("Start Point Set:", clickPoint);
        menuState.setmoveStartPoint(clickPoint);
        menuState.setmoveStage("end");
      }

      // STEP 2: Set End Point & Execute API
      else if (menuState.moveStage === "end") {
        const start = menuState.moveStartPoint;
        const end = clickPoint;

        if (!start) {
          // console.error("Missing Start Point! Resetting.");
          menuState.setmoveStage("start");
          return;
        }

        // console.log("End Point Set:", end);
        menuState.setmoveEndPoint(end);
        menuState.setmoveStage("idle");

        try {
          imageHandler.setIsSegmenting(true); // Show loading spinner

          // 1. Prepare Image
          const blob = await fetch(imageHandler.imageSrc).then((r) => r.blob());
          const originalFile = new File([blob], "source.png", {
            type: blob.type,
          });
          const {
            file: resizedImage,
            originalWidth,
            originalHeight,
          } = await resizeTo512(originalFile);

          // 2. Prepare Coordinates (Mapped to 512x512)
          const points = {
            x1: (start.imageX / originalWidth) * 512,
            y1: (start.imageY / originalHeight) * 512,
            x2: (end.imageX / originalWidth) * 512,
            y2: (end.imageY / originalHeight) * 512,
          };

          // console.log("🚀 Triggering API with points:", points);

          // 3. Call API (Conditional based on tool)

          // Generate binary mask for API
          let maskFile = imageHandler.maskPath;
          if (maskFile) {
            // console.log("lol");
            let maskBinary = await generateBinaryMaskFile(maskFile, 512, 512);
            maskFile = await ensureRGBMask(maskBinary);
          }

          let response;
          if (menuState.activeTool === "move") {
            // Use Move API
            response = await runDragInpaint(resizedImage, maskFile, points);
          } else {
            // Use Drag API (Expects handle/target structure)
            const dragPoints = {
              handle: { x: points.x1, y: points.y1 },
              target: { x: points.x2, y: points.y2 },
            };
            response = await runDragModel(resizedImage, maskFile, dragPoints);
          }

          // 4. Handle Response
          let finalUrl = null;
          // Normalize response format
          const b64 = response.image_b64 || response.image_base64;

          if (b64) {
            // FIX: Check if the backend already sent the prefix "data:image..."
            // If it starts with 'data:', use it as is. If not, add the prefix.
            const fetchSrc = b64.startsWith("data:")
              ? b64
              : `data:image/png;base64,${b64}`;

            const res = await fetch(fetchSrc);
            const b = await res.blob();
            const finalBlob = await rescaleFrom512(
              b,
              originalWidth,
              originalHeight
            );
            finalUrl = URL.createObjectURL(finalBlob);
          } else if (response instanceof Blob) {
            const finalBlob = await rescaleFrom512(
              response,
              originalWidth,
              originalHeight
            );
            finalUrl = URL.createObjectURL(finalBlob);
          } else if (response.saved_images) {
            // // Fallback if runDragModel returned raw JSON but logic inside runDragModel didn't fetch blob
            // console.warn(
            //   "Received raw JSON from Drag API, ensure runDragModel fetches the blob."
            // );
          }

          if (finalUrl) {
            // console.log("Operation Successful. Updating Image.");
            setImageSrcWithSnapshot(finalUrl);

            // Cleanup
            imageHandler.handleReset();
            menuState.setActiveMenu("main");
            menuState.setActiveTool(null);
          } else {
            throw new Error("No valid image data returned from API");
          }
        } catch (error) {
          // console.error("Move/Drag Failed:", error);
          alert(`Operation failed: ${error.message}`);
        } finally {
          imageHandler.setIsSegmenting(false);
          menuState.setmoveStartPoint(null);
          menuState.setmoveEndPoint(null);
        }
      }
      return;
    }

    // === ADD/SUBTRACT MASK LOGIC ===
    if (menuState.activeMenu === "addSubtract") {
      let nextFg = imageHandler.fgPoints;
      let nextBg = imageHandler.bgPoints;

      if (menuState.maskSelectionMode === "add") {
        nextFg = [...nextFg, clickPoint];
      } else if (menuState.maskSelectionMode === "subtract") {
        nextBg = [...nextBg, clickPoint];
      }

      imageHandler.setFgPoints(nextFg);
      imageHandler.setBgPoints(nextBg);

      const xs = [
        ...nextFg.map((p) => Math.round(p.imageX)),
        ...nextBg.map((p) => Math.round(p.imageX)),
      ];
      const ys = [
        ...nextFg.map((p) => Math.round(p.imageY)),
        ...nextBg.map((p) => Math.round(p.imageY)),
      ];
      const labels = [...nextFg.map(() => 1), ...nextBg.map(() => 0)];

      const runSeg = async () => {
        imageHandler.setIsSegmenting(true);
        try {
          const blob = await fetch(imageHandler.imageSrc).then((r) => r.blob());
          const originalFile = new File([blob], "image.png", {
            type: blob.type,
          });
          const {
            file: resizedFile,
            originalWidth,
            originalHeight,
          } = await resizeTo512(originalFile);

          const scaledXs = xs.map((x) => Math.round((x / originalWidth) * 512));
          const scaledYs = ys.map((y) =>
            Math.round((y / originalHeight) * 512)
          );

          const data = await segment(resizedFile, scaledXs, scaledYs, labels);

          if (data && data.mask_coords) {
            const newPolygons = data.mask_coords.map((contour) => ({
              points: contour.map(([x, y]) => ({
                x: (x / 512) * originalWidth,
                y: (y / 512) * originalHeight,
              })),
              stroke: "lime",
            }));
            setPolygonsWithSnapshot(newPolygons);

            // Create mask canvas from polygon points (same as raster brush)
            const maskCanvas = document.createElement("canvas");
            maskCanvas.width = originalWidth;
            maskCanvas.height = originalHeight;
            const mctx = maskCanvas.getContext("2d");

            // Fill with black background
            mctx.fillStyle = "black";
            mctx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);

            // Draw white polygons
            mctx.fillStyle = "white";
            newPolygons.forEach((poly) => {
              mctx.beginPath();
              poly.points.forEach((p, i) => {
                if (i === 0) {
                  mctx.moveTo(p.x, p.y);
                } else {
                  mctx.lineTo(p.x, p.y);
                }
              });
              mctx.closePath();
              mctx.fill();
            });

            // Convert canvas to blob and then to File
            maskCanvas.toBlob((blob) => {
              const file = new File([blob], "mask.png", { type: "image/png" });
              setMaskPathWithSnapshot(file);
            });
          }
        } catch (e) {
          // console.error(e);
        } finally {
          imageHandler.setIsSegmenting(false);
        }
      };
      runSeg();
    }
  };

  // --- HELPERS ---

  const shouldShowSlider =
    menuState.activeMenu === "main" &&
    menuState.selectedMainIcon === "adjust" &&
    menuState.showActiveMenu &&
    menuState.activeEditControl;

  const getProgressPercentage = (id) => {
    const control = editIcons.find((c) => c.id === id);
    if (!control) return 0;
    const value = imageEditor.editValues[id];
    return ((value - control.min) / (control.max - control.min)) * 100;
  };

  const getClickMode = () => {
    if (menuState.activeMenu === "addSubtract") {
      return menuState.maskSelectionMode;
    }
    return "select";
  };

  const handleConfirmAndDownload = async () => {
    if (imageHandler.maskPath) {
      try {
        const base64 = await generateBinaryMask(imageHandler.maskPath);
        // saveMa`skLocally(base64);
      } catch (e) {
        // console.error("Mask generation failed", e);
      }
    }
    menuState.handleConfirmMenu();
  };

  /* ------------------------------------------------------------
   * UPSCALE HANDLER
   * ------------------------------------------------------------ */
  const handleUpscale = async () => {
    // console.log("Tool: Upscale - Initializing");
    try {
      imageHandler.setIsSegmenting(true);

      // 1. Get current image as file
      const blob = await fetch(imageHandler.imageSrc).then((r) => r.blob());
      const file = new File([blob], "image.png", { type: blob.type });

      // 2. Call Upscale API
      const upscaledFile = await processImageUpscale(
        file,
        "high quality, sharp"
      );

      // 3. Update Image
      const newUrl = URL.createObjectURL(upscaledFile);
      setImageSrcWithSnapshot(newUrl);

      // 4. Reset UI
      menuState.setActiveMenu("main");
      menuState.setActiveTool(null);
    } catch (error) {
      // console.error("Upscale failed:", error);
      alert("Upscale failed.");
    } finally {
      imageHandler.setIsSegmenting(false);
    }
  };

  const handleExport = () => {
    const link = document.createElement("a");
    link.download = "edited-image.png";
    // Use processed image if available (filters applied), otherwise base image
    link.href = currentProcessedImage || imageHandler.imageSrc;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleCommitAdjustments = () => {
    if (currentProcessedImage) {
      pushSnapshot(); // Save state before applying changes
      setImageSrcWithSnapshot(currentProcessedImage);
      imageEditor.resetEditValues();
      setCurrentProcessedImage(null); // Reset processed image state
    }
  };

  return (
    <>
      <div className="overflow-hidden h-auto">
        <style>{`
          @keyframes compress-to-circle {
            0% { width: 100%; height: 4rem; border-radius: 9999px; }
            100% { width: 4rem; height: 4rem; border-radius: 9999px; }
          }
          @keyframes expand-from-circle {
            0% { width: 4rem; height: 4rem; border-radius: 9999px; }
            100% { width: 100%; height: 4rem; border-radius: 9999px; }
          }
          .animate-compress { animation: compress-to-circle 0.9s cubic-bezier(0.34, 1.56, 0.64, 1) forwards; }
          .animate-expand { animation: expand-from-circle 0.9s cubic-bezier(0.34, 1.56, 0.64, 1) forwards; }
        `}</style>

        <div className="flex flex-col w-full h-screen-safe bg-black overflow-hidden">
          <input
            ref={imageHandler.fileInputRef}
            type="file"
            accept="image/*"
            onChange={imageHandler.handleImageUpload}
            className="hidden"
          />
          <canvas ref={imageHandler.canvasExportRef} className="hidden" />

          <Header onExport={handleExport} />

          <main
            className={`flex-1 flex flex-col items-center justify-start px-3 overflow-hidden ${menuState.showCropUI ? "pt-1 pb-0" : "pt-1 pb-0"
              }`}
          >


            {menuState.showCropUI ? (
              <CropToolUI
                ref={cropRef}
                imageSrc={imageHandler.imageSrc}
                pushSnapshot={pushSnapshot} // <- push into global history
                handleUndo={handleUndo} // <- global undo
                handleRedo={handleRedo} // <- global redo
                canUndo={canUndo()} // <- initial enabled state
                canRedo={canRedo()} // <- initial enabled state
                onStateChange={setShowCropConfirm}
                onBackToMain={(newImage) => {
                  if (newImage) setImageSrcWithSnapshot(newImage); // use snapshot wrapper
                  menuState.setShowCropUI(false);
                  menuState.setActiveMenu("main");
                }}
              />
            ) : (
              <ImageCanvas
                imageSrc={imageHandler.imageSrc}
                compareSrc={compareSrc}
                isComparing={isComparing}
                polygons={imageHandler.polygons}
                setPolygons={imageHandler.setPolygons}
                maskPath={imageHandler.maskPath}
                setMaskPath={imageHandler.setMaskPath}
                fgPoints={
                  menuState.activeMenu === "addSubtract" ||
                    menuState.activeMenu === "second"
                    ? imageHandler.fgPoints
                    : []
                }
                bgPoints={
                  menuState.activeMenu === "addSubtract" ||
                    menuState.activeMenu === "second"
                    ? imageHandler.bgPoints
                    : []
                }
                isSegmenting={imageHandler.isSegmenting}
                onPointClick={handleCanvasClick}
                onPointRemove={imageHandler.handlePointRemove}
                clickMode={getClickMode()}
                moveStartPoint={menuState.moveStartPoint}
                moveEndPoint={menuState.moveEndPoint}
                activeTool={menuState.activeTool}
                moveStage={menuState.moveStage}
                activeMenu={menuState.activeMenu}
                maskSelectionMode={menuState.maskSelectionMode}
                isBrushActive={menuState.isBrushActive}
                brushSize={menuState.brushSize}
                filterStyle={imageEditor.getFilterStyle}
                editValues={imageEditor.editValues}
                spotlightPolygons={
                  menuState.activeMenu === "addSubtract" ||
                  menuState.activeMenu === "second"
                }
                isProcessingAnim={isProcessingAnim}
                quillMode={quillMode}
                quillPoints={quillPoints}
                setQuillPoints={setQuillPoints}
                penColor={penColor}
                onDragEnd={handleDragEnd}
                onImageProcessed={setCurrentProcessedImage}
              />
            )}

          </main>

          {/* --------------------- FOOTER --------------------- */}
          <footer
            className={`flex-none px-4 transition-all duration-300 ${shouldShowSlider ? "pb-2 pt-0" : "pb-2 pt-0"
              }`}
          >
            <Toolbar
              activeMenu={menuState.activeMenu}
              handleConfirmMenu={handleConfirmAndDownload}
              onUndo={handleUndo}
              onRedo={handleRedo}
              canUndo={canUndo}
              canRedo={canRedo}
              onCompareStart={handleCompareStart}
              onCompareEnd={handleCompareEnd}
              onToolSelect={menuState.handleMainIconClick}
              activeTool={menuState.selectedMainIcon}
              currentOperation={menuState.activeTool}
              handleMainIconClick={menuState.handleMainIconClick}
              onAnalyzeClick={handleActionMenuanalyze}
              isAnalyzing={analysisLoading}
              showCropConfirm={showCropConfirm}
              onCropConfirm={() => cropRef.current?.confirm()}
              onCropReset={() => cropRef.current?.reset()}
            />

            <MenuBar
              activeMenu={menuState.activeMenu}
              setActiveMenu={menuState.setActiveMenu}
              showActiveMenu={menuState.showActiveMenu}
              setShowActiveMenu={menuState.setShowActiveMenu}
              isAnimating={menuState.isAnimating}
              setIsAnimating={menuState.setIsAnimating}
              selectedMainIcon={menuState.selectedMainIcon}
              setSelectedMainIcon={menuState.setSelectedMainIcon}
              handleMainIconClick={menuState.handleMainIconClick}
              handleCloseMaskSelection={menuState.handleCloseMaskSelection}
              maskSelectionMode={menuState.maskSelectionMode}
              setMaskSelectionMode={menuState.setMaskSelectionMode}
              handleCloseActionMenu={menuState.handleCloseActionMenu}
              onerase={handleActionMenuerase}
              onmove={handleActionMenumove}
              ondrag={handleActionMenudrag}
              isBrushActive={menuState.isBrushActive}
              setIsBrushActive={menuState.setIsBrushActive}
              brushSize={menuState.brushSize}
              setBrushSize={menuState.setBrushSize}
              activeEditControl={menuState.activeEditControl}
              setActiveEditControl={menuState.setActiveEditControl}
              getProgressPercentage={getProgressPercentage}
              handleAutoAdjust={imageEditor.handleAutoAdjust}
              handleSliderChange={imageEditor.handleSliderChange}
              editIcons={editIcons}
              CircularProgress={CircularProgress}
              selectedTool={menuState.selectedTool}
              setSelectedTool={menuState.setSelectedTool}
              onBrushClick={() => menuState.setShowStylesMenu(true)}
              onQuillClick={() => {
                menuState.setActiveMenu("magic");
              }}
              quillMode={quillMode}
              setQuillMode={setQuillMode}
              onQuillDone={handleQuillDone}
              onQuillCancel={handleQuillCancel}
              onUpscale={handleUpscale}
              penColor={penColor}
              setPenColor={setPenColor}
              onCropClick={() => {
                setShowCropUI(true);
                menuState.setActiveMenu("crop");
              }}
              editValues={imageEditor.editValues}
              selectedSelectionTool={menuState.selectedSelectionTool}
              setSelectedSelectionTool={menuState.setSelectedSelectionTool}
              showSelectionMenu={menuState.showSelectionMenu}
              handleOpenSelectionMenu={menuState.handleOpenSelectionMenu}
              handleCloseSelectionMenu={menuState.handleCloseSelectionMenu}
              backgroundImage={imageHandler.imageSrc || undefined}
              onCropCancel={() => cropRef.current?.cancel()}
              onAspectRatioSelect={(ratio) =>
                cropRef.current?.setAspectRatio(ratio)
              }
              onRotate={() => cropRef.current?.rotate()}
              onFlipX={() => cropRef.current?.flipX()}
              onFlipY={() => cropRef.current?.flipY()}
              isProcessingAnim={isProcessingAnim}
              setIsProcessingAnim={setIsProcessingAnim}
              onClearSelection={imageHandler.handleReset}
              onAnalyzeClick={handleActionMenuanalyze}
              isAnalyzing={analysisLoading}
              onGeminiPromptSubmit={handleGeminiPromptSubmit}
              onCommitAdjustments={handleCommitAdjustments}
            />
          </footer>
        </div>

        {menuState.showStylesMenu && (
          <StylesMenu
            onBack={() => menuState.setShowStylesMenu(false)}
            imageSrc={imageHandler.imageSrc}
            onImageUpdate={(newImageUrl) => {
              setImageSrcWithSnapshot(newImageUrl);
              menuState.setShowStylesMenu(false);
            }}
          />
        )}

        {/* Analysis Result Panel */}
        <AnalysisResultPanel
          isVisible={showAnalysisPanel}
          isLoading={analysisLoading}
          analysisResult={analysisResult}
          error={analysisError}
          onClose={() => setShowAnalysisPanel(false)}
        />
      </div>
    </>
  );
}
