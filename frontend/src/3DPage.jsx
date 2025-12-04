import React, { useState, useEffect, Suspense, useMemo, useRef, useCallback } from "react";
import Toolbar from "./components/menus/Toolbar";
import ImageCanvas from "./ImageCanvas";
import Render3DBar from "./components/Render3DBar";
// Hooks
import { useImageHandler } from "./hooks/useImageHandler";
import { useMenuState } from "./hooks/useMenuState";
import { useImageEditor } from "./hooks/useImageEditor";
import Header from "./components/Header";
// API functions
import { runmoveFromMask, runSegmentation } from "./api";
import { processRelighting } from "./api/lbmAPI";
import { toast } from 'sonner';

// UI components
import StylesMenu from "./components/menus/StylesMenu";

// React Router
import { useLocation, useNavigate } from "react-router-dom";

// 3D Components
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Environment } from "@react-three/drei";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";
import * as THREE from "three";


const TwinklingStars = () => {
    const STAR_COUNT = 100;
    return (
        <div className="absolute inset-0 pointer-events-none z-[95] overflow-hidden bg-black/50">
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

// 3D Model Component - now accepts dynamic GLB URL
function Model({ glbUrl }) {
    const [scene, setScene] = useState(null);

    useEffect(() => {
        if (!glbUrl) return;

        const loader = new GLTFLoader();
        loader.load(
            glbUrl,
            (gltf) => {
                console.log('GLB model loaded successfully');
                setScene(gltf.scene);
            },
            (progress) => {
                console.log('Loading progress:', (progress.loaded / progress.total * 100) + '%');
            },
            (error) => {
                console.error('Error loading GLB model:', error);
            }
        );

        // Cleanup
        return () => {
            if (glbUrl && glbUrl.startsWith('blob:')) {
                URL.revokeObjectURL(glbUrl);
            }
        };
    }, [glbUrl]);

    if (!scene) {
        return null;
    }

    return <primitive object={scene} />;
}

// 3D Viewer Component
function GLBViewer({ backgroundImage, backgroundCanvasRef, onThreeCanvasReady, glbUrl }) {
    const containerRef = React.useRef(null);
    // State to hold the calculated display size (CSS pixels) to match aspect ratio
    const [displayStyle, setDisplayStyle] = React.useState({ width: '100%', height: '100%' });

    // Draw background image to canvas at original resolution & Calculate Aspect Ratio
    React.useEffect(() => {
        if (backgroundImage && backgroundCanvasRef.current && containerRef.current) {
            const canvas = backgroundCanvasRef.current;
            const ctx = canvas.getContext('2d');
            const img = new Image();

            img.onload = () => {
                // 1. Set Internal Resolution (High Quality)
                canvas.width = img.width;
                canvas.height = img.height;

                // 2. Draw image at full resolution
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, img.width, img.height);

                // 3. Calculate CSS Display Dimensions (Object-Fit: Contain)
                // This ensures the canvas + 3D view don't stretch the image
                const container = containerRef.current;
                const containerWidth = container.clientWidth;
                const containerHeight = container.clientHeight;

                const imgRatio = img.width / img.height;
                const containerRatio = containerWidth / containerHeight;

                let finalWidth, finalHeight;

                if (containerRatio > imgRatio) {
                    // Container is wider than image -> limit by height
                    finalHeight = containerHeight;
                    finalWidth = containerHeight * imgRatio;
                } else {
                    // Container is taller than image -> limit by width
                    finalWidth = containerWidth;
                    finalHeight = containerWidth / imgRatio;
                }

                setDisplayStyle({
                    width: `${finalWidth}px`,
                    height: `${finalHeight}px`,
                    position: 'relative' // Crucial for overlaying 3D
                });
            };

            img.src = backgroundImage;
        }
    }, [backgroundImage, backgroundCanvasRef]);

    return (
        // Outer container fills the screen and centers the content
        <div ref={containerRef} className="w-full h-full flex items-center justify-center bg-black">

            {/* Inner Wrapper: Explicitly sized to match Image Aspect Ratio */}
            <div style={displayStyle}>

                {/* Background Canvas (2D Image) */}
                <canvas
                    ref={backgroundCanvasRef}
                    className="absolute inset-0 z-0"
                    style={{ width: '100%', height: '100%', pointerEvents: 'none' }}
                />

                {/* 3D Canvas (Overlay) */}
                <div className="absolute inset-0 z-10">
                    <Canvas
                        camera={{ position: [0, 0, 5], fov: 50 }}
                        gl={{ preserveDrawingBuffer: true, alpha: true, antialias: true }}
                        onCreated={({ gl }) => {
                            if (onThreeCanvasReady) onThreeCanvasReady(gl.domElement);
                        }}
                    >
                        <Suspense fallback={null}>
                            <Model glbUrl={glbUrl} />
                            <Environment preset="studio" />
                            {/* ContactShadows REMOVED here */}
                        </Suspense>
                        <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} minDistance={2} maxDistance={10} />
                    </Canvas>
                </div>
            </div>
        </div>
    );
}

export default function GLBHandler() {
    const imageHandler = useImageHandler();
    const menuState = useMenuState();
    const imageEditor = useImageEditor();
    const location = useLocation();
    const navigate = useNavigate();

    // Blend mode state
    const [blendMode, setBlendMode] = useState(null);

    // References
    const canvasContainerRef = React.useRef(null);
    const backgroundCanvasRef = React.useRef(null);
    const threeCanvasRef = React.useRef(null);

    // GLB model state
    const [glbModelUrl, setGlbModelUrl] = useState(null);

    // Relighting state
    const [isRelighting, setIsRelighting] = useState(false);
    const [relightingError, setRelightingError] = useState(null);

    // Overlay image state
    const [overlayImage, setOverlayImage] = useState(null);
    const [overlayPosition, setOverlayPosition] = useState({ x: 50, y: 50 });
    const [overlayRotation, setOverlayRotation] = useState(0);
    const [overlayScale, setOverlayScale] = useState(1);

    // --------------- UNDO / REDO ----------------
    const undoStackRef = useRef([]);
    const redoStackRef = useRef([]);

    const snapshot = useCallback((obj) => {
        if (typeof structuredClone !== "undefined") return structuredClone(obj);
        return JSON.parse(JSON.stringify(obj));
    }, []);

    const takeSnapshot = useCallback(() => {
        return snapshot({
            overlayPosition,
            overlayRotation,
            overlayScale,
            blendMode,
        });
    }, [overlayPosition, overlayRotation, overlayScale, blendMode, snapshot]);

    const pushSnapshot = useCallback(() => {
        try {
            undoStackRef.current.push(takeSnapshot());
            if (undoStackRef.current.length > 30) undoStackRef.current.shift();
            redoStackRef.current = [];
        } catch (e) {
            console.warn("Failed to push snapshot", e);
        }
    }, [takeSnapshot]);

    const applySnapshot = useCallback((snap) => {
        if (!snap) return;
        try {
            if (snap.overlayPosition) setOverlayPosition(snap.overlayPosition);
            if (typeof snap.overlayRotation === 'number') setOverlayRotation(snap.overlayRotation);
            if (typeof snap.overlayScale === 'number') setOverlayScale(snap.overlayScale);
            if (snap.blendMode) setBlendMode(snap.blendMode);
        } catch (e) {
            console.error("Failed to apply snapshot data:", e);
        }
    }, []);

    const handleUndo = useCallback(() => {
        if (!undoStackRef.current.length) return;
        try {
            redoStackRef.current.push(takeSnapshot());
            const prev = undoStackRef.current.pop();
            applySnapshot(prev);
        } catch (e) {
            console.warn("Undo failed", e);
        }
    }, [takeSnapshot, applySnapshot]);

    const handleRedo = useCallback(() => {
        if (!redoStackRef.current.length) return;
        try {
            undoStackRef.current.push(takeSnapshot());
            const next = redoStackRef.current.pop();
            applySnapshot(next);
        } catch (e) {
            console.warn("Redo failed", e);
        }
    }, [takeSnapshot, applySnapshot]);

    const canUndo = () => undoStackRef.current.length > 0;
    const canRedo = () => redoStackRef.current.length > 0;

    // Memoize overlay image URL to prevent creating new blob URLs on every render
    const overlayImageUrl = useMemo(() => {
        if (!overlayImage) return null;
        return URL.createObjectURL(overlayImage);
    }, [overlayImage]);

    // Cleanup overlay image URL on unmount or when overlayImage changes
    useEffect(() => {
        return () => {
            if (overlayImageUrl) {
                URL.revokeObjectURL(overlayImageUrl);
            }
        };
    }, [overlayImageUrl]);

    // Interaction Flags
    const [ismoveging, setIsmoveging] = useState(false);
    const [isRotating, setIsRotating] = useState(false);
    const [isResizing, setIsResizing] = useState(false);

    // Interaction Data State
    const [moveOffset, setmoveOffset] = useState({ x: 0, y: 0 });
    const [initialAngle, setInitialAngle] = useState(0);
    const [initialScale, setInitialScale] = useState(1);
    const [resizeCorner, setResizeCorner] = useState(null);

    // --- Vector Math State for Anchor Resizing ---
    const [startCenter, setStartCenter] = useState({ x: 0, y: 0 });
    const [anchorPoint, setAnchorPoint] = useState({ x: 0, y: 0 });
    const [startDist, setStartDist] = useState(0);


    // --- 1. move HANDLERS ---

    const handleOverlayMouseDown = (e) => {
        pushSnapshot();
        e.stopPropagation();
        setIsmoveging(true);

        if (canvasContainerRef.current) {
            const mainRect = canvasContainerRef.current.getBoundingClientRect();
            const clickXPercent = ((e.clientX - mainRect.left) / mainRect.width) * 100;
            const clickYPercent = ((e.clientY - mainRect.top) / mainRect.height) * 100;

            setmoveOffset({
                x: clickXPercent - overlayPosition.x,
                y: clickYPercent - overlayPosition.y
            });
        }
    };

    const handleOverlayTouchStart = (e) => {
        pushSnapshot();
        e.stopPropagation();
        const touch = e.touches[0];
        setIsmoveging(true);

        if (canvasContainerRef.current) {
            const mainRect = canvasContainerRef.current.getBoundingClientRect();
            const clickXPercent = ((touch.clientX - mainRect.left) / mainRect.width) * 100;
            const clickYPercent = ((touch.clientY - mainRect.top) / mainRect.height) * 100;

            setmoveOffset({
                x: clickXPercent - overlayPosition.x,
                y: clickYPercent - overlayPosition.y
            });
        }
    };


    // --- 2. RESIZE START ---

    const handleResizeStart = (e, corner) => {
        pushSnapshot();
        e.stopPropagation();
        if (!canvasContainerRef.current) return;

        setIsResizing(true);
        setResizeCorner(corner);
        setInitialScale(overlayScale);

        const mainRect = canvasContainerRef.current.getBoundingClientRect();
        const centerX = mainRect.left + (overlayPosition.x / 100) * mainRect.width;
        const centerY = mainRect.top + (overlayPosition.y / 100) * mainRect.height;
        setStartCenter({ x: centerX, y: centerY });

        const mouseX = e.clientX;
        const mouseY = e.clientY;

        const anchorX = 2 * centerX - mouseX;
        const anchorY = 2 * centerY - mouseY;
        setAnchorPoint({ x: anchorX, y: anchorY });

        const dist = Math.sqrt(Math.pow(mouseX - anchorX, 2) + Math.pow(mouseY - anchorY, 2));
        setStartDist(dist);
    };

    const handleResizeTouchStart = (e, corner) => {
        pushSnapshot();
        e.stopPropagation();
        const touch = e.touches[0];
        if (!canvasContainerRef.current) return;

        setIsResizing(true);
        setResizeCorner(corner);
        setInitialScale(overlayScale);

        const mainRect = canvasContainerRef.current.getBoundingClientRect();
        const centerX = mainRect.left + (overlayPosition.x / 100) * mainRect.width;
        const centerY = mainRect.top + (overlayPosition.y / 100) * mainRect.height;
        setStartCenter({ x: centerX, y: centerY });

        const mouseX = touch.clientX;
        const mouseY = touch.clientY;

        const anchorX = 2 * centerX - mouseX;
        const anchorY = 2 * centerY - mouseY;
        setAnchorPoint({ x: anchorX, y: anchorY });

        const dist = Math.sqrt(Math.pow(mouseX - anchorX, 2) + Math.pow(mouseY - anchorY, 2));
        setStartDist(dist);
    };


    // --- 3. MOVE HANDLERS ---

    const handleOverlayMouseMove = (e) => {
        if (ismoveging) {
            const rect = e.currentTarget.getBoundingClientRect();
            const x = ((e.clientX - rect.left) / rect.width) * 100;
            const y = ((e.clientY - rect.top) / rect.height) * 100;
            setOverlayPosition({ x: x - moveOffset.x, y: y - moveOffset.y });

        } else if (isRotating) {
            const rect = e.currentTarget.getBoundingClientRect();
            const centerX = rect.left + rect.width / 2;
            const centerY = rect.top + rect.height / 2;
            const angle = Math.atan2(e.clientY - centerY, e.clientX - centerX) * (180 / Math.PI);
            setOverlayRotation(angle - initialAngle);

        } else if (isResizing && canvasContainerRef.current) {
            const mainRect = canvasContainerRef.current.getBoundingClientRect();
            const mouseX = e.clientX;
            const mouseY = e.clientY;

            const currentDist = Math.sqrt(Math.pow(mouseX - anchorPoint.x, 2) + Math.pow(mouseY - anchorPoint.y, 2));
            const scaleRatio = startDist > 0 ? currentDist / startDist : 1;
            const newScale = Math.max(0.1, initialScale * scaleRatio);
            setOverlayScale(newScale);

            const oldVectorX = startCenter.x - anchorPoint.x;
            const oldVectorY = startCenter.y - anchorPoint.y;

            const newCenterX = anchorPoint.x + (oldVectorX * scaleRatio);
            const newCenterY = anchorPoint.y + (oldVectorY * scaleRatio);

            const newXPercent = ((newCenterX - mainRect.left) / mainRect.width) * 100;
            const newYPercent = ((newCenterY - mainRect.top) / mainRect.height) * 100;

            setOverlayPosition({ x: newXPercent, y: newYPercent });
        }
    };

    const handleOverlayTouchMove = (e) => {
        const touch = e.touches[0];

        if (ismoveging) {
            const rect = e.currentTarget.getBoundingClientRect();
            const x = ((touch.clientX - rect.left) / rect.width) * 100;
            const y = ((touch.clientY - rect.top) / rect.height) * 100;
            setOverlayPosition({ x: x - moveOffset.x, y: y - moveOffset.y });

        } else if (isRotating) {
            const rect = e.currentTarget.getBoundingClientRect();
            const centerX = rect.left + rect.width / 2;
            const centerY = rect.top + rect.height / 2;
            const angle = Math.atan2(touch.clientY - centerY, touch.clientX - centerX) * (180 / Math.PI);
            setOverlayRotation(angle - initialAngle);

        } else if (isResizing && canvasContainerRef.current) {
            const mainRect = canvasContainerRef.current.getBoundingClientRect();
            const mouseX = touch.clientX;
            const mouseY = touch.clientY;

            const currentDist = Math.sqrt(Math.pow(mouseX - anchorPoint.x, 2) + Math.pow(mouseY - anchorPoint.y, 2));
            const scaleRatio = startDist > 0 ? currentDist / startDist : 1;
            const newScale = Math.max(0.1, initialScale * scaleRatio);
            setOverlayScale(newScale);

            const oldVectorX = startCenter.x - anchorPoint.x;
            const oldVectorY = startCenter.y - anchorPoint.y;

            const newCenterX = anchorPoint.x + (oldVectorX * scaleRatio);
            const newCenterY = anchorPoint.y + (oldVectorY * scaleRatio);

            const newXPercent = ((newCenterX - mainRect.left) / mainRect.width) * 100;
            const newYPercent = ((newCenterY - mainRect.top) / mainRect.height) * 100;

            setOverlayPosition({ x: newXPercent, y: newYPercent });
        }
    };


    // --- 4. STANDARD HANDLERS ---

    const handleOverlayMouseUp = () => {
        setIsmoveging(false);
        setIsRotating(false);
        setIsResizing(false);
        setResizeCorner(null);
    };

    const handleRotateStart = (e) => {
        pushSnapshot();
        e.stopPropagation();
        setIsRotating(true);
        const rect = e.currentTarget.parentElement.parentElement.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        const angle = Math.atan2(e.clientY - centerY, e.clientX - centerX) * (180 / Math.PI);
        setInitialAngle(angle - overlayRotation);
    };

    const handleRotateTouchStart = (e) => {
        pushSnapshot();
        e.stopPropagation();
        const touch = e.touches[0];
        setIsRotating(true);
        const rect = e.currentTarget.parentElement.parentElement.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        const angle = Math.atan2(touch.clientY - centerY, touch.clientX - centerX) * (180 / Math.PI);
        setInitialAngle(angle - overlayRotation);
    };

    // Track if we've already processed the location state
    const hasProcessedState = React.useRef(false);

    useEffect(() => {
        // Only process location state once to prevent infinite loops
        if (hasProcessedState.current) return;
        if (!location.state) return;

        hasProcessedState.current = true;

        if (location.state.overlayImage) setOverlayImage(location.state.overlayImage);
        if (location.state.backgroundImage && imageHandler) imageHandler.setImageSrc(location.state.backgroundImage);
        if (location.state.blendMode) setBlendMode(location.state.blendMode);

        // Handle GLB model blob - create URL for Three.js loader
        if (location.state.glbModel) {
            const glbBlob = location.state.glbModel;
            const url = URL.createObjectURL(glbBlob);
            setGlbModelUrl(url);
            console.log('Created GLB URL:', url);
        }
    }, [location.state, imageHandler]);

    // Cleanup GLB URL on unmount only
    useEffect(() => {
        return () => {
            if (glbModelUrl) {
                URL.revokeObjectURL(glbModelUrl);
            }
        };
    }, []);

    const handleCanvasClick = async (xCanvas, yCanvas, xDisplay, yDisplay) => {
        const clickPoint = { imageX: xCanvas, imageY: yCanvas, viewX: xDisplay, viewY: yDisplay };

        if (menuState.activeTool === "move") {
            if (menuState.moveStage === "start") {
                menuState.setmoveStartPoint(clickPoint);
                menuState.setmoveStage("end");
            } else if (menuState.moveStage === "end") {
                const start = menuState.moveStartPoint || clickPoint;
                const end = clickPoint;
                menuState.setmoveEndPoint(end);
                menuState.setmoveStage("idle");
                const poly = imageHandler.polygons[0];
                await runmoveFromMask(imageHandler.imageSrc, poly, start, end, {
                    setIsSegmenting: imageHandler.setIsSegmenting,
                    setImageSrc: imageHandler.setImageSrc,
                    setFgPoints: imageHandler.setFgPoints,
                    setBgPoints: imageHandler.setBgPoints,
                    setPolygons: imageHandler.setPolygons,
                    setActiveTool: menuState.setActiveTool,
                    setmoveStage: menuState.setmoveStage,
                    setmoveStartPoint: menuState.setmoveStartPoint,
                    setmoveEndPoint: menuState.setmoveEndPoint,
                    setActiveMenu: menuState.setActiveMenu,
                    setShowActiveMenu: menuState.setShowActiveMenu,
                });
            }
            return;
        }

        if (menuState.activeMenu === "addSubtract") {
            let nextFg = imageHandler.fgPoints;
            let nextBg = imageHandler.bgPoints;

            if (menuState.maskSelectionMode === "add") nextFg = [...nextFg, clickPoint];
            else if (menuState.maskSelectionMode === "subtract") nextBg = [...nextBg, clickPoint];

            imageHandler.setFgPoints(nextFg);
            imageHandler.setBgPoints(nextBg);

            const xs = [...nextFg.map((p) => p.imageX), ...nextBg.map((p) => p.imageX)];
            const ys = [...nextFg.map((p) => p.imageY), ...nextBg.map((p) => p.imageY)];
            const labels = [...nextFg.map(() => 1), ...nextBg.map(() => 0)];

            runSegmentation(imageHandler.imageSrc, xs, ys, labels, {
                setPolygons: imageHandler.setPolygons,
                setIsSegmenting: imageHandler.setIsSegmenting,
            });
        }
    };

    const getClickMode = () => {
        if (menuState.activeMenu === "addSubtract") {
            return menuState.maskSelectionMode === "add" || menuState.maskSelectionMode === "subtract" ? "add" : "select";
        }
        return "select";
    };

    return (
        <>
            <style>{`
        @keyframes compress-to-circle { 0% { width: 100%; height: 4rem; border-radius: 9999px; } 100% { width: 4rem; height: 4rem; border-radius: 9999px; } }
        @keyframes expand-from-circle { 0% { width: 4rem; height: 4rem; border-radius: 9999px; } 100% { width: 100%; height: 4rem; border-radius: 9999px; } }
        .animate-compress { animation: compress-to-circle 0.9s cubic-bezier(0.34, 1.56, 0.64, 1) forwards; }
        .animate-expand { animation: expand-from-circle 0.9s cubic-bezier(0.34, 1.56, 0.64, 1) forwards; }
      `}</style>

            <div className="flex flex-col w-full h-screen-safe bg-black overflow-hidden">
                <input ref={imageHandler.fileInputRef} type="file" accept="image/*" onChange={imageHandler.handleImageUpload} className="hidden" />
                <canvas ref={imageHandler.canvasExportRef} className="hidden" />

                <Header
                    onBack={() => navigate("/photoroom", { state: { backgroundImage: imageHandler.imageSrc } })}
                    onExport={() => console.log("Export clicked")}
                />

                <main ref={canvasContainerRef} className="flex-1 flex flex-col items-center justify-center px-4 py-4 overflow-hidden relative">
                    {/* Relighting Loading Overlay */}
                    {isRelighting && (
                        <TwinklingStars></TwinklingStars>
                    )}

                    {blendMode === "3d" ? (
                        <GLBViewer
                            backgroundImage={imageHandler.imageSrc}
                            backgroundCanvasRef={backgroundCanvasRef}
                            onThreeCanvasReady={(canvas) => { threeCanvasRef.current = canvas; }}
                            glbUrl={glbModelUrl}
                        />
                    ) : (
                        <>
                            <ImageCanvas
                                imageSrc={imageHandler.imageSrc}
                                polygons={imageHandler.polygons}
                                setPolygons={imageHandler.setPolygons}
                                fgPoints={imageHandler.fgPoints}
                                bgPoints={imageHandler.bgPoints}
                                isSegmenting={imageHandler.isSegmenting}
                                onPointClick={handleCanvasClick}
                                onPointRemove={imageHandler.handlePointRemove}
                                clickMode={getClickMode()}
                                moveStartPoint={menuState.moveStartPoint}
                                moveEndPoint={menuState.moveEndPoint}
                                activeTool={menuState.activeTool}
                                moveStage={menuState.moveStage}
                                activeMenu={menuState.activeMenu}
                                filterStyle={imageEditor.getFilterStyle}
                                editValues={imageEditor.editValues}
                                spotlightPolygons={menuState.activeMenu === "addSubtract" || menuState.activeMenu === "second"}
                            />

                            {overlayImage && (
                                <div
                                    className="absolute inset-0 pointer-events-none"
                                    onMouseMove={handleOverlayMouseMove}
                                    onTouchMove={handleOverlayTouchMove}
                                    onMouseUp={handleOverlayMouseUp}
                                    onTouchEnd={handleOverlayMouseUp}
                                    onMouseLeave={handleOverlayMouseUp}
                                >
                                    <div
                                        className="absolute pointer-events-auto"
                                        style={{
                                            left: `${overlayPosition.x}%`,
                                            top: `${overlayPosition.y}%`,
                                            transform: `translate(-50%, -50%)`,
                                        }}
                                    >
                                        <div
                                            style={{
                                                transform: `rotate(${overlayRotation}deg) scale(${overlayScale})`,
                                                transformOrigin: 'center',
                                                position: 'relative',
                                            }}
                                        >
                                            <img
                                                src={overlayImageUrl}
                                                alt="Overlay"
                                                className="select-none"
                                                style={{
                                                    maxWidth: '200px',
                                                    maxHeight: '200px',
                                                    objectFit: 'contain',
                                                    border: '2px solid #FE5959',
                                                    borderRadius: '8px',
                                                    boxShadow: '0 4px 12px rgba(254, 89, 89, 0.3)',
                                                    cursor: ismoveging ? 'grabbing' : 'grab',
                                                }}
                                                onMouseDown={handleOverlayMouseDown}
                                                onTouchStart={handleOverlayTouchStart}
                                                movegable={false}
                                            />

                                            <div
                                                className="absolute -top-8 left-1/2 transform -translate-x-1/2 w-6 h-6 bg-blue-500 rounded-full cursor-pointer flex items-center justify-center shadow-lg"
                                                onMouseDown={handleRotateStart}
                                                onTouchStart={handleRotateTouchStart}
                                                style={{ border: '2px solid white' }}
                                            >
                                                <span className="text-white text-xs">↻</span>
                                            </div>

                                            {/* Resize Handles */}
                                            {['tl', 'tr', 'bl', 'br'].map(corner => (
                                                <div
                                                    key={corner}
                                                    className={`absolute w-3 h-3 bg-green-500 rounded-full shadow-lg ${corner === 'tl' ? '-top-1 -left-1 cursor-nwse-resize' :
                                                            corner === 'tr' ? '-top-1 -right-1 cursor-nesw-resize' :
                                                                corner === 'bl' ? '-bottom-1 -left-1 cursor-nesw-resize' :
                                                                    '-bottom-1 -right-1 cursor-nwse-resize'
                                                        }`}
                                                    onMouseDown={(e) => handleResizeStart(e, corner)}
                                                    onTouchStart={(e) => handleResizeTouchStart(e, corner)}
                                                    style={{ border: '1.5px solid white' }}
                                                />
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            )}
                        </>
                    )}
                </main>

                <footer className="flex-none px-4 pb-8 pt-2 relative">
                    <Toolbar 
                        activeMenu={menuState.activeMenu} 
                        handleConfirmMenu={menuState.handleConfirmMenu} 
                        className="relative z-20"
                        onUndo={handleUndo}
                        onRedo={handleRedo}
                        canUndo={canUndo}
                        canRedo={canRedo}
                    />

                    <div className="w-full mt-4 relative z-10">
                        <Render3DBar
                            onModeChange={(mode) => {
                                pushSnapshot();
                                setBlendMode(mode);
                            }}
                            isRelighting={isRelighting}
                            onConfirm={async (barMode) => {
                                const mode = blendMode || barMode || "standard";

                                if (mode === "3d" && backgroundCanvasRef.current && threeCanvasRef.current) {
                                    try {
                                        setIsRelighting(true);
                                        setRelightingError(null);

                                        const bgCanvas = backgroundCanvasRef.current;
                                        const threeCanvas = threeCanvasRef.current;
                                        const width = bgCanvas.width;
                                        const height = bgCanvas.height;
                                        const exportCanvas = document.createElement("canvas");
                                        exportCanvas.width = width;
                                        exportCanvas.height = height;
                                        const exportCtx = exportCanvas.getContext("2d", { alpha: true });

                                        // Composite 3D + background
                                        exportCtx.drawImage(bgCanvas, 0, 0, width, height);
                                        exportCtx.drawImage(threeCanvas, 0, 0, width, height);

                                        // Convert canvas to blob for LBM API
                                        const compositedBlob = await new Promise((resolve) => {
                                            exportCanvas.toBlob(resolve, 'image/png', 1.0);
                                        });

                                        // Create file from blob
                                        const compositedFile = new File(
                                            [compositedBlob],
                                            'composited-3d.png',
                                            { type: 'image/png' }
                                        );

                                        console.log('Sending composited image to LBM for relighting...');

                                        // Call LBM API for relighting
                                        const relitBlob = await processRelighting(compositedFile, 1);

                                        // Convert relit blob to data URL
                                        const relitDataURL = await new Promise((resolve) => {
                                            const reader = new FileReader();
                                            reader.onloadend = () => resolve(reader.result);
                                            reader.readAsDataURL(relitBlob);
                                        });

                                        console.log('Relighting complete!');
                                        setIsRelighting(false);
                                        navigate("/photoroom", { state: { compositedImage: relitDataURL } });
                                    } catch (error) {
                                        console.error('Error during 3D blend and relighting:', error);
                                        setRelightingError(error.message || 'Failed to process 3D blend');
                                        setIsRelighting(false);
                                        toast.error("Failed to process 3D blend: " + error.message);
                                    }
                                } else if (mode === "standard") {
                                    try {
                                        if (!imageHandler.imageSrc || !overlayImage) {
                                            toast.error("Missing background or overlay");
                                            return;
                                        }
                                        const bgImg = new Image();
                                        bgImg.onload = () => {
                                            const bgWidth = bgImg.width;
                                            const bgHeight = bgImg.height;
                                            const exportCanvas = document.createElement("canvas");
                                            exportCanvas.width = bgWidth;
                                            exportCanvas.height = bgHeight;
                                            const exportCtx = exportCanvas.getContext("2d");

                                            exportCtx.drawImage(bgImg, 0, 0, bgWidth, bgHeight);

                                            const overlayImg = new Image();
                                            overlayImg.onload = () => {
                                                exportCtx.save();
                                                const overlayX = (overlayPosition.x / 100) * bgWidth;
                                                const overlayY = (overlayPosition.y / 100) * bgHeight;

                                                exportCtx.translate(overlayX, overlayY);
                                                exportCtx.rotate((overlayRotation * Math.PI) / 180);
                                                exportCtx.scale(overlayScale, overlayScale);

                                                const uiMaxSize = 200;
                                                const bgDisplaySize = Math.min(bgWidth, bgHeight);
                                                const scaleFactor = bgDisplaySize / 500;

                                                let drawWidth, drawHeight;
                                                const overlayAspect = overlayImg.width / overlayImg.height;

                                                if (overlayAspect > 1) {
                                                    drawWidth = uiMaxSize * scaleFactor;
                                                    drawHeight = (uiMaxSize * scaleFactor) / overlayAspect;
                                                } else {
                                                    drawHeight = uiMaxSize * scaleFactor;
                                                    drawWidth = (uiMaxSize * scaleFactor) * overlayAspect;
                                                }

                                                exportCtx.drawImage(overlayImg, -drawWidth / 2, -drawHeight / 2, drawWidth, drawHeight);
                                                exportCtx.restore();

                                                const dataURL = exportCanvas.toDataURL("image/png", 1.0);
                                                navigate("/photoroom", { state: { compositedImage: dataURL } });
                                            };
                                            overlayImg.src = URL.createObjectURL(overlayImage);
                                        };
                                        bgImg.src = imageHandler.imageSrc;
                                    } catch (error) {
                                        console.error(error);
                                    }
                                }
                            }}
                            onCancel={() => navigate("/photoroom", { state: { backgroundImage: imageHandler.imageSrc } })}
                        />
                    </div>
                </footer>
            </div>
            {menuState.showStylesMenu && (
                <StylesMenu
                    onBack={() => menuState.setShowStylesMenu(false)}
                    imageSrc={imageHandler.imageSrc}
                    onImageUpdate={(newImageUrl) => {
                        imageHandler.setImageSrc(newImageUrl);
                        menuState.setShowStylesMenu(false);
                    }}
                />
            )}
        </>
    );
}