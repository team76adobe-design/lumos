import React, { useRef, useEffect, useState, useCallback } from "react";
import PropTypes from "prop-types";

import {
  colourToRgbObj,
  getEffectiveRadius,
  calculateBounds,
  produceRgbShades,
  convertObjToString
} from "../../utils/utils";

import hexStrings from "../../utils/hexStrings";

const fullCircle = 2 * Math.PI;
const quarterCircle = fullCircle / 4;

export default function ColourWheel({
  radius,
  lineWidth,
  colours = hexStrings,
  shades = 16,
  padding = 0,
  dynamicCursor = true,
  spacers = {
    colour: "#000",
    shadowColour: "rgba(0,0,0,0.5)",
    shadowBlur: 5,
  },
  onColourSelected,
  preset = false,
  presetColour,
}) {
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);
  const animationRef = useRef(null);

  const [rgb, setRgb] = useState(null);
  const [innerWheelOpen, setInnerWheelOpen] = useState(false);
  const [centerCircleOpen, setCenterCircleOpen] = useState(false);

  // Precomputed radii & bounds
  const outerWheelRadius = radius;
  const innerWheelRadius = outerWheelRadius - lineWidth - padding;
  const centerCircleRadius = innerWheelRadius - lineWidth - padding;

  const firstSpacerRadius = outerWheelRadius - lineWidth;
  const secondSpacerRadius = innerWheelRadius - lineWidth;

  const outerBounds = calculateBounds(
    radius - lineWidth,
    radius
  );

  const innerBounds = calculateBounds(
    innerWheelRadius - lineWidth,
    innerWheelRadius
  );

  const centerBounds = calculateBounds(
    0,
    centerCircleRadius
  );

  const firstSpacerBounds = calculateBounds(
    firstSpacerRadius - padding,
    firstSpacerRadius
  );

  const secondSpacerBounds = calculateBounds(
    secondSpacerRadius - padding,
    secondSpacerRadius
  );

  // ------------------------------
  // Animation Helper
  // ------------------------------
  const animateInnerWheel = useCallback(() => {
    let start = null;
    const duration = 400;

    const step = (timestamp) => {
      if (!start) start = timestamp;
      const progress = timestamp - start;
      const pct = Math.min(progress / duration, 1);
      const ease = 1 - Math.pow(1 - pct, 4); // easeOutQuart

      drawInnerWheel(ease);

      if (progress < duration) {
        animationRef.current = requestAnimationFrame(step);
      } else {
        drawCenterCircle(); // Ensure center is drawn at end
      }
    };

    if (animationRef.current) cancelAnimationFrame(animationRef.current);
    animationRef.current = requestAnimationFrame(step);
  }, []);

  // ------------------------------
  // Canvas Initialization
  // ------------------------------
  const initCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    ctxRef.current = ctx;
    ctx.clearRect(0, 0, radius * 2, radius * 2);

    drawOuterWheel();
    drawSpacers();
  }, []);

  // ------------------------------
  // Utility: mouse position
  // ------------------------------
  const getRelativeMousePos = (clientX, clientY) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();

    const onCanvas = {
      x: clientX - rect.left,
      y: clientY - rect.top,
    };

    const dist = Math.sqrt(
      (onCanvas.x - radius) * (onCanvas.x - radius) +
      (onCanvas.y - radius) * (onCanvas.y - radius)
    );

    return { onCanvas, fromCenter: dist };
  };

  // ------------------------------
  // Outer wheel (Hue wheel)
  // ------------------------------
  const drawOuterWheel = useCallback(() => {
    const ctx = ctxRef.current;
    if (!ctx) return;

    const width = radius * 2;
    const effectiveRadius = getEffectiveRadius(radius, lineWidth);

    const rgbArr = colours.map((c) => colourToRgbObj(c));

    rgbArr.forEach((rgb, i) => {
      ctx.beginPath();

      const start = (fullCircle / rgbArr.length) * i;
      const end = (fullCircle / rgbArr.length) * (i + 1);

      ctx.arc(width / 2, width / 2, effectiveRadius, start, end);
      ctx.lineWidth = lineWidth;
      ctx.strokeStyle = `rgb(${rgb.r}, ${rgb.g}, ${rgb.b})`;
      
      // Add subtle glow
      ctx.shadowColor = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.4)`;
      ctx.shadowBlur = 4;
      
      ctx.stroke();
      
      ctx.shadowBlur = 0; // Reset
    });
  }, [radius, lineWidth, colours]);

  // ------------------------------
  // Spacers
  // ------------------------------
  const drawSpacers = useCallback(() => {
    const ctx = ctxRef.current;
    if (!ctx || !spacers) return;

    const width = radius * 2;

    const drawOne = (r) => {
      const er = getEffectiveRadius(r, padding);

      ctx.beginPath();
      ctx.arc(width / 2, width / 2, er, 0, fullCircle);
      ctx.lineWidth = padding;
      ctx.shadowColor = spacers.shadowColour;
      ctx.shadowBlur = spacers.shadowBlur;
      ctx.strokeStyle = spacers.colour;
      ctx.stroke();
      ctx.shadowColor = "transparent";
    };

    drawOne(firstSpacerRadius);
    drawOne(secondSpacerRadius);
  }, [radius, padding, spacers]);

  // ------------------------------
  // Inner shades wheel
  // ------------------------------
  const drawInnerWheel = useCallback(
    (animPct = 1) => {
      if (!rgb) return;

      const ctx = ctxRef.current;
      const width = radius * 2;

      ctx.clearRect(0, 0, width, width);
      drawOuterWheel();
      drawSpacers();

      const rgbShades = produceRgbShades(rgb.r, rgb.g, rgb.b, shades);
      const effectiveRadius = getEffectiveRadius(innerWheelRadius, lineWidth);

      ctx.save();
      // Animate opacity and slight scale/rotation if desired
      ctx.globalAlpha = animPct;
      
      // Optional: Scale effect from center
      // ctx.translate(width/2, width/2);
      // ctx.scale(0.8 + 0.2 * animPct, 0.8 + 0.2 * animPct);
      // ctx.translate(-width/2, -width/2);

      rgbShades.forEach((shade, i) => {
        ctx.beginPath();

        const start = (fullCircle / rgbShades.length) * i + quarterCircle;
        const end =
          (fullCircle / rgbShades.length) * (i + 1) + Math.PI / 2;

        ctx.arc(width / 2, width / 2, effectiveRadius, start, end);
        ctx.lineWidth = lineWidth;
        ctx.strokeStyle = `rgb(${shade.r}, ${shade.g}, ${shade.b})`;
        ctx.stroke();
      });
      
      ctx.restore();
    },
    [rgb, radius, shades, innerWheelRadius, lineWidth, drawOuterWheel, drawSpacers]
  );

  // ------------------------------
  // Center selected color circle
  // ------------------------------
  const drawCenterCircle = useCallback(() => {
    if (!rgb) return;

    const ctx = ctxRef.current;
    const width = radius * 2;

    ctx.beginPath();
    ctx.arc(width / 2, width / 2, centerCircleRadius, 0, fullCircle);
    ctx.fillStyle = `rgb(${rgb.r}, ${rgb.g}, ${rgb.b})`;
    
    // Glow for center
    ctx.shadowColor = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.6)`;
    ctx.shadowBlur = 15;
    
    ctx.fill();
    ctx.shadowBlur = 0;

    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(255,255,255,0.2)";
    ctx.stroke();
  }, [rgb, radius, centerCircleRadius]);

  // ------------------------------
  // Mouse events
  // ------------------------------
  const handleHover = (e) => {
    if (!dynamicCursor) return;

    const { fromCenter } = getRelativeMousePos(e.clientX, e.clientY);

    const canvas = canvasRef.current;

    if (outerBounds.inside(fromCenter)) {
      canvas.style.cursor = "crosshair";
    } else if (innerBounds.inside(fromCenter) && innerWheelOpen) {
      canvas.style.cursor = "crosshair";
    } else if (centerBounds.inside(fromCenter) && centerCircleOpen) {
      canvas.style.cursor = "pointer";
    } else {
      canvas.style.cursor = "auto";
    }
  };

  const handleClick = (e) => {
    const { onCanvas, fromCenter } = getRelativeMousePos(
      e.clientX,
      e.clientY
    );

    const ctx = ctxRef.current;
    const [r, g, b] = ctx.getImageData(onCanvas.x, onCanvas.y, 1, 1).data;
    const newRgb = { r, g, b };
    const rgbString = convertObjToString(newRgb);

    // ----- OUTER RING -----
    if (outerBounds.inside(fromCenter)) {
      setRgb(newRgb);
      setInnerWheelOpen(true);
      setCenterCircleOpen(true);

      onColourSelected?.(rgbString);
      animateInnerWheel(); // Trigger animation
      return;
    }

    // ----- INNER RING -----
    if (innerBounds.inside(fromCenter) && innerWheelOpen) {
      setRgb(newRgb);
      setCenterCircleOpen(true);

      onColourSelected?.(rgbString);
      drawCenterCircle();
      return;
    }
  };

  // ------------------------------
  // Component Lifecycle (hooks)
  // ------------------------------
  useEffect(() => {
    if (!canvasRef.current) return;

    const ctx = canvasRef.current.getContext("2d");
    ctxRef.current = ctx;

    // Initial Draw
    if (preset) {
      const startRgb = colourToRgbObj(presetColour);
      setRgb(startRgb);
      drawOuterWheel();
      drawSpacers();
      drawInnerWheel();
      drawCenterCircle();
    } else {
      drawOuterWheel();
      drawSpacers();
    }

    // Mount Animation (Pop-in)
    let start = null;
    const duration = 600;
    const animateMount = (timestamp) => {
      if (!start) start = timestamp;
      const progress = timestamp - start;
      const pct = Math.min(progress / duration, 1);
      // Elastic pop effect
      const ease = pct === 0 ? 0 : pct === 1 ? 1 : 
        Math.pow(2, -10 * pct) * Math.sin((pct * 10 - 0.75) * (2 * Math.PI) / 3) + 1;

      if (canvasRef.current) {
        canvasRef.current.style.transform = `scale(${ease})`;
        canvasRef.current.style.opacity = Math.min(pct * 2, 1); // Fade in faster
      }

      if (progress < duration) {
        requestAnimationFrame(animateMount);
      }
    };
    requestAnimationFrame(animateMount);

  }, []);

  return (
    <canvas
      ref={canvasRef}
      width={radius * 2}
      height={radius * 2}
      onMouseMove={handleHover}
      onClick={handleClick}
      style={{ touchAction: "none" }}
    />
  );
}

ColourWheel.propTypes = {
  radius: PropTypes.number.isRequired,
  lineWidth: PropTypes.number.isRequired,
  colours: PropTypes.array,
  shades: PropTypes.number,
  padding: PropTypes.number,
  dynamicCursor: PropTypes.bool,
  spacers: PropTypes.object,
  onColourSelected: PropTypes.func,
  preset: PropTypes.bool,
  presetColour: PropTypes.string,
};

ColourWheel.defaultProps = {
  colours: hexStrings,
  shades: 16,
  padding: 0,
  dynamicCursor: true,
  preset: false,
  spacers: {
    colour: "#000",
    shadowColour: "rgba(0,0,0,0.5)",
    shadowBlur: 5,
  },
};
