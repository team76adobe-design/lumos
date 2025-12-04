/**
 * CSS styles for crop tool UI
 */
export const CROP_TOOL_STYLES = `
  .crop-box {
    position: absolute;
    border: 2px solid rgba(255,255,255,0.95);
    border-radius: 4px;
    z-index: 1000 !important;
    background: transparent;
    touch-action: none;
  }

  .outside-dim.piece {
    background: rgba(0,0,0,0.55);
    position: absolute;
    pointer-events: none;
    z-index: 900 !important;
    transition: opacity 0.1s linear;
  }

  .crop-image {
    position: relative;
    z-index: 800 !important;
  }

  .grid-overlay {
    inset: 0;
    position: absolute;
    pointer-events: none;
    display: none;
  }

  .crop-box.show-grid {}

  .crop-box.show-grid .grid-overlay {
    display: block;
  }

  .grid-overlay .line {
    position: absolute;
    background: rgba(255,255,255,0.25);
  }

  .grid-overlay .v1 {
    left: 33.33%;
    top: 0;
    bottom: 0;
    width: 1px;
  }

  .grid-overlay .v2 {
    left: 66.66%;
    top: 0;
    bottom: 0;
    width: 1px;
  }

  .grid-overlay .h1 {
    top: 33.33%;
    left: 0;
    right: 0;
    height: 1px;
  }

  .grid-overlay .h2 {
    top: 66.66%;
    left: 0;
    right: 0;
    height: 1px;
  }

  .corner-handle {
    width: 40px;
    height: 40px;
    position: absolute;
    z-index: 1100;
    pointer-events: auto;
  }

  .corner-handle::before,
  .corner-handle::after {
    content: "";
    position: absolute;
    background: white;
    border-radius: 1px;
  }

  .corner-handle.nw {
    top: -2px;
    left: -2px;
  }

  .corner-handle.nw::before {
    width: 20px;
    height: 4px;
    top: 0;
    left: 0;
  }

  .corner-handle.nw::after {
    width: 4px;
    height: 20px;
    top: 0;
    left: 0;
  }

  .corner-handle.ne {
    top: -2px;
    right: -2px;
  }

  .corner-handle.ne::before {
    width: 20px;
    height: 4px;
    top: 0;
    right: 0;
  }

  .corner-handle.ne::after {
    width: 4px;
    height: 20px;
    top: 0;
    right: 0;
  }

  .corner-handle.sw {
    bottom: -2px;
    left: -2px;
  }

  .corner-handle.sw::before {
    width: 20px;
    height: 4px;
    bottom: 0;
    left: 0;
  }

  .corner-handle.sw::after {
    width: 4px;
    height: 20px;
    bottom: 0;
    left: 0;
  }

  .corner-handle.se {
    bottom: -2px;
    right: -2px;
  }

  .corner-handle.se::before {
    width: 20px;
    height: 4px;
    bottom: 0;
    right: 0;
  }

  .corner-handle.se::after {
    width: 4px;
    height: 20px;
    bottom: 0;
    right: 0;
  }

  .edge-handle {
    background: white;
    border-radius: 2px;
    position: absolute;
    z-index: 1100;
  }

  .edge-handle.n {
    top: -3px;
    left: 50%;
    width: 40px;
    height: 6px;
    transform: translateX(-50%);
  }

  .edge-handle.s {
    bottom: -3px;
    left: 50%;
    width: 40px;
    height: 6px;
    transform: translateX(-50%);
  }

  .edge-handle.e {
    right: -3px;
    top: 50%;
    width: 6px;
    height: 40px;
    transform: translateY(-50%);
  }

  .edge-handle.w {
    left: -3px;
    top: 50%;
    width: 6px;
    height: 40px;
    transform: translateY(-50%);
  }

  .confirm-row {
    position: absolute;
    bottom: 100px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    gap: 12px;
    z-index: 1500;
  }
`;

/**
 * Aspect ratio options with their labels and ratios
 */
export const ASPECT_RATIOS = {
  Original: "original",
  "9:16": 9 / 16,
  "1:1": 1,
  "3:4": 3 / 4,
  "4:3": 4 / 3,
  "16:9": 16 / 9,
  "5:4": 5 / 4,
  Freeform: null,
};

/**
 * Initial transform state
 */
export const INITIAL_TRANSFORM = {
  x: 0,
  y: 0,
  scale: 1,
  rotation: 0,
  flipX: 1,
  flipY: 1,
};

/**
 * Scale constraints
 */
export const SCALE_LIMITS = {
  MIN: 0.05,
  MAX: 4,
};

/**
 * Minimum crop box size
 */
export const MIN_CROP_SIZE = 40;
