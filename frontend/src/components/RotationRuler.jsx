import { useRef, useState } from "react";

export function RotationRuler({ rotation, onRotationChange }) {
  const rulerRef = useRef(null);
  const moveInfo = useRef({ startX: 0, rect: null });
  const [ismoveging, setIsmoveging] = useState(false);

  // Normalize - ensure rotation always stays within -180 to 180
  const normalizeRotation = (rot) => {
    let normalized = rot % 360;
    if (normalized > 180) normalized -= 360;
    if (normalized < -180) normalized += 360;
    return normalized;
  };

  const updateRotation = (clientX, rect) => {
    if (!rect) return;

    const relativeX = clientX - rect.left;
    const percentage = Math.max(0, Math.min(1, relativeX / rect.width));

    const newRotation = percentage * 360 - 180;
    const rounded = Math.round(newRotation * 10) / 10;

    onRotationChange(rounded);
  };

  const handlePointerDown = (e) => {
    if (!rulerRef.current) return;

    e.currentTarget.setPointerCapture(e.pointerId);
    setIsmoveging(true);

    const rect = rulerRef.current.getBoundingClientRect();
    moveInfo.current = { rect };

    updateRotation(e.clientX, rect);
  };

  const handlePointerMove = (e) => {
    if (ismoveging && moveInfo.current.rect) {
      e.preventDefault();
      updateRotation(e.clientX, moveInfo.current.rect);
    }
  };

  const handlePointerUp = (e) => {
    setIsmoveging(false);
    e.currentTarget.releasePointerCapture(e.pointerId);
  };

  const normalizedRotation = normalizeRotation(rotation);
  const indicatorPosition = ((normalizedRotation + 180) / 360) * 100;

  const ticks = [];
  for (let i = 0; i <= 48; i++) {
    const isLargeTick = i % 8 === 0;
    const leftPosition = (i / 48) * 100;

    ticks.push(
      <div
        key={i}
        className="absolute bg-white top-0 pointer-events-none"
        style={{
          left: `${leftPosition}%`,
          height: isLargeTick ? "20px" : "12px",
          width: "1px",
          opacity: isLargeTick ? 0.6 : 0.3,
        }}
      ></div>
    );
  }

  return (
    <div className="w-full flex justify-center mt-3 mb-6 select-none">
      <div className="relative h-[32px] w-[95%]">
        {/* Ruler Area */}
        <div
          ref={rulerRef}
          className="absolute h-[40px] left-0 top-0 w-full cursor-ew-resize touch-none"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
        >
          {ticks}
        </div>

        {/* Labels */}
        <div className="absolute h-[16px] top-[32px] w-full pointer-events-none text-[12px] whitespace-nowrap">
          <p className="absolute left-0 -translate-x-1/2 text-neutral-500">
            -180°
          </p>
          <p className="absolute left-[25%] -translate-x-1/2 text-neutral-500">
            -90°
          </p>
          <p className="absolute left-[50%] -translate-x-1/2 text-[#fe5959]">
            0°
          </p>
          <p className="absolute left-[75%] -translate-x-1/2 text-neutral-500">
            90°
          </p>
          {/* Added whitespace-nowrap to parent, so this 180° won't break */}
          <p className="absolute left-[100%] -translate-x-1/2 text-neutral-500">
            180°
          </p>
        </div>

        {/* Indicator */}
        <div
          className={`absolute bg-[#fe5959] h-[32px] top-0 w-[2px] pointer-events-none z-10 ${
            ismoveging ? "" : "transition-all duration-100 ease-out"
          }`}
          style={{
            left: `${indicatorPosition}%`,
            transform: "translateX(-50%)",
          }}
        ></div>

        {/* Floating Value Bubble */}
        <div
          className={`absolute top-[-22px] bg-[#fe5959] px-2 py-1 rounded text-white text-[10px] pointer-events-none z-20 ${
            ismoveging ? "" : "transition-all duration-100 ease-out"
          }`}
          style={{
            left: `${indicatorPosition}%`,
            transform: "translateX(-50%)",
          }}
        >
          {Math.round(normalizedRotation)}°
        </div>
      </div>
    </div>
  );
}