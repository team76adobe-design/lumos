import { animate, motion, useMotionValue, useMotionValueEvent, useTransform } from 'motion/react';
import { useEffect, useRef, useState } from 'react';

const MAX_OVERFLOW = 15;

export default function ElasticSlider({
  defaultValue = 50,
  startingValue = 0,
  maxValue = 100,
  className = '',
  isStepped = false,
  stepSize = 1,
  leftIcon = <>-</>,
  rightIcon = <>+</>,
  value,
  onChange,
  onValueCommit,
  gradient = "from-[#8B3A3D] to-[#FF6469]",
  gradientStyle = undefined,
  trackStyle = undefined,
  showLabel = true,
  showThumb = false,
  thumbColor = "white"
}) {
  return (
    <div className={`flex flex-col items-center justify-center gap-4 w-64 ${className}`}>
      <Slider
        defaultValue={defaultValue}
        startingValue={startingValue}
        maxValue={maxValue}
        isStepped={isStepped}
        stepSize={stepSize}
        leftIcon={leftIcon}
        rightIcon={rightIcon}
        value={value}
        onChange={onChange}
        onValueCommit={onValueCommit}
        gradient={gradient}
        gradientStyle={gradientStyle}
        trackStyle={trackStyle}
        showLabel={showLabel}
        showThumb={showThumb}
        thumbColor={thumbColor}
      />
    </div>
  );
}

function Slider({ defaultValue, startingValue, maxValue, isStepped, stepSize, leftIcon, rightIcon, value, onChange, onValueCommit, gradient, gradientStyle, trackStyle, showLabel, showThumb, thumbColor }) {
  const [internalValue, setInternalValue] = useState(defaultValue);
  const sliderRef = useRef(null);
  const [region, setRegion] = useState('middle');
  const clientX = useMotionValue(0);
  const overflow = useMotionValue(0);
  const scale = useMotionValue(1);

  const currentValue = value !== undefined ? value : internalValue;
  const setValue = (newValue) => {
    if (onChange) onChange(newValue);
    else setInternalValue(newValue);
  };

  useEffect(() => {
    if (value !== undefined) setInternalValue(value);
  }, [value]);

  useMotionValueEvent(clientX, 'change', latest => {
    if (sliderRef.current) {
      const { left, right } = sliderRef.current.getBoundingClientRect();
      let newValue;

      if (latest < left) {
        setRegion('left');
        newValue = left - latest;
      } else if (latest > right) {
        setRegion('right');
        newValue = latest - right;
      } else {
        setRegion('middle');
        newValue = 0;
      }

      overflow.jump(decay(newValue, MAX_OVERFLOW));
    }
  });

  const handlePointerMove = e => {
    if (e.buttons > 0 && sliderRef.current) {
      const { left, width } = sliderRef.current.getBoundingClientRect();
      let newValue = startingValue + ((e.clientX - left) / width) * (maxValue - startingValue);

      if (isStepped) newValue = Math.round(newValue / stepSize) * stepSize;
      newValue = Math.min(Math.max(newValue, startingValue), maxValue);

      setValue(newValue);
      clientX.jump(e.clientX);
    }
  };

  const handlePointerDown = e => {
    handlePointerMove(e);
    e.currentTarget.setPointerCapture(e.pointerId);
  };

  const handlePointerUp = () => {
    animate(overflow, 0, { type: 'spring', bounce: 0.5 });
    if (onValueCommit) onValueCommit();
  };

  const getRangePercentage = () => {
    const totalRange = maxValue - startingValue;
    if (totalRange === 0) return 0;
    return ((currentValue - startingValue) / totalRange) * 100;
  };

  return (
    <>
      <motion.div
        onHoverStart={() => animate(scale, 1.15)}
        onHoverEnd={() => animate(scale, 1)}
        onTouchStart={() => animate(scale, 1.15)}
        onTouchEnd={() => animate(scale, 1)}
        style={{
          scale,
          opacity: useTransform(scale, [1, 1.2], [0.7, 1])
        }}
        className="flex w-full touch-none select-none items-center justify-center gap-4"
      >
        <motion.div
          animate={{
            scale: region === 'left' ? [1, 1.4, 1] : 1,
            transition: { duration: 0.25 }
          }}
          style={{
            x: useTransform(() => (region === 'left' ? -overflow.get() / scale.get() : 0))
          }}
        >
          {leftIcon}
        </motion.div>

        {/* Slider Bar Container */}
        <div
          ref={sliderRef}
          className="relative flex w-full max-w-[240px] flex-grow cursor-grab touch-none select-none items-center py-1.5"
          onPointerMove={handlePointerMove}
          onPointerDown={handlePointerDown}
          onPointerUp={handlePointerUp}
        >
          <motion.div
            style={{
              transformOrigin: useTransform(() => {
                if (sliderRef.current) {
                  const { left, width } = sliderRef.current.getBoundingClientRect();
                  return clientX.get() < left + width / 2 ? 'right' : 'left';
                }
              }),
              height: useTransform(scale, [1, 1.2], [20, 36]),  // THICKER BAR
              marginTop: useTransform(scale, [1, 1.2], [0, -10]),
              marginBottom: useTransform(scale, [1, 1.2], [0, -10])
            }}
            className="flex flex-grow relative"
          >
            <div 
              className="relative h-full flex-grow overflow-hidden rounded-full bg-[#1a1a1a]"
              style={trackStyle}
            >

              {/* Progress Fill */}
              <div
                className={`absolute h-full rounded-full ${!gradientStyle ? `bg-gradient-to-r ${gradient}` : ''}`}
                style={{ 
                  width: `${getRangePercentage()}%`,
                  ...(gradientStyle || {})
                }}
              />

              {/* VERTICAL TICK MARKS (ON TOP) */}
              <div className="absolute inset-0 pointer-events-none z-10">
                {[20, 40, 60, 80].map((pos) => (
                  <div
                    key={pos}
                    className="absolute w-[2px] bg-white/60 rounded"
                    style={{
                      left: `${pos}%`,
                      height: '60%',
                      top: '20%'
                    }}
                  />
                ))}
              </div>

            </div>

            {/* Thumb */}
            {showThumb && (
              <div
                className="absolute top-1/2 w-5 h-5 rounded-full border-2 border-white shadow-sm z-20 pointer-events-none"
                style={{
                  left: `${getRangePercentage()}%`,
                  transform: 'translate(-50%, -50%)',
                  backgroundColor: thumbColor
                }}
              />
            )}
          </motion.div>
        </div>

        <motion.div
          animate={{
            scale: region === 'right' ? [1, 1.4, 1] : 1,
            transition: { duration: 0.25 }
          }}
          style={{
            x: useTransform(() => (region === 'right' ? overflow.get() / scale.get() : 0))
          }}
        >
          {rightIcon}
        </motion.div>
      </motion.div>

      {/* Value Text (Slightly Higher) */}
      {showLabel && (
        <p className="absolute text-gray-500 transform -translate-y-10 text-lg font-medium tracking-wide">
          {Math.round(currentValue)}
        </p>
      )}
    </>
  );
}

function decay(value, max) {
  if (max === 0) return 0;
  const entry = value / max;
  const sigmoid = 2 * (1 / (1 + Math.exp(-entry)) - 0.5);
  return sigmoid * max;
}
