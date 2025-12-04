import React, { useState, useEffect } from "react";
import ElasticSlider from "../ElasticSlider";
import tinycolor from "tinycolor2";

const ColorSliders = ({ color, onChange }) => {
  const [hsv, setHsv] = useState(tinycolor(color).toHsv());

  useEffect(() => {
    const newHsv = tinycolor(color).toHsv();
    // Only update if significantly different to avoid loop/jitter
    // But here we trust the parent passes the updated color back
    setHsv(newHsv);
  }, [color]);

  const handleHChange = (h) => {
    const newHsv = { ...hsv, h };
    setHsv(newHsv);
    onChange(tinycolor(newHsv).toHexString());
  };

  const handleSChange = (s) => {
    const newHsv = { ...hsv, s: s / 100 };
    setHsv(newHsv);
    onChange(tinycolor(newHsv).toHexString());
  };

  const handleVChange = (v) => {
    const newHsv = { ...hsv, v: v / 100 };
    setHsv(newHsv);
    onChange(tinycolor(newHsv).toHexString());
  };

  // Dynamic Gradients
  const hueColor = tinycolor({ h: hsv.h, s: 1, v: 1 }).toHexString();
  const satStart = tinycolor({ h: hsv.h, s: 0, v: 1 }).toHexString(); // White
  const satEnd = tinycolor({ h: hsv.h, s: 1, v: 1 }).toHexString();   // Pure Color
  const valStart = "#000000";
  const valEnd = tinycolor({ h: hsv.h, s: hsv.s, v: 1 }).toHexString();

  const satGradient = `linear-gradient(to right, ${satStart}, ${satEnd})`;
  const valGradient = `linear-gradient(to right, ${valStart}, ${valEnd})`;

  return (
    <div className="w-full flex flex-col gap-2 items-center justify-center py-1 max-w-[600px] mx-auto">
      {/* Hue Slider */}
      <div className="w-full flex items-center gap-4">
        <div className="flex-1">
             <ElasticSlider
                defaultValue={hsv.h}
                startingValue={0}
                maxValue={360}
                value={hsv.h}
                onChange={handleHChange}
                leftIcon={<span className="text-white text-[12px]">0</span>}
                rightIcon={<span className="text-white text-[12px]">255</span>}
                className="w-full"
                showLabel={false}
                trackStyle={{ background: 'linear-gradient(to right, #f00 0%, #ff0 17%, #0f0 33%, #0ff 50%, #00f 67%, #f0f 83%, #f00 100%)' }}
                gradientStyle={{ background: 'transparent' }}
                showThumb={true}
                thumbColor={hueColor}
             />
        </div>
      </div>

      {/* Saturation Slider */}
      <div className="w-full flex items-center gap-4">
        <div className="flex-1">
             <ElasticSlider
                defaultValue={hsv.s * 100}
                startingValue={0}
                maxValue={100}
                value={hsv.s * 100}
                onChange={handleSChange}
                leftIcon={<span className="text-white text-[12px]">0</span>}
                rightIcon={<span className="text-white text-[12px]">100</span>}
                className="w-full"
                showLabel={false}
                gradientStyle={{ background: satGradient }}
             />
        </div>
      </div>

      {/* Value Slider */}
      <div className="w-full flex items-center gap-4">
        <div className="flex-1">
             <ElasticSlider
                defaultValue={hsv.v * 100}
                startingValue={0}
                maxValue={100}
                value={hsv.v * 100}
                onChange={handleVChange}
                leftIcon={<span className="text-white text-[12px]">0</span>}
                rightIcon={<span className="text-white text-[12px]">100</span>}
                className="w-full"
                showLabel={false}
                gradientStyle={{ background: valGradient }}
             />
        </div>
      </div>
    </div>
  );
};

export default ColorSliders;
