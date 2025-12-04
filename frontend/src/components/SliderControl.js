import React, { useState, useEffect } from "react";
import { editIcons } from "../constants";
import ElasticSlider from './ElasticSlider.js'

const TICK_COUNT = 6;

const SliderControl = ({
  activeEditControl,
  editValues,
  handleSliderChange,
}) => {
  const [localValue, setLocalValue] = useState(editValues[activeEditControl] ?? editIcons.find((c) => c.id === activeEditControl)?.min ?? 0);

  useEffect(() => {
    const control = editIcons.find((c) => c.id === activeEditControl);
    if (control) {
      setLocalValue(editValues[activeEditControl] ?? control.min);
    }
  }, [editValues, activeEditControl]);

  const control = editIcons.find((c) => c.id === activeEditControl);
  if (!control) return null;

  const value = localValue;
  const percentage =
    ((value - control.min) / (control.max - control.min || 1)) * 100;

  const handleInputChange = (e) => {
    setLocalValue(Number(e.target.value));
  };

  const handleMouseUp = () => {
    handleSliderChange(activeEditControl, localValue);
  };

  return (
    <div className="w-full flex items-center justify-between">
      {/* SLIDER WRAPPER — NOW NARROWER */}
      <div className="flex-1 flex items-center justify-center w-full mt-6">
        <ElasticSlider
          className="w-full max-w-[600px]"   // <— add this line

          leftIcon={<span className="text-white">-</span>}
          rightIcon={<span className="text-white">+</span>}
          startingValue={control.min}
          defaultValue={value}
          maxValue={control.max}
          isStepped
          stepSize={1}
          value={value}
          onChange={(newValue) => {
            setLocalValue(newValue);
            handleSliderChange(activeEditControl, newValue);
          }}
          onValueCommit={() => {}}
        />
      </div>

      
    </div>
  );
};

export default SliderControl;
