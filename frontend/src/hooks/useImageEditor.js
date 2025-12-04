import { useState, useMemo } from 'react';

export const useImageEditor = () => {
    const [editValues, setEditValues] = useState({
        brightness: 100,
        exposure: 0,
        brilliance: 100,
        highlights: 0,
        contrast: 100,
        saturation: 100,
        vibrancy: 100,
        warmth: 0,
        noise_reduction: 0,
        sharpness: 100,
        definition: 100,
    });

    const handleSliderChange = (id, value) => {
        setEditValues((prev) => ({ ...prev, [id]: value }));
    };

    const handleAutoAdjust = () => {
        setEditValues({
            brightness: Math.random() * 100 + 50,
            exposure: Math.random() * 200 - 100,
            brilliance: Math.random() * 100 + 50,
            highlights: Math.random() * 200 - 100,
            contrast: Math.random() * 100 + 50,
            saturation: Math.random() * 100 + 50,
            vibrancy: Math.random() * 100 + 50,
            warmth: Math.random() * 200 - 100,
            noise_reduction: Math.random() * 100,
            sharpness: Math.random() * 100 + 50,
            definition: Math.random() * 100 + 50,
        });
    };

    const resetEditValues = () => {
        setEditValues({
            brightness: 100,
            exposure: 0,
            brilliance: 100,
            highlights: 0,
            contrast: 100,
            saturation: 100,
            vibrancy: 100,
            warmth: 0,
            noise_reduction: 0,
            sharpness: 100,
            definition: 100,
        });
    };

    // Normalized values for pixel processor (range [-1,1])
    const normalized = useMemo(() => ({
        exposure: editValues.exposure / 100,
        brightness: (editValues.brightness - 100) / 100,
        contrast: (editValues.contrast - 100) / 100,
        brilliance: (editValues.brilliance - 100) / 100,
        highlights: editValues.highlights / 100,
        saturation: (editValues.saturation - 100) / 100,
        vibrancy: (editValues.vibrancy - 100) / 100,
        warmth: editValues.warmth / 100,
        noise_reduction: editValues.noise_reduction / 100,
        sharpness: (editValues.sharpness - 100) / 100,
        definition: (editValues.definition - 100) / 100,
    }), [editValues]);

    return {
        editValues,
        normalized,   // ← use this inside useImageEditorProcessor
        handleSliderChange,
        handleAutoAdjust,
        resetEditValues,
    };
};
