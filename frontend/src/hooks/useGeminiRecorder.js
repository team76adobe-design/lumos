import { useState, useRef, useEffect } from "react";

export function useGeminiRecorder() {
  const MAX_POINTS = 57;
  const TIME_WINDOW_SECONDS = 1;
  const POINTS_PER_UPDATE = 3;
  const INTERVAL_MS = (TIME_WINDOW_SECONDS * 1000 * POINTS_PER_UPDATE) / MAX_POINTS;

  const [isRecording, setIsRecording] = useState(false);
  const [waveData, setWaveData] = useState(Array(MAX_POINTS).fill(null));
  const [transcribedText, setTranscribedText] = useState("");

  const audioCtxRef = useRef(null);
  const analyserRef = useRef(null);
  const dataArrayRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);
  const animationRef = useRef(null);
  const lastUpdateRef = useRef(0);
  const currentAmpsRef = useRef(Array(POINTS_PER_UPDATE).fill(0));
  const ampIndexRef = useRef(0);
  const shouldSaveRef = useRef(false);

  const resetWave = () => setWaveData(Array(MAX_POINTS).fill(null));

  const stopRecording = () => {
    setIsRecording(false);

    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }

    if (audioCtxRef.current) {
      try {
        if (audioCtxRef.current.state !== "closed") {
          audioCtxRef.current.close();
        }
      } catch (e) {
        console.warn("Error closing audio context", e);
      }
      audioCtxRef.current = null;
    }
    resetWave();
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const audioCtx = new AudioCtx();
      if (audioCtx.state === "suspended") await audioCtx.resume();

      audioCtxRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.8;
      source.connect(analyser);
      analyserRef.current = analyser;

      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      dataArrayRef.current = dataArray;

      // MediaRecorder
      if (typeof MediaRecorder !== "undefined") {
        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorderRef.current = mediaRecorder;
        recordedChunksRef.current = [];

        mediaRecorder.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) recordedChunksRef.current.push(e.data);
        };

        mediaRecorder.onstop = () => {
          const chunks = recordedChunksRef.current;
          if (shouldSaveRef.current && chunks.length) {
            const mimeType = mediaRecorder.mimeType || "audio/webm;codecs=opus";
            const blob = new Blob(chunks, { type: mimeType });
            const formData = new FormData();
            formData.append("file", blob, "recording.webm");

            fetch("https://a53af8f43400.ngrok-free.app/transcribe", {
              method: "POST",
              body: formData,
            })
              .then((res) => res.json())
              .then((data) => {
                console.log(data);
                console.log("✅ Transcription:", data.transcription);
                if (data.transcription) setTranscribedText(data.transcription);
              })
              .catch((err) => console.error("❌ Upload failed:", err));
          }
          shouldSaveRef.current = false;
          recordedChunksRef.current = [];
          mediaRecorderRef.current = null;
          stopRecording();
        };

        mediaRecorder.start();
      }

      resetWave();
      setIsRecording(true);
      lastUpdateRef.current = performance.now();
      ampIndexRef.current = 0;

      const render = () => {
        if (!analyserRef.current || !dataArrayRef.current) return;
        analyserRef.current.getByteTimeDomainData(dataArrayRef.current);

        let sum = 0;
        for (let i = 0; i < dataArrayRef.current.length; i++) {
          const v = dataArrayRef.current[i] / 128.0 - 1.0;
          sum += v * v;
        }
        const rms = Math.sqrt(sum / dataArrayRef.current.length);
        const amp = Math.min(1, rms * 8);

        currentAmpsRef.current[ampIndexRef.current] = amp;
        ampIndexRef.current = (ampIndexRef.current + 1) % POINTS_PER_UPDATE;

        const now = performance.now();
        if (now - lastUpdateRef.current >= INTERVAL_MS) {
          lastUpdateRef.current = now;
          const avgAmp =
            currentAmpsRef.current.reduce((a, b) => a + b, 0) / POINTS_PER_UPDATE;

          setWaveData((prev) => {
            const arr = [...prev];
            arr.shift();
            arr.push(avgAmp);
            return arr;
          });
          ampIndexRef.current = 0;
        }
        animationRef.current = requestAnimationFrame(render);
      };
      render();
    } catch (err) {
      console.error("Mic error:", err);
    }
  };

  const confirmRecording = () => {
    shouldSaveRef.current = true;
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    } else {
      stopRecording();
    }
  };

  const cancelRecording = () => {
    shouldSaveRef.current = false;
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    } else {
      stopRecording();
    }
  };

  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        shouldSaveRef.current = false;
        mediaRecorderRef.current.stop();
      }
      stopRecording();
    };
  }, []);

  return {
    isRecording,
    waveData,
    transcribedText,
    setTranscribedText,
    startRecording,
    confirmRecording,
    cancelRecording,
  };
}