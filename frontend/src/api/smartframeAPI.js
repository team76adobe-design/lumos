// Import helper to hit the backend
import { hitBackend } from "./apiConstants";

// This is the port where your SmartFrame API is running
const PORT = 8006;


export const loadSmartFrameModel = async () => {
  try {
    await hitBackend(PORT, "/load", { method: "POST" });
    console.log("SmartFrame model loaded.");
  } catch (error) {
    console.error("Failed to load SmartFrame model:", error);
  }
};

export const runSmartFrame = async (imageFile) => {
  // Ensure model is loaded (optional if pre-loaded, but safe to keep or remove if backend handles it)
  // await loadSmartFrameModel(); 

  // Prepare form-data for /run
  const fd = new FormData();
  fd.append("image", imageFile);

  // RUN inference
  const res = await hitBackend(PORT, "/run", {
    method: "POST",
    body: fd,
  });

  const jsonOutput = await res.json();

  // UNLOAD model (optional)
  await hitBackend(PORT, "/unload", { method: "POST" });

  // Return inference result
  return jsonOutput;
};


// Optional: preload the RL model on backend
export const loadSmartFrame = async () => {
  const res = await hitBackend(PORT, "/load", { method: "POST" });
  return res.json();
};
