import { hitBackend, PORTS } from "./apiConstants";

const PORT = PORTS.QUILL;

/* ======================================================
   ONE FUNCTION: LOAD → RUN → UNLOAD
   mode = "json"      → send JSON payload
   mode = "formdata"  → send FormData with files
====================================================== */
export const runMagicQuillFullCycle = async ({
  mode = "json",

  // JSON params
  original_image,
  total_mask,
  add_color_image = null,
  add_edge_image,
  remove_edge_image,
  positive_prompt = null,
  negative_prompt = null,

  // FormData
  formData = null,
}) => {
  try {
    /* --------------------------
       1️⃣ LOAD MODEL
    -------------------------- */
    const loadData = new FormData();
    loadData.append("device", "cuda");
    loadData.append("revision", "");

    const loadResp = await hitBackend(PORT, "/load", {
      method: "POST",
      body: loadData,
    });

    // console.log("[MagicQuill] MODEL LOADED");

    /* --------------------------
       2️⃣ RUN MODEL
    -------------------------- */

    let runResp;

    if (mode === "formdata") {
      // console.log("[MagicQuill] RUNNING IN FORMDATA MODE");

      runResp = await hitBackend(PORT, "/edit_image", {
        method: "POST",
        body: formData, // 🟢 send actual files
      });
    } else {
      // console.log("[MagicQuill] RUNNING IN JSON MODE");

      const payload = {
        original_image,
        total_mask,
        add_color_image,
        add_edge_image,
        remove_edge_image,
        positive_prompt,
        negative_prompt,
      };

      runResp = await hitBackend(PORT, "/edit_image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    }

    // Expect PNG/Blob
    const blob = await runResp.blob();
    const finalOutput = URL.createObjectURL(blob);

    // console.log("[MagicQuill] RUN DONE");

    /* --------------------------
       3️⃣ UNLOAD MODEL
    -------------------------- */
    // Not required every time — huge slowdown
    const unloadResp = await hitBackend(PORT, "/unload", { method: "POST" });

    return finalOutput;
  } catch (error) {
    // console.error("[MagicQuill] Full cycle error:", error);
    throw error;
  }
};
