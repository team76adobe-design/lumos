# --- 1. Imports ---
import torch
import torchaudio
import threading
import time
from fastapi import FastAPI, File, UploadFile
from pyngrok import ngrok
import uvicorn
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# # --- 2. Load model and processor ---
print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "openai/whisper-tiny"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
model.eval() # Set model to evaluation mode
print(f"Model loaded successfully on {device}.")
from fastapi.middleware.cors import CORSMiddleware


# --- 3. STT logic ---
def transcribe_audio_file(file_object):
    """
    Performs STT on an audio file-like object.
    """
    try:
        waveform, sample_rate = torchaudio.load(file_object)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return "[Error: Could not load audio file.]"

    # Resample to 16kHz
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )(waveform)

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    input_audio = waveform.squeeze().numpy()

    print("Processing audio...")
    input_features = processor(input_audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)

    print("Generating transcription...")
    with torch.no_grad():
        predicted_ids = model.generate(input_features, language='en')
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    print(f"Transcription: {transcription[0]}")
    return transcription[0]


# --- 4. FastAPI app ---
app = FastAPI(title="Whisper STT API", description="Speech-to-Text via Whisper")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],  # tighten later if you want
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Send a POST to /transcribe with an audio file."}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    print(f"Received file: {file.filename}")
    text = transcribe_audio_file(file.file)
    return {"filename": file.filename, "transcription": text}


# --- 5. Run server in a thread + ngrok (Your working method) ---
if __name__ == "__main__":
    port = 8122

    def run_server():
        # This will start Uvicorn in this new thread
        print(f"Starting Uvicorn server on port {port}...")
        uvicorn.run(app, host="0.0.0.0", port=port)

    # Create and start the server thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait a moment for the server to initialize
    time.sleep(2)
    print("Server thread started.")

    try:
        # Start ngrok
        public_url = ngrok.connect(port)
        print("---" * 20)
        print(f"✅ Public API URL: {public_url}")
        print("Available endpoint:")
        print(f"  POST {public_url}/transcribe")
        print("---" * 20)
        print("(Press Ctrl+C to shut down)")

        # Keep the main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down server and ngrok tunnel...")
        ngrok.disconnect(public_url)
        print("Shutdown complete.")
    except Exception as e:
        print(f"An error occurred: {e}")
        ngrok.disconnect(public_url)


