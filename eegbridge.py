import asyncio
import json
import numpy as np
import websockets

# Configuration
# Default address for local server is typically ws://localhost:1616 
# If running on a dedicated Raspberry Pi, use its IP or ws://raspberrypi.local:1616
SERVER_URL = "ws://localhost:1616" 

N_CHANNELS = 16
SAMPLING_RATE = 250   # Octopus 16 standard rate (Hz)
WINDOW_DURATION = 2   # How many seconds of thought history to look at
MAX_SAMPLES = SAMPLING_RATE * WINDOW_DURATION

# Thread-safe rolling buffer initialization
# Keeps historical data of shape (channels, samples)
data_buffer = np.zeros((N_CHANNELS, MAX_SAMPLES))
samples_collected = 0

def process_live_window(window_matrix):
    """
    This function runs every time a full 2-second thought window is ready.
    Connect this directly to your machine learning model classifier.
    """
    # window_matrix shape is exactly (16, 500)
    print(f"--- Thought Window Ready! Shape: {window_matrix.shape} ---")
    
    # Example integration placeholder:
    # 1. Apply your bandpass filter (8-30 Hz)
    # 2. Add batch dimension: np.expand_dims(filtered_window, axis=0)
    # 3. prediction = bci_pipeline.predict(batch)
    # 4. Trigger arm movement if prediction == 1
    
    # For debugging, print mean signal amplitude across pins
    print(f"Current Signal Amplitudes Mean: {np.mean(np.abs(window_matrix)):.2f}")


async def stream_pieeg_data():
    global data_buffer, samples_collected
    
    print(f"Connecting to PiEEG Server at {SERVER_URL}...")
    
    try:
        async with websockets.connect(SERVER_URL) as ws:
            print("Connected successfully! Receiving live Octopus 16 stream...")
            
            async for msg in ws:
                # Every incoming frame from PiEEG server is plain JSON
                frame = json.loads(msg)
                
                # Check for standard data frames containing channel packets
                if "channels" in frame:
                    # Expecting an array of 16 values matching pogo pin readings
                    channel_data = np.array(frame["channels"])
                    
                    if len(channel_data) != N_CHANNELS:
                        continue # Skip malformed packets
                        
                    # Shift old data left, append new sample to the right end
                    data_buffer = np.roll(data_buffer, -1, axis=1)
                    data_buffer[:, -1] = channel_data
                    
                    samples_collected += 1
                    
                    # Every time we accumulate a clean window step, process it
                    # We evaluate the window frequently (e.g., every 25 samples / 0.1 seconds)
                    # to keep the system responsive rather than waiting a full 2 seconds.
                    if samples_collected >= MAX_SAMPLES and (samples_collected % 25 == 0):
                        # Create a static snapshot copy for processing to avoid threading race conditions
                        window_snapshot = np.copy(data_buffer)
                        process_live_window(window_snapshot)
                        
    except Exception as e:
        print(f"Connection error or server closed: {e}")

if __name__ == "__main__":
    # Start the asynchronous event loop to process the real-time websocket packets
    asyncio.run(stream_pieeg_data())
