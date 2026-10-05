import asyncio
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

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

def parse_channel_frame(message):
    """
    Extract one EEG sample and its server timestamp from a PiEEG JSON message.

    The expected data message is {"t": ..., "n": ..., "channels": [value, ...]}.
    Other message types are ignored so status/heartbeat messages do not enter
    the training data.
    """
    if "channels" not in message:
        return None

    values = np.asarray(message["channels"], dtype=np.float64)
    if values.ndim != 1 or values.size != N_CHANNELS:
        return None
    if not np.all(np.isfinite(values)):
        return None
    return values, message.get("t")


def process_live_window(window_matrix):
    """Hook for a classifier once a labeled recording pipeline is available."""
    print(
        f"Window ready: shape={window_matrix.shape}, "
        f"mean_amplitude={np.mean(np.abs(window_matrix)):.2f}"
    )


def save_recording(samples, timestamps, output_path, label, brain_region, sample_rate):
    """Save a recording and enough metadata to reproduce its preprocessing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    effective_rate = sample_rate
    if timestamps is not None and len(timestamps) > 1:
        span = timestamps[-1] - timestamps[0]
        if span > 0:
            effective_rate = (len(timestamps) - 1) / span
    metadata = {
        "label": label,
        "brain_region": brain_region,
        "channels": N_CHANNELS,
        "sample_rate_hz": sample_rate,
        "effective_sample_rate_hz": round(float(effective_rate), 2),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "PiEEG WebSocket",
    }
    np.savez_compressed(
        output_path,
        eeg=samples,
        timestamps=np.asarray(timestamps) if timestamps is not None else np.asarray([]),
        label=np.asarray(label),
        metadata=np.asarray(json.dumps(metadata)),
    )
    print(f"Saved {samples.shape[0]} samples to {output_path} "
          f"(effective rate {effective_rate:.1f} Hz)")


async def stream_pieeg_data(
    server_url, output_path, label, brain_region, sample_rate, duration
):
    window_samples = sample_rate * WINDOW_DURATION
    window_step = max(1, sample_rate // 10)
    rolling_buffer = np.zeros((N_CHANNELS, window_samples), dtype=np.float64)
    recorded_samples = []
    sample_times = []
    window_count = 0

    print(f"Connecting to PiEEG server at {server_url}...")
    try:
        async with websockets.connect(server_url) as ws:
            print(f"Connected. Recording label '{label}'. Press Ctrl+C to stop.")
            start_time = asyncio.get_running_loop().time()

            async for raw_message in ws:
                message = json.loads(raw_message)
                parsed = parse_channel_frame(message)
                if parsed is None:
                    continue
                channel_data, sample_time = parsed

                recorded_samples.append(channel_data)
                if sample_time is not None:
                    sample_times.append(float(sample_time))
                rolling_buffer = np.roll(rolling_buffer, -1, axis=1)
                rolling_buffer[:, -1] = channel_data
                window_count += 1

                if window_count >= window_samples and window_count % window_step == 0:
                    process_live_window(rolling_buffer.copy())
    
                if duration and asyncio.get_running_loop().time() - start_time >= duration:
                    break
    except KeyboardInterrupt:
        print("\nRecording interrupted.")
    except (OSError, websockets.WebSocketException, json.JSONDecodeError) as error:
        print(f"Connection or data error: {error}")
    finally:
        if recorded_samples:
            save_recording(
                np.asarray(recorded_samples),
                np.asarray(sample_times) if sample_times else None,
                output_path,
                label,
                brain_region,
                sample_rate,
            )
        else:
            print("No valid EEG samples were received; nothing was saved.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record labeled PiEEG/Octopus 16 data")
    parser.add_argument("--url", default=SERVER_URL, help="PiEEG WebSocket URL")
    parser.add_argument("--label", required=True, help="Movement or rest label")
    parser.add_argument("--region", required=True, choices=("C3", "C4"))
    parser.add_argument("--output", type=Path, default=Path("recordings"))
    parser.add_argument("--duration", type=float, default=0, help="Seconds; 0 means until Ctrl+C")
    parser.add_argument("--rate", type=int, default=SAMPLING_RATE)
    args = parser.parse_args()

    filename = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{args.region}_{args.label}.npz"
    )
    output_path = args.output / filename
    asyncio.run(
        stream_pieeg_data(
            args.url,
            output_path,
            args.label,
            args.region,
            args.rate,
            args.duration,
        )
    )
