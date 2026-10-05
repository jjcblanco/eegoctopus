"""
BLE -> WebSocket bridge for the PiEEG Octopus 16.

Connects to the Octopus 16 over Bluetooth Low Energy (BLE) and re-serves the
16-channel stream over a local WebSocket using the same JSON protocol as
pieeg-server, so eegbridge.py works unchanged:

    {"status": "connected", "sample_rate": 250, "channels": 16, ...}
    {"t": <epoch seconds>, "n": <sample index>, "channels": [16 floats]}

Packet format (pieeg-club/Octopus_16 firmware, 51 bytes):
    0xA0 | counter | 24 B ADC1 (8ch x 3B, big-endian int24) |
    24 B ADC2 (8ch x 3B) | 0xC0

Raw ADC counts are converted to microvolts using the firmware defaults:
internal VREF = 1.2 V and per-channel gain = 4 (GAIN register 0x2222), so

    uV = raw * 1.2 / (4 * 2^23) * 1e6  ~=  raw * 0.03576

Override with --uv-scale, or pass --raw to keep ADC counts.

Usage:
    python octopus_bridge.py            # scan BLE, serve ws://localhost:1616
    python octopus_bridge.py --scan     # just list visible BLE devices and exit
"""

import argparse
import asyncio
import json
import time

import numpy as np
import websockets

SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
DATA_CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
DEFAULT_DEVICE_NAME = "bioron_16"

SAMPLE_RATE = 250
N_CHANNELS = 16
PACKET_SIZE = 51
HEADER_BYTE = 0xA0
FOOTER_BYTE = 0xC0

# ADS131M08 firmware defaults: internal VREF 1.2 V, gain 4 (reg 0x2222)
DEFAULT_UV_SCALE = 1.2 / (4 * 2**23) * 1e6  # uV per ADC count

# 10-20 electrode labels in Octopus channel order
CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3",
    "C3", "Cz", "C4", "T4", "P3", "Pz", "P4", "Oz",
]


def int24_be(b0, b1, b2):
    """Big-endian 24-bit two's complement -> signed int."""
    val = (b0 << 16) | (b1 << 8) | b2
    return val - 0x1000000 if val & 0x800000 else val


def parse_packet(data, uv_scale):
    """Parse one 51-byte BLE notification into 16 channel values."""
    if len(data) != PACKET_SIZE or data[0] != HEADER_BYTE or data[-1] != FOOTER_BYTE:
        return None
    values = np.empty(N_CHANNELS, dtype=np.float64)
    for i in range(8):
        b = 2 + i * 3
        values[i] = int24_be(data[b], data[b + 1], data[b + 2])
    for i in range(8):
        b = 26 + i * 3
        values[8 + i] = int24_be(data[b], data[b + 1], data[b + 2])
    return {"counter": data[1], "channels": values * uv_scale}


async def find_octopus(name, timeout):
    """Locate the Octopus by its service UUID, falling back to BLE name."""
    from bleak import BleakScanner

    print(f"[BLE] Scanning up to {timeout}s for Octopus 16 ...")
    dev = await BleakScanner.find_device_by_filter(
        lambda d, adv: SERVICE_UUID.lower() in [str(u).lower() for u in adv.service_uuids],
        timeout=timeout,
    )
    if dev:
        print(f"[BLE] Found by service UUID: {dev.name!r}  {dev.address}")
        return dev
    if name:
        dev = await BleakScanner.find_device_by_name(name, timeout=timeout)
        if dev:
            print(f"[BLE] Found by name: {dev.name!r}  {dev.address}")
            return dev
    print("[BLE] Octopus 16 not found. Visible devices:")
    devices = await BleakScanner.discover(timeout=5, return_adv=True)
    for addr, (d, adv) in devices.items():
        uuids = ", ".join(str(u) for u in adv.service_uuids) or "-"
        print(f"  {addr}  {d.name!r:24s}  {adv.rssi} dBm  {uuids}")
    return None


async def scan_only(name, timeout):
    await find_octopus(name, timeout)


class Bridge:
    """Receives BLE notifications and fans them out to WebSocket clients."""

    def __init__(self, uv_scale):
        self.uv_scale = uv_scale
        self.clients = set()
        self.sample_count = 0
        self.last_counter = None
        self.dropped = 0
        self._loop = None

    def _send_to_all(self, frame):
        for ws in list(self.clients):
            self._loop.call_soon_threadsafe(self._schedule_send, ws, frame)

    def _schedule_send(self, ws, frame):
        asyncio.create_task(self._send(ws, frame))

    async def _send(self, ws, frame):
        try:
            await ws.send(frame)
        except websockets.ConnectionClosed:
            self.clients.discard(ws)

    def on_notify(self, _handle, data):
        parsed = parse_packet(bytes(data), self.uv_scale)
        if parsed is None:
            return
        counter = parsed["counter"]
        if self.last_counter is not None:
            gap = (counter - self.last_counter - 1) & 0xFF
            if gap:
                self.dropped += gap
                print(f"[BLE] {gap} packet(s) dropped (total {self.dropped})")
        self.last_counter = counter
        self.sample_count += 1
        frame = json.dumps(
            {
                "t": time.time(),
                "n": self.sample_count,
                "channels": [round(float(v), 3) for v in parsed["channels"]],
            }
        )
        self._send_to_all(frame)

    async def handler(self, ws):
        self.clients.add(ws)
        await ws.send(
            json.dumps(
                {
                    "status": "connected",
                    "sample_rate": SAMPLE_RATE,
                    "channels": N_CHANNELS,
                    "channel_names": CHANNEL_NAMES,
                    "filter": False,
                    "mock": False,
                    "source": "Octopus 16 BLE",
                }
            )
        )
        try:
            await ws.wait_closed()
        finally:
            self.clients.discard(ws)


async def run(args):
    from bleak import BleakClient

    bridge = Bridge(args.uv_scale)
    bridge._loop = asyncio.get_running_loop()

    server = await websockets.serve(bridge.handler, args.host, args.port)
    print(f"[WS] Serving on ws://{args.host}:{args.port}")

    # The Octopus advertises intermittently and drops idle links (battery
    # saver), so scan/connect in a retry loop and reconnect after any drop.
    attempt = 0
    while True:
        attempt += 1
        address = args.address
        if address is None:
            print(f"[BLE] Scan attempt {attempt} ...")
            device = await find_octopus(args.name, args.timeout)
            if device is None:
                print("[BLE] Not found; retrying in 5s ...")
                await asyncio.sleep(5)
                continue
            address = device.address

        disconnected = asyncio.Event()

        def on_disconnect(_client):
            disconnected.set()

        try:
            async with BleakClient(
                address, timeout=20, disconnected_callback=on_disconnect
            ) as client:
                try:
                    await client.request_mtu(100)
                except Exception:
                    pass
                await client.start_notify(DATA_CHAR_UUID, bridge.on_notify)
                print(
                    f"[BLE] Connected {address} (MTU={client.mtu_size}). "
                    f"Streaming 16ch @ {SAMPLE_RATE} Hz."
                )
                while not disconnected.is_set():
                    try:
                        await asyncio.wait_for(disconnected.wait(), timeout=10)
                    except asyncio.TimeoutError:
                        print(f"[BLE] {bridge.sample_count} samples received")
        except asyncio.CancelledError:
            print("\nStopping ...")
            break
        except Exception as error:
            print(f"[BLE] Connection error: {error}")
        print("[BLE] Link lost; re-scanning ...")
        await asyncio.sleep(3)

    server.close()
    await server.wait_closed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bridge the Octopus 16 BLE stream to a local WebSocket"
    )
    parser.add_argument("--port", type=int, default=1616, help="WebSocket port")
    parser.add_argument("--host", default="localhost", help="Bind address")
    parser.add_argument("--name", default=DEFAULT_DEVICE_NAME,
                        help="BLE advertised name fallback")
    parser.add_argument("--address", default=None,
                        help="BLE MAC address to connect directly (skip scan)")
    parser.add_argument("--timeout", type=float, default=15, help="BLE scan seconds")
    parser.add_argument("--raw", action="store_true",
                        help="Keep raw ADC counts (no uV conversion)")
    parser.add_argument("--uv-scale", type=float, default=DEFAULT_UV_SCALE,
                        help="uV per ADC count (default 1.2V ref, gain 4)")
    parser.add_argument("--scan", action="store_true",
                        help="Only list BLE devices, then exit")
    args = parser.parse_args()

    if args.raw:
        args.uv_scale = 1.0

    if args.scan:
        asyncio.run(scan_only(args.name, args.timeout))
    else:
        try:
            asyncio.run(run(args))
        except KeyboardInterrupt:
            pass
