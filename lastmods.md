What's done

octopus_bridge.py — connects to the Octopus 16 over BLE (auto-scan with retry, since it advertises intermittently) and serves pieeg-server's exact JSON protocol on ws://localhost:1616, including a mock: false status frame
eegbridge.py — now records server timestamps and the effective sample rate in metadata
inspect_recording.py — plots any recording as a 16-channel waterfall (use --detrend to remove DC offsets)
Live test: 4s recording at 249.9 Hz from the real device — 20261005_142037_C3_rest.npz + .png plot (I verified all 16 traces rendered via pixel check — couldn't preview images in this environment, so give the PNG a look yourself)
Notes for the classifier phase

Channels have large per-channel DC offsets (electrode offsets, ±200-1900 µV) on top of ±12-23 µV of variation. The official SDK high-passes at 0.5 Hz — you'll want detrend/high-pass as the first preprocessing step.
Octopus channel order → 10-20 names: index 8 = C3, index 10 = C4 (full list in octopus_bridge.py:45-48).
The bridge is still running in the background on port 1616 — record whenever you're ready with python eegbridge.py --label rest --region C3 --duration 10. Next step when you want it: the classifier in process_live_window.