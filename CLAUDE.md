# Voili Server — for AI instances

Voili Server runs Inworld Voice Isolator v1 (Apache-2.0) live in the macOS system audio path: input BlackHole 2ch → model on MPS → output device, with a time-aligned original for the mix slider and a web control panel on 127.0.0.1:8770.

- Start: `README.md` (user view), then `voili_server.py` (module head: Ends/Ways/Means/Provides/Expects/Ensures/Fails).
- Setup: `./install.sh` (venv in `.venv`, Swift helpers into `bin/`), run `./start.sh`, quit `./stop.sh`.
- Two stages: offline (two-sided model, sliding windows 0.5 · 1 · 0.25 s) and real time (causal model, 100 ms pieces). Bandwidth limit in `bandgrenze.py`; "full" must stay bit-identical to the unmodified model.
- Never run GPU benchmarks while someone is listening: the live server and the benchmark share the GPU, and the listener hears dropouts.
- Quotes of Hans-Dieter Zucht in comments stay in German: they record why a decision was taken.
- Panel texts live in the `TX` table (de/en) of the page script; server errors are codes (`E_…`) translated there.
- In Hans-Dieter Zucht's vault the project folder is the source of truth; this repository is a curated snapshot (`tools/snapshot.sh` in the vault).
