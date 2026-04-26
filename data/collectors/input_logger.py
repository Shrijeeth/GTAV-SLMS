"""Log keyboard and mouse inputs while playing GTA V."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional


def start_logging(
    output_path: str | Path,
    duration_s: Optional[float] = None,
) -> None:
    """Record keyboard presses and mouse movements to a JSONL file."""
    from pynput import keyboard, mouse

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    t0 = time.time()

    def _ts() -> float:
        return time.time() - t0

    # ---- keyboard ----
    def on_press(key):
        name = key.char if hasattr(key, "char") and key.char else str(key)
        records.append({"t": _ts(), "type": "key_down", "key": name})

    def on_release(key):
        name = key.char if hasattr(key, "char") and key.char else str(key)
        records.append({"t": _ts(), "type": "key_up", "key": name})
        if key == keyboard.Key.f10:  # emergency stop
            return False

    # ---- mouse ----
    def on_move(x, y):
        records.append({"t": _ts(), "type": "mouse_move", "x": x, "y": y})

    def on_click(x, y, button, pressed):
        records.append({
            "t": _ts(),
            "type": "mouse_down" if pressed else "mouse_up",
            "button": str(button), "x": x, "y": y,
        })

    kl = keyboard.Listener(on_press=on_press, on_release=on_release)
    ml = mouse.Listener(on_move=on_move, on_click=on_click)
    kl.start()
    ml.start()

    try:
        if duration_s:
            time.sleep(duration_s)
        else:
            kl.join()
    except KeyboardInterrupt:
        pass
    finally:
        kl.stop()
        ml.stop()

    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "
")
    print(f"Logged {len(records)} events -> {out}")
