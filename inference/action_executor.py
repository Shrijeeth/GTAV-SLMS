"""Execute predicted actions via keyboard/mouse using pynput."""
from __future__ import annotations

import time
from typing import Dict, List


class ActionExecutor:
    """Convert model action outputs to keyboard/mouse events."""

    def __init__(self, key_hold_ms: int = 50, mouse_smoothing: bool = True):
        from pynput.keyboard import Controller as KBController
        from pynput.mouse import Controller as MouseController
        self.kb = KBController()
        self.mouse = MouseController()
        self.key_hold_ms = key_hold_ms
        self.mouse_smoothing = mouse_smoothing
        self._pressed_keys: set = set()

    def execute(self, binary_actions: Dict[str, bool],
                continuous_actions: Dict[str, float]) -> None:
        """Press/release keys and move mouse according to predictions."""
        from pynput.keyboard import Key, KeyCode

        key_map = {
            "SPACE": Key.space,
            "SHIFT": Key.shift,
            "CTRL": Key.ctrl_l,
        }

        # Binary keys
        for key_name, active in binary_actions.items():
            key = key_map.get(key_name) or KeyCode.from_char(key_name.lower())
            if active and key not in self._pressed_keys:
                self.kb.press(key)
                self._pressed_keys.add(key)
            elif not active and key in self._pressed_keys:
                self.kb.release(key)
                self._pressed_keys.discard(key)

        # Continuous mouse movement
        dx = continuous_actions.get("dx", 0.0)
        dy = continuous_actions.get("dy", 0.0)
        if self.mouse_smoothing:
            dx = int(dx * 0.7)
            dy = int(dy * 0.7)
        self.mouse.move(int(dx), int(dy))

    def release_all(self) -> None:
        """Release all held keys."""
        for key in list(self._pressed_keys):
            self.kb.release(key)
        self._pressed_keys.clear()
