"""
Xbox Controller Input — maps controller axes/buttons to drone velocity commands.

Mapping (SDL2 / pygame GameController):
    Left stick X/Y  → horizontal velocity (vx, vy)
    Right trigger    → ascend (vz+)
    Left trigger     → descend (vz-)
    B button         → reset episode
    Start button     → quit

Usage:
    python -m eval.controller --test   # print raw axes to verify mapping
"""

import numpy as np

try:
    import pygame
except ImportError:
    pygame = None


class XboxController:
    """Read Xbox controller and return (vx, vy, vz) velocity commands in [-1, 1]."""

    def __init__(self, deadzone=0.15, controller_index=0):
        if pygame is None:
            raise ImportError("pygame is required: pip install pygame")

        pygame.init()
        pygame.joystick.init()

        count = pygame.joystick.get_count()
        if count == 0:
            raise RuntimeError("No controller found. Connect an Xbox controller and retry.")

        self.joystick = pygame.joystick.Joystick(controller_index)
        self.joystick.init()
        self.deadzone = deadzone
        self.name = self.joystick.get_name()
        print(f"Controller connected: {self.name}")

    def _apply_deadzone(self, value):
        if abs(value) < self.deadzone:
            return 0.0
        # Rescale so deadzone→0 and 1→1
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)

    def get_velocity(self) -> np.ndarray:
        """Returns (vx, vy, vz) in [-1, 1]."""
        pygame.event.pump()

        # Left stick: XY movement
        lx = self._apply_deadzone(self.joystick.get_axis(0))   # left/right
        ly = self._apply_deadzone(self.joystick.get_axis(1))    # forward/back (inverted)

        # Triggers: Z movement
        # SDL2 maps triggers as axes 4 (LT) and 5 (RT) on most controllers
        # Range is -1 (released) to +1 (fully pressed)
        num_axes = self.joystick.get_numaxes()
        if num_axes >= 6:
            lt_raw = self.joystick.get_axis(4)
            rt_raw = self.joystick.get_axis(5)
            lt = (lt_raw + 1.0) / 2.0  # normalize to 0-1
            rt = (rt_raw + 1.0) / 2.0
        elif num_axes >= 3:
            # Fallback: some drivers use axis 2 for triggers
            lt = max(0.0, self.joystick.get_axis(2))
            rt = max(0.0, -self.joystick.get_axis(2))
        else:
            lt = 0.0
            rt = 0.0

        vz = rt - lt  # RT = up, LT = down

        return np.array([lx, -ly, vz], dtype=np.float32)

    def get_buttons(self) -> dict:
        """Returns button states."""
        pygame.event.pump()
        return {
            "reset": self.joystick.get_button(1),    # B
            "quit": self.joystick.get_button(7),      # Start / Menu
            "action": self.joystick.get_button(0),    # A
        }

    def close(self):
        pygame.joystick.quit()
        pygame.quit()


def test_controller():
    """Print raw controller values for calibration."""
    ctrl = XboxController()
    print(f"Controller: {ctrl.name}")
    print(f"Axes: {ctrl.joystick.get_numaxes()}, Buttons: {ctrl.joystick.get_numbuttons()}")
    print("Move sticks and press triggers. Press Start to quit.\n")

    try:
        while True:
            vel = ctrl.get_velocity()
            buttons = ctrl.get_buttons()

            # Print raw axes for debugging
            raw = [ctrl.joystick.get_axis(i) for i in range(ctrl.joystick.get_numaxes())]
            raw_str = " ".join(f"A{i}:{v:+.2f}" for i, v in enumerate(raw))

            print(f"\rVel: vx={vel[0]:+.2f} vy={vel[1]:+.2f} vz={vel[2]:+.2f} | "
                  f"Btn: B={buttons['reset']} Start={buttons['quit']} | "
                  f"Raw: {raw_str}  ", end="", flush=True)

            if buttons["quit"]:
                print("\nQuit pressed.")
                break

            pygame.time.wait(50)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        ctrl.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test controller mapping")
    args = parser.parse_args()

    if args.test:
        test_controller()
    else:
        print("Use --test to print raw controller values")
