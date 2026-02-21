# CHANGELOGS

## 0.0.01 - 2026-02-21
- Fixed P1 mode-safety issue: selecting competition mode now forces `NetworkManager` to run with `simulation_mode=False`, even if `Settings.SIMULATION_MODE=True`.
- Refactored network mode checks to use instance-level `self.simulation_mode` instead of global settings so runtime mode is explicit and controllable.
