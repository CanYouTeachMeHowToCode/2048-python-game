# 2048-python-game
2048 game implemented by python3 and tkinter

AI part: Expectiminimax with alpha-beta pruning (Normal level), Reinforcement Learning (Hard level)

## Requirements

- Python 3.8+
- tkinter (bundled with the standard Windows/macOS Python installers; on Linux you may need to install it separately, e.g. `sudo apt install python3-tk`)

## Installation

The game lives in the `2048/` folder, which is a self-contained Python package (`game2048`, `src/` layout). Install it in editable mode:

```bash
cd 2048
pip install -e .
```

This installs the `game2048` package plus a `play2048` console command.

## Running the game

After installing, either of these will launch the game window:

```bash
play2048
```

```bash
python -m game2048.main
```

### Controls

- **Arrow keys** — move tiles (Player mode)
- **p** — pause / unpause
- **Esc** — while paused, or after winning/losing, return to the home page
- **r** — restart (once the game is over)
- **c** — keep playing after reaching 2048

## Running the tests

From the `2048/` folder (after installing the package as above):

```bash
pytest
```

## Project structure

```
2048/
  pyproject.toml        # package metadata, editable-install config, pytest config
  src/game2048/
    board.py            # board state and move/merge logic
    AI.py               # Expectiminimax / alpha-beta AI opponent
    RL.py                # reinforcement-learning experiments (not wired into the UI yet)
    UI.py                # tkinter UI and game loop
    main.py              # entry point (play2048 / python -m game2048.main)
  test/
    test_board.py
    test_ai.py
```
