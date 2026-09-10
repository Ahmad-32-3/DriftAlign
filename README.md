# DriftAlign

I train a classifier where the data looks one way, then I test where the data looks shifted. Before scoring, I try a step that slides the training cloud toward the test cloud (Sinkhorn / optimal transport).

Mixing both worlds and shuffling is only a debug check. The number I trust is accuracy on the shifted data after that move, printed next to the same classifier with no transport step.

## Run

```bash
python -m pytest tests/ -q
python scripts/run.py
npm --prefix web install
npm --prefix web run dev
```

`scripts/run.py` prints target-domain success with transport next to no-transport. Python, NumPy/SciPy or POT (Python Optimal Transport), and a small sklearn or torch head.

## Layout

- `src/` source/target split, alignment, eval
- `scripts/run.py`
- `tests/` leak injection
- `web/` case-study page
