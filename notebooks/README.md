# notebooks/

- `01_walkthrough.py` - step-by-step demo of every stage of the
  pipeline on a small synthetic dataset. Runs as a plain script
  (`python notebooks/01_walkthrough.py`) or opens as cells in
  VS Code / Jupyter via the `# %%` markers.

Future additions:

- `02_real_data.ipynb` - once a real-data loader lands, this should
  walk through pulling, cleaning, and scoring a Polymarket or Kalshi
  resolved-event sample.
- `03_deflated_sharpe.ipynb` - compute Deflated Sharpe Ratio for the
  leaderboard's top models (Bailey & Lopez de Prado 2014).
