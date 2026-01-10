# Determinacy-in-Lipparini-s-Dadaist-Sums
# Infinite-Combinatorial-Game-Sums-ISEF

## Project Overview

This repository investigates the determinacy of Lipparini's infinite combinatorial Nim sums through AI-driven probabilistic simulations and mathematical analysis.

**Research question:**  
Can we prove determinacy in Lipparini's infinite combinatorial Nim sums using AI probabilistic simulations?

By simulating large datasets of game positions and applying machine learning models, the project seeks patterns and empirical evidence supporting (or challenging) determinacy in these infinite games.

---

## Repository Structure

| File                                  | Description                                                         |
|----------------------------------------|---------------------------------------------------------------------|
| `app.py`                              | Main application/UI or experiment runner.                           |
| `infinite_nim_lipparini_50000.csv`     | Dataset of 50,000 simulated infinite Nim sum positions.             |
| `lippariniNimSumsSim.py`               | Simulates infinite Nim games and generates datasets.                |
| `main.py`                             | Central runner for simulations, experiments, and ML predictions.    |
| `rf_model_lipparini.pkl`               | Trained Random Forest model for win/loss prediction.                |
| `scaler_lipparini.pkl`                 | Saved scaler for feature normalization used in ML models.           |

---

## Methods

- **Simulation:** Generate random samples of infinite Nim sum positions (truncated for computation), following Lipparini’s formal rules.
- **Machine Learning:** Train a Random Forest classifier on simulated datasets to predict game outcomes and evaluate pattern consistency.
- **Mathematical Modeling:** Rigorously define game structure and win conditions based on recent research.
- **Proof Exploration:** Explore mathematical and computational approaches to determinacy, connecting AI results to theory.

---

## Getting Started

1. **Clone the repository**
git clone https://github.com/SumukhKoundinya/Infinite-Combinatorial-Game-Sums-ISEF

2. **Install dependencies**

3. **Run simulations**

4. **Run experiments / ML inference**


Pre-trained model and scaler are provided for reproducibility.

---

## Documentation

- Formal definitions, propositions, and proof outlines are found in `/docs` (add writeups as needed).
- Results and visualizations may be added in notebooks or documented outputs.

---

## Citation

If you use this code, data, or research in a publication or competition, please cite this repository and reference the research question.

---

## Author

Sumukh Koundinya  

---

## License

MIT License
