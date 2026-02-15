# Δ-Nim under Lipparini’s Dadaist Sums  
# Infinite-Combinatorial-Game-Sums-ISEF

## Project Overview

This repository studies **Δ-Nim**, a Nim-like impartial game on countably many heaps, defined using Lipparini’s infinite sum (Dadaist) framework for combinatorial games. [file:100]

We prove a **complete determinacy classification** of Δ-Nim positions based on the support size |supp(h)|:

- Sparse case: |supp(h)| ≤ 60  
  The position is winning for the next player (N-position) if and only if the nim-sum ⊕h is nonzero. [file:145]

- Dense finite case: 60 < |supp(h)| < ∞  
  Player II has a pairing strategy that maintains ⊕h = 0, so every such position is a P-position.

- Infinite support case: |supp(h)| = ∞  
  Extending the dense pairing strategy using infinitely many reserve heaps, Player II again has a winning strategy; all infinite-support positions are P-positions.

**Research question (final form):**  
Is every Δ-Nim position h in N^N completely determined by a support-size density classification (sparse / dense finite / infinite), with nim-sum deciding the sparse case and pairing strategies deciding the dense and infinite cases?

We combine:

- Mathematical proofs of the three cases above, and  
- Machine learning (Random Forest) trained on 50,000 labeled positions,

to show strong agreement between theory and empirical predictions.

---

## Repository Structure

| File                               | Description                                                                                       |
|------------------------------------|---------------------------------------------------------------------------------------------------|
| `infinite_nim_lipparini_50000.csv` | Dataset of 50,000 Δ-Nim positions with labels (N/P) from the theoretical classification.         |
| `lippariniNimSumsSim.py`           | Generates random Δ-Nim positions (finite truncations) consistent with the Dadaist / Δ-sum rules. |
| `main.py`                          | Entry point for generating data, training models, and evaluating performance.                    |
| `rf_model_lipparini.pkl`           | Trained Random Forest model for predicting N/P outcomes for Δ-Nim positions.                     |
| `scaler_lipparini.pkl`             | Saved scaler for feature normalization used during model training and inference.                 |

Note: Some filenames still say “Lipparini Nim” for historical reasons; the game studied here is Δ-Nim under Lipparini-style infinite sum rules.

---

## Methods

- **Game Definition (Δ-Nim):**  
  Positions are sequences h in N^N with countable support. Moves are standard Nim moves on a single heap, embedded in a Lipparini-style infinite sum rule (Dadaist sum) that ensures no loops and well-founded play. [file:100]

- **Theoretical Classification:**
  - Sparse case: |supp(h)| ≤ 60  
    Bouton’s theorem applies; the nim-sum ⊕h fully decides N/P. [file:145]  
  - Dense finite case: 60 < |supp(h)| < ∞  
    A pairing strategy for Player II maintains ⊕h = 0 and yields P-positions.  
  - Infinite support case: |supp(h)| = ∞  
    The dense pairing strategy extends using infinitely many reserves; Player II still wins, so these are P-positions.

- **Simulation:**  
  Generate random Δ-Nim positions with various support sizes, apply the theoretical classification to label positions, and record features such as heap sizes, support size, and nim-sum.

- **Machine Learning:**  
  Train a Random Forest classifier on the labeled dataset (~50k positions) to predict N/P outcomes. The best model achieves:
  - Test accuracy ≈ 0.9764  
  - MCC ≈ 0.9253  

  This confirms that the simple density-based rules align extremely well with the position space explored.

- **Proof Exploration:**  
  The code supports experimentation and sanity checks around the conjectured/now-proved thresholds (for example, support size 60) and helps explore alternative density cutoffs or variants.

---

## Getting Started

1. Clone the repository

   ```bash
   git clone https://github.com/SumukhKoundinya/Determinacy-in-Lipparini-s-Dadaist-Sums.git
   cd Determinacy-in-Lipparini-s-Dadaist-Sums
