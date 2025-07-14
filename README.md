```markdown
# ZTA Intrusion Detection Notebook

This Jupyter notebook implements a two-part deep-learning framework for intrusion detection in vehicular (V2X) networks under a Zero-Trust Architecture (ZTA). It addresses:

1. Extreme class imbalance via post-hoc threshold calibration of a deep-learning ensemble.  
2. Temporal concept drift via an unsupervised KL-divergence drift detector in a convolutional autoencoder’s latent space.

---

## ⚙️ Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-org/ZTA-IDS.git
   cd ZTA-IDS
   ```

2. **Create and activate a Python 3 virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

1. **Launch Jupyter Notebook**  
   ```bash
   jupyter notebook
   ```
2. **Open** `ZTA.ipynb` and run all cells from top to bottom.  
3. All intermediate and final metrics, tables, and plots are saved under the `results/` folder.

---

## 📖 Notebook Walkthrough

- **Performance Summary (Cell #10)**  
  Trains or loads four model variants (`Standard`, `Attention`, `Streaming`, `Augmented`) and computes Accuracy, Precision, Recall, F1. Outputs a table and saves it as `.txt` and `.pdf`.

- **Threshold Optimization (Cell #11)**  
  Sweeps decision thresholds ∈ [0.1, 0.9) on ensemble probabilities to maximize F1. Reports optimal threshold (≈ 0.88) and F1 (≈ 0.464).

- **F1 Score Curve by Threshold (Cell #12)**  
  Plots F1 vs. threshold to visualize the trade-off.

- **Final Evaluation (Cell #13)**  
  Applies the optimal threshold, prints and saves the classification report (Precision, Recall, F1) for “Normal” vs. “Attack”.

- **Confusion Matrix (Cell #14)**  
  Displays the confusion matrix for the optimized ensemble.

- **ROC & Precision-Recall Curves (Cell #15)**  
  Computes and plots ROC and PR curves, reports AUCs.

- **Stratified Cross-Validation (Cell #16)**  
  Defines a `cross_validate_model` using `StratifiedKFold` to report average F1 across 5 folds.

- **Contrastive Learning Module (Cell #17)**  
  *(Optional)* Sketch of a Siamese network for feature representation learning via contrastive loss.

- **Drift Detection Visualizations**  
  - **KL Divergence Plot**: time-series of KL(p‖q) between consecutive latent windows  
  - **t-SNE Embedding**: 2D scatter of concatenated latent vectors across windows

---

## 📊 Outputs

After running the notebook, check the `results/` directory for:

- `metrics_summary.txt` / `.pdf` – Model performance table  
- `final_evaluation.txt` – Classification report with optimized threshold  
- `kl_divergence_plot.png` – Drift detector signal over time  
- `tsne_plot.png` – Latent-space clustering across time windows  

---

## 🛠️ Customization

- **Data inputs**: Modify data-loading cells to point at your own V2X or time-series intrusion dataset.  
- **Model definitions**: Swap in alternative architectures (e.g. pure RNN, Transformer) by editing the `models_list` construction.  
- **Drift detector**: Adjust window sizes or replace KL divergence with other distributional metrics.

---

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
```
