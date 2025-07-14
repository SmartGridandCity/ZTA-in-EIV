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

### Pipeline Main Steps

1. **Data Ingestion & Preprocessing**  
   - Load raw CAN/V2X CSV, handle missing values, resample/interpolate at 100 µs, normalize signals.

2. **Time-Series Windowing & Labeling**  
   - Slide a 50-step window with 5-step stride, compute mean label per window, assign “Normal” (0) or “Attack” (1).

3. **Class Balancing**  
   - Upsample minority (attack) windows to match the count of normal windows.

4. **Model Training**  
   - Build and train four model variants:  
     • Standard CNN–LSTM  
     • Streaming (no attention)  
     • Attention-augmented  
     • Data-augmented (time-jitter & signal-masking)

5. **Ensemble & Performance Summary**  
   - Average model probabilities, compute Accuracy, Precision, Recall, F1.  
   - Save `metrics_summary.txt` / `metrics_summary.pdf`.

6. **Threshold Optimization**  
   - Sweep decision thresholds in [0.1, 0.9) to maximize F1.  
   - Generate `f1_vs_threshold.png`, record optimal threshold and F1 in `final_evaluation.txt`.

7. **Final Evaluation & Plots**  
   - Apply optimal threshold, produce:  
     • Confusion matrix → `confusion_matrix.png`  
     • ROC curve → `roc_curve.png`  
     • Precision–Recall curve → `precision_recall_curve.png`

8. **Unsupervised Drift Detection**  
   - Train a Conv-Autoencoder, extract latent vectors for consecutive time chunks.  
   - Compute KL divergence between chunks and save `kl_divergence_scores.txt` and `kl_divergence_plot.png`.

9. **Latent-Space Visualization**  
   - Concatenate all latent vectors, run t-SNE, and save `tsne_plot.png` for drift inspection.


---

### Generated Outputs

- confusion_matrix.png  
- f1_vs_threshold.png  
- final_evaluation.txt  
- kl_divergence_plot.png  
- kl_divergence_scores.txt  
- metrics_summary.pdf  
- metrics_summary.txt  
- precision_recall_curve.png  
- roc_curve.png  
- tsne_plot.png  
---

## 🛠️ Customization

- **Data inputs**: Modify data-loading cells to point at your own V2X or time-series intrusion dataset.  
- **Model definitions**: Swap in alternative architectures (e.g. pure RNN, Transformer) by editing the `models_list` construction.  
- **Drift detector**: Adjust window sizes or replace KL divergence with other distributional metrics.

---

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
```
