# 📝 Technical Report: Automated Grammar Scoring Engine
## ⚠️ Usage Note for Reviewers
This notebook was developed and executed in the **Kaggle** environment. 
It relies on the specific directory structure of the SHL Competition dataset:
- Input Path: `/kaggle/input/shl-intern-hiring-assessment-2025/`

**To run this locally or on Colab:**
1. You must download the dataset.
2. Update the `TEST_CSV_PATH` and `TEST_AUDIO_DIR` variables in the code to point to your local dataset location.


## 1. Overview
This notebook presents a multi-modal Deep Learning solution for automated grammar scoring (0.0 - 5.0). The approach fuses acoustic features (WavLM) with semantic and syntactic features (DeBERTa, Qwen 2.5) to robustly predict grammar quality from spoken audio.

## 2. Methodology & Architecture
Our solution treats the problem as a regression task using a **Hybrid Stacking Ensemble**.

### A. Preprocessing & Feature Extraction
1.  **Audio Processing:** 
    *   Raw audio is resampled to 16kHz using `librosa`.
    *   **ASR (Automatic Speech Recognition):** `OpenAI Whisper (Medium)` is used to transcribe audio to text with timestamps to calculate speech fluency metrics (speaking rate, silence ratio).
2.  **Feature Engineering (Multi-Modal):**
    *   **Acoustic Embeddings:** We use **WavLM (Microsoft)** to extract dense representations of the audio signal, capturing prosody and hesitation markers.
    *   **Semantic Embeddings:** We use **DeBERTa-v3** to capture the semantic meaning of the transcribed text.
    *   **LLM-as-a-Judge (Zero-Shot):** We employ **Qwen 2.5-1.5B-Instruct** to act as a "Grammar Teacher." We query the LLM to score the transcription on a scale of 1-5 and use this score as a high-level feature. We also calculate **Perplexity** to measure the model's "surprise" at the grammatical structure.

### B. Dimensionality Reduction
Due to the high dimensionality of embeddings (1500+) relative to the dataset size (409 samples), we apply **Principal Component Analysis (PCA)** to reduce the vector space to 64 components (32 Text + 32 Audio), retaining the most significant variance while preventing overfitting.

### C. Modeling Strategy (Stacking)
We use a **Stacking Regressor** architecture:
*   **Base Learners:** Ridge Regression (Linear), SVR (Non-linear), and LightGBM (Gradient Boosting).
*   **Meta Learner:** A RidgeCV meta-model learns the optimal weight combination of the base learners to minimize RMSE.

## 3. Evaluation Metrics
The model is evaluated using **5-Fold Cross-Validation** to ensure robustness. The primary metrics are:
*   **RMSE (Root Mean Squared Error):** Measures average deviation from ground truth.
*   **Pearson Correlation:** Measures how well the predicted ranking aligns with human judgement.
