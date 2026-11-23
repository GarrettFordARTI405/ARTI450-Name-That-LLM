# Name That LLM — ARTI 450 Project 1

**Ferris State University · ARTI 450 · Professor Afolabi**  
**Team:** Garrett Ford, Ferris Farmer, and Caden Raleigh

## Project overview

This project uses supervised **multiclass classification** to identify which of **18 known large language models (LLMs)** generated a given text response. It is **not** a human-versus-AI detector. We compare a TensorFlow/Keras multilayer perceptron (MLP) with two conventional classifiers, Logistic Regression and a Linear Support Vector Machine (SVM), using shared training, validation, and held-out test data.

## Dataset

**Source:** [Kaggle — Name That LLM](https://www.kaggle.com/competitions/name-that-llm/data)

The repository includes `train.csv`, containing **2,699 labeled passages**, **150 prompts**, and **18 source-model labels**. Its source columns are `id`, `prompt_id`, `text`, and `model`. Text is the primary input; `model` is the target label. The `prompt_id` column is used to prevent passages answering the same prompt from crossing data splits.

| Split | Passages | Purpose |
| --- | ---: | --- |
| Training | 1,890 | Fit text transformations and train models |
| Validation | 396 | Choose hyperparameters and early stopping |
| Held-out test | 413 | Final evaluation after model selection |

All 18 labels appear in each split. **Important:** The repository's Kaggle `test.csv` is unlabeled; the final comparison uses the labeled, prompt-grouped holdout taken from `train.csv`, not Kaggle's `test.csv`.

## Preprocessing and feature selection

The notebook checks for missing values, blank passages, duplicate rows, class balance, and passage-length outliers. It then:

1. Splits passages by `prompt_id` (random seed `42`), with no prompt overlap among training, validation, and test sets.
2. Label-encodes the 18 model names.
3. Extracts word TF-IDF (1–2 grams; up to 5,000 features) and character TF-IDF (3–5 grams; up to 10,000 features).
4. Adds five scaled writing-style measurements.
5. Applies chi-square selection to TF-IDF features, preserving the five style features.
6. Compares **all 15,005 features** against **5,005 selected features**.

TF-IDF vectorizers, feature selector, and scaler are fitted using **training data only** to prevent leakage.

## Models

- **Logistic Regression:** regularization strength `C` selected using validation data.
- **Linear SVM (`LinearSVC`):** regularization strength `C` selected using validation data.
- **Final Keras MLP:** two ReLU hidden layers of **128 and 64 neurons**, dropout and L2 regularization, 18-way softmax output, Adam optimizer (`learning_rate=0.001`), sparse categorical cross-entropy loss, `batch_size=64`, and up to **35 epochs**. Two dropout/L2 configurations are compared separately on each feature set, with early stopping monitored on validation loss (`patience=6`, `restore_best_weights=True`). Validation macro F1 is the primary Keras configuration-selection criterion.

The notebook preserves an **earlier scikit-learn MLP** in **Section 9A** as a *historical baseline*. The required two-hidden-layer neural network and its final results are in **Sections 9B and 10B**. Do **not** use Section 10A's old MLP results as final Keras results.

## Final held-out test results

The following values are rounded to three decimal places from the completed run. All six combinations use the same **413 test passages**.

| Classifier | Feature set | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| Logistic Regression | All (15,005) | 0.334 | 0.315 |
| Logistic Regression | Selected (5,005) | 0.327 | 0.306 |
| Linear SVM | All (15,005) | **0.356** | **0.334** |
| Linear SVM | Selected (5,005) | 0.346 | 0.321 |
| Keras MLP | All (15,005) | 0.322 | 0.300 |
| Keras MLP | Selected (5,005) | 0.310 | 0.284 |

**Best model:** Linear SVM with all 15,005 features (35.6% test accuracy; macro F1 = 0.334). Chi-square selection did not improve test macro F1 for any of the three model families. The Keras MLP's training/validation curves show overfitting: training loss continued to fall after validation loss plateaued or rose. This comparison does not establish source identification for unknown LLMs or out-of-distribution text.

For macro precision, macro recall, exact unrounded results, and feature-selection deltas, run **Section 10B** and review the generated CSV files.

## How to run the notebook

1. Download or clone the repository. Keep `NameThatLLM.ipynb` and `train.csv` together in the repository root.
2. Open a terminal **in the repository folder** (so relative paths resolve properly).
3. Optionally set up a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

4. Install requirements and launch Jupyter:

   ```powershell
   python -m pip install -r requirements.txt
   jupyter lab
   ```

5. Open `NameThatLLM.ipynb`, then select **Kernel → Restart Kernel and Run All Cells**.
6. Save the executed notebook once all sections finish.

**Environment used in the documented run:** Python 3.13.9, TensorFlow 2.21.0, Keras 3.15.1, scikit-learn 1.7.2, NumPy 2.3.5; TensorFlow backend on Windows CPU (no GPU detected). Results may vary slightly with library versions, hardware, or non-deterministic operations.

**For teammates:** Do not hard-code a personal Windows path with `os.chdir(r"C:\\Users\\...")` inside the notebook. Launch Jupyter from the repository directory instead. The dataset-reading code should use `pd.read_csv("train.csv")`.

## Repository files and outputs

- `NameThatLLM.ipynb` — EDA, preprocessing, selected features, traditional models, Keras MLP, final evaluation, discussion, and conclusion.
- `train.csv` — labeled dataset used for the experiments.
- `test.csv` and `sample_submission.parquet` — Kaggle competition files, **not** the labeled evaluation set.
- `figures/mlp_learning_curves_all.png` and `figures/mlp_learning_curves_selected.png` — Keras training and validation loss/accuracy curves.
- `figures/confusion_*.png` — six held-out-test confusion matrices (three classifiers × two feature sets).
- `final_test_results.csv` — final metrics (generated when Section 10B runs).
- `feature_selection_effects.csv` — selected-minus-full comparison (generated when Section 10B runs).
- `models/mlp_all.keras` and `models/mlp_selected.keras` — winning saved Keras models (generated locally; may be omitted from GitHub if too large).
- `requirements.txt` — packages needed for rerunning the notebook.

## Team contributions

The team report template assigns Garrett Ford the introduction, dataset/EDA, and preprocessing/feature-selection write-up; Ferris Farmer the traditional-model and final-comparison write-up; and Caden Raleigh the MLP architecture/training, discussion, and conclusion write-up. The notebook retains the team's previous experimentation and incorporates the Keras MLP and final evaluation work. Specific coding contributions should be confirmed by the team.

## References

- Kaggle. *Name That LLM* [Data set]. https://www.kaggle.com/competitions/name-that-llm/data
- Keras documentation: https://keras.io/
- Scikit-learn documentation: https://scikit-learn.org/stable/
- TensorFlow documentation: https://www.tensorflow.org/
