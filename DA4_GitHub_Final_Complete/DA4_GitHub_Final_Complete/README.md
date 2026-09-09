# DA-4 — Probabilistic Customer Segmentation and Segment Prediction

A reproducible MDI3003 Advanced Predictive Analytics DA-4 project covering probabilistic classification with Naive Bayes models and a TabTransformer comparison.

## Project layout

```text
DA4_Customer_Segmentation_NaiveBayes_TabTransformer/
├── DA4_Customer_Segmentation_NaiveBayes_TabTransformer.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── src/
│   └── README.md
└── lab04_outputs/
    ├── figures/
    ├── models/
    └── results/
```

## Datasets

### Dataset A — JanataHack Customer Segmentation
Place the supplied ZIP file with the exact filename:

`archive (1).zip`

The notebook extracts and uses `Train.csv`.

### Dataset B — UCI Bank Marketing
Place the supplied ZIP file with the exact filename:

`bank+marketing.zip`

The notebook extracts the nested bank marketing archives and uses `bank-additional-full.csv`.

The datasets themselves are not duplicated in this GitHub package unless required by the submission rules.

## Methodology

- Fixed random seed: 42
- Locked train/test split
- 5-fold stratified cross-validation on the training data
- Macro F1 as the primary model-selection metric
- Accuracy and weighted F1 also reported
- DummyClassifier baseline
- Gaussian Naive Bayes
- Bernoulli Naive Bayes
- Categorical Naive Bayes
- TabTransformer for Dataset A
- Locked test-set evaluation
- Confusion matrices
- Error analysis with prediction confidence
- Saved predictions and fitted model artifacts

## Running the notebook

Install dependencies:

```bash
pip install -r requirements.txt
```

Then open:

```text
DA4_Customer_Segmentation_NaiveBayes_TabTransformer.ipynb
```

Upload/place the two required ZIP datasets where the notebook can find them, then run all cells from top to bottom.

The notebook creates:

```text
da4_outputs/
├── figures/
├── models/
└── results/
```

These generated artifacts contain the evaluation plots, CSV result tables, predictions, and saved model files.

## Important

The notebook already contains executed outputs from the DA-4 analysis. Do not delete the output cells before submission if the evaluator needs to see the results directly in GitHub/Colab.
