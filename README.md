# CM3203 Phishing Email Detection

This project is my CM3203 final-year dissertation project at Cardiff University. The main aim was to compare different NLP approaches for phishing email detection, starting with simple TF-IDF baselines and going up to deeper neural models and DistilBERT.

The main question behind it was pretty straightforward: are the bigger and more expensive models actually worth using? In some cases, yes, but not always by a huge margin. One of the more interesting outcomes was that a lightweight model like TF-IDF + LinearSVM could get extremely close to the heavier models on raw performance, while DistilBERT still had an advantage when it came to reducing false positives.

## Datasets

I used two datasets in this project, and they are very different in both size and difficulty.

**Kaggle Human-LLM Generated Phishing/Legitimate Emails** mixes real phishing emails written by humans with synthetic emails written by LLMs, across both phishing and legitimate classes. After cleaning and deduplication, the dataset comes down to 2,392 emails in total (1,724 legitimate and 668 phishing). That means it is relatively small, with 1,435 training examples and 957 test examples. Because the classes are imbalanced, F1 for the phishing class is more useful than just looking at overall accuracy. A nice thing about this dataset is that it also makes it possible to compare performance on human-written versus LLM-written emails.

**MeAJOR** is much larger and comes from several public email datasets combined together. After cleaning and deduplication, it contains 104,572 emails (57,947 legitimate and 46,625 phishing), split into 62,743 training examples and 41,829 test examples. The class balance is a bit tighter here, and the dataset is more varied in terms of writing style and sender domains, which makes it feel more realistic for an actual deployed phishing filter.

Both datasets use stratified 60/40 train/test splits with `random_state=42` so the comparisons stay consistent.

## Preprocessing

For the Kaggle data, I combined the `subject` and `body` fields into a single text input for the human-written emails. The LLM-generated rows already include a `text` column, so those did not need extra merging. After combining the four source files, I removed 1,202 duplicate texts and dropped 240 rows that did not have usable content.

For MeAJOR, I joined the subject and body together as well, but with explicit `Subject:` and `Body:` prefixes before running deduplication. Around 4,112 duplicate text-label pairs were removed from the original 108,685-row dataset, and one row with a missing label was dropped too.

The processed train/test splits are saved as Parquet files under `data/processed/`. They are not tracked in git because of file size.

## Models

I tested seven model families across both datasets.

**TF-IDF + classical classifiers.** These are the main lightweight baselines. I used a 50,000-term TF-IDF vocabulary with unigrams and bigrams, plus sublinear TF scaling. The vectoriser is always fitted only on the training set to avoid leakage. On top of that, I trained Multinomial Naive Bayes, Logistic Regression, Linear SVM, and Random Forest with 100 trees.

**SimHash + kNN.** This was included as a more unusual baseline. It compresses TF-IDF vectors into binary fingerprints using random projections, then classifies emails with k-nearest neighbours using Hamming distance. On the Kaggle dataset, I tried both 64-bit and 128-bit fingerprints and tested k values from 1 to 9 before choosing the best final setup.

**TextCNN.** This model uses convolutional filters of sizes 2, 3, and 4 over learned embeddings, followed by global max pooling and a final classification layer. In practice, it is good at picking up short local patterns and phrases that show up often in phishing emails.

**BiLSTM.** This is a two-layer bidirectional LSTM that reads sequences in both directions before classifying them. The MeAJOR version includes an extra dense hidden layer before the output because the larger dataset could support a slightly bigger model.

**DistilBERT.** For the transformer model, I fine-tuned `distilbert-base-uncased` for 5 epochs using AdamW with a learning rate of `2e-5`, weight decay of `0.01`, and a 10% warmup schedule. Sequences are truncated to 256 tokens. The pre-trained tokenizer is used as-is, so there is no custom vocabulary fitting.

The Kaggle deep learning notebooks use PyTorch, while the MeAJOR deep learning notebooks use Keras/TensorFlow. I kept the evaluation consistent by calculating the final metrics with scikit-learn after collecting predictions. Inference timing excludes DataLoader overhead and is averaged across three runs.

## Results

The main metrics reported here are accuracy, precision, recall, F1 for the phishing class, false positive rate, and inference time per email. I also put the combined tables into `results/tables/results_summary.ipynb`, with CSV exports saved in `results/tables/`.

### Kaggle dataset (957 test emails)

| Model | Accuracy | Precision | Recall | F1 | FPR | Inference (ms/email) |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + MultinomialNB | 0.8631 | 1.0000 | 0.5094 | 0.6749 | 0.0000 | 0.0013 |
| TF-IDF + LogisticRegression | 0.9415 | 1.0000 | 0.7903 | 0.8828 | 0.0000 | 0.0014 |
| TF-IDF + LinearSVM | 0.9906 | 0.9962 | 0.9700 | 0.9829 | 0.0014 | 0.0004 |
| TF-IDF + RandomForest | 0.9833 | 0.9960 | 0.9438 | 0.9692 | 0.0014 | 0.0278 |
| SimHash + kNN (128-bit, k=1) | 0.9122 | 0.8828 | 0.7903 | 0.8340 | 0.0406 | 0.0504 |
| TextCNN | 0.9833 | 0.9846 | 0.9551 | 0.9696 | 0.0058 | 1.5106 |
| BiLSTM | 0.9666 | 0.9181 | 0.9663 | 0.9416 | 0.0333 | 1.1693 |
| DistilBERT | 0.9885 | 0.9923 | 0.9663 | 0.9791 | 0.0029 | 23.6200 |

### MeAJOR dataset (41,829 test emails)

| Model | Accuracy | Precision | Recall | F1 | FPR | Inference (ms/email) |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + MultinomialNB | 0.9630 | 0.9846 | 0.9316 | 0.9574 | 0.0117 | 0.0006 |
| TF-IDF + LogisticRegression | 0.9770 | 0.9759 | 0.9724 | 0.9741 | 0.0193 | 0.0002 |
| TF-IDF + LinearSVM | 0.9845 | 0.9819 | 0.9834 | 0.9826 | 0.0146 | 0.0002 |
| TF-IDF + RandomForest | 0.9775 | 0.9857 | 0.9634 | 0.9744 | 0.0112 | 0.0378 |
| SimHash + kNN (128-bit, k=9) | 0.8615 | 0.8493 | 0.8379 | 0.8436 | 0.1196 | 1.2267 |
| TextCNN | 0.9831 | 0.9768 | 0.9855 | 0.9811 | 0.0189 | 0.3211 |
| BiLSTM | 0.9792 | 0.9779 | 0.9753 | 0.9766 | 0.0177 | 3.1347 |
| DistilBERT | 0.9896 | 0.9923 | 0.9842 | 0.9882 | 0.0062 | 7.6370 |

A few results stood out quite clearly. On the Kaggle dataset, LinearSVM basically keeps up with DistilBERT on F1 (0.983 vs 0.979) while being dramatically faster at inference. The more detailed source breakdown was also interesting: MultinomialNB struggled badly on LLM-generated phishing emails, with an F1 of just 0.06 there compared to 0.80 on human-written phishing emails. LinearSVM and RandomForest both did much better, which suggests they were picking up useful structural patterns rather than just memorising superficial wording.

On MeAJOR, the gap between models gets even tighter. LinearSVM reaches 0.983 F1, which is very close to DistilBERT at 0.989. Even so, DistilBERT still has the better false positive rate, which matters a lot in a real email setting. Across 41,829 test emails, that difference works out to roughly 290 legitimate emails incorrectly flagged by DistilBERT compared to about 625 for LinearSVM.

SimHash was probably the weakest overall approach, especially on MeAJOR. Its best setup there (128-bit, k=9) only reached 0.844 F1 with an FPR of 0.120, which is far worse than the TF-IDF baselines. My best guess is that once the dataset gets larger, the compressed fingerprints lose too much information and the Hamming-distance kNN search becomes less reliable around the decision boundary. The threshold and k sweeps also showed the usual trade-off between recall and false positives quite clearly.

## Repository structure

```
data/
  raw/               original source files (not tracked)
  processed/         cleaned Parquet splits (not tracked)
notebooks/
  kaggle_preparation.ipynb
  kaggle_classical_baselines.ipynb
  kaggle_deep_learning.ipynb
  kaggle_distilbert.ipynb
  kaggle_simhash.ipynb
  meajor_preparation.ipynb
  meajor_classical_baselines.ipynb
  meajor_deep_learning.ipynb
  meajor_distilbert.ipynb
  meajor_simhash.ipynb
results/
  figures/           confusion matrix and training curve images
  metrics/           per-model CSV result files
  tables/            consolidated summary notebook and exported comparison tables
models/
  classical/         saved model artifacts
```

The `results/` folder is organised by output type:

- `results/figures/` stores confusion matrices, threshold sweeps, and training curves.
- `results/metrics/` stores raw per-notebook CSV outputs, including `all_results.csv`.
- `results/tables/` stores the summary notebook (`results_summary.ipynb`) plus generated comparison tables such as `kaggle_summary_table.csv`, `meajor_summary_table.csv`, and `all_results_normalised.csv`.

## Setup

```bash
pip install -r requirements.txt
pip install transformers tensorflow seaborn
```

The `requirements.txt` file covers the main dependencies used across the project, including pandas, Parquet support, PyTorch, scikit-learn, matplotlib, and `transformers`. You still need to install `tensorflow` separately for the MeAJOR deep learning notebook. `seaborn` is optional and is only used in a few plotting cells.

The notebooks should be run in order within each dataset group: preparation first, then classical baselines, deep learning, DistilBERT, and SimHash. The preparation notebooks need to be run before the modelling notebooks because they generate the processed Parquet splits used everywhere else.
