"""Reusable data, training, metrics, and KNN; importing does not execute any Part."""
import csv
from pathlib import Path
import numpy as np
DATA = Path("data/records.csv")
OUTPUT = Path("outputs")

def read_data():
    with DATA.open(newline='', encoding='utf-8') as file:
        rows = list(csv.DictReader(file))
    columns = ['months_registered', 'missed_visit_count_12m',
               'copay_burden_ratio', 'avg_bill_inr_thousands',
               'adherence_score', 'chronic_condition_flag', 'insured_flag']
    x = np.array([[float(row[col]) for col in columns] for row in rows], dtype=float)
    y = np.array([int(row['no_show']) for row in rows], dtype=int)
    return x, y, columns


def split_standardise(x, y):
    selected = np.zeros(len(y), dtype=bool)
    for outcome in (0, 1):
        indices = np.flatnonzero(y == outcome)
        selected[indices[:7 * len(indices) // 10]] = True
    train_idx = np.flatnonzero(selected)
    test_idx = np.flatnonzero(~selected)
    mean = x[train_idx].mean(axis=0)
    sd = x[train_idx].std(axis=0, ddof=1)
    if np.any(sd == 0):
        raise ValueError('A training feature has zero sample standard deviation')
    return ((x[train_idx] - mean) / sd, y[train_idx],
            (x[test_idx] - mean) / sd, y[test_idx])


def sigmoid(z):
    # Numerically stable sigmoid, algebraically identical to 1/(1+exp(-z)).
    return np.exp(np.minimum(z, 0)) / (1 + np.exp(-np.abs(z)))


def bce(x, y, weights, bias):
    logits = x @ weights + bias
    # log(1+exp(z)) - y*z is binary cross-entropy, computed stably.
    return float(np.mean(np.logaddexp(0, logits) - y * logits))


def train_logistic(x, y, eta, updates, snapshots=()):
    weights = np.zeros(x.shape[1], dtype=float)
    bias = 0.0
    history = {0: bce(x, y, weights, bias)}
    for step in range(1, updates + 1):
        errors = sigmoid(x @ weights + bias) - y
        grad_w = x.T @ errors / len(y)
        grad_b = float(np.mean(errors))
        weights -= eta * grad_w
        bias -= eta * grad_b
        if step in snapshots:
            history[step] = bce(x, y, weights, bias)
    return weights, bias, history


def confusion(y, predicted):
    tp = int(np.sum((y == 1) & (predicted == 1)))
    fp = int(np.sum((y == 0) & (predicted == 1)))
    tn = int(np.sum((y == 0) & (predicted == 0)))
    fn = int(np.sum((y == 1) & (predicted == 0)))
    return dict(tp=tp, fp=fp, tn=tn, fn=fn)


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def metrics(cells):
    tp, fp, tn, fn = (cells[key] for key in ('tp', 'fp', 'tn', 'fn'))
    return {**cells,
            'accuracy': safe_ratio(tp + tn, tp + fp + tn + fn),
            'precision': safe_ratio(tp, tp + fp),
            'recall': safe_ratio(tp, tp + fn),
            'f1': safe_ratio(2 * tp, 2 * tp + fp + fn),
            'specificity': safe_ratio(tn, tn + fp)}


def knn_predict(x_reference, y_reference, x_queries, k):
    predictions = np.empty(len(x_queries), dtype=int)
    for i, query in enumerate(x_queries):
        distances_squared = np.sum((x_reference - query) ** 2, axis=1)
        nearest = np.argsort(distances_squared, kind='stable')[:k]
        predictions[i] = int(np.sum(y_reference[nearest]) >= (k + 1) // 2)
    return predictions


def cost(cells):
    return 50 * cells['fp'] + 400 * cells['fn']


def write_csv(path, header, rows):
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(header)
        for row in rows:
            writer.writerow([f'{v:.4f}' if isinstance(v, (float, np.floating))
                             else v for v in row])



def knn_probabilities(reference, labels, queries, k):
    probabilities = np.empty(len(queries))
    for i, query in enumerate(queries):
        squared = np.sum((reference-query)**2, axis=1)
        neighbours = np.argsort(squared, kind='stable')[:k]
        probabilities[i] = np.mean(labels[neighbours])
    return probabilities

def best_k_from_training(x, y):
    folds = np.arange(len(y)) % 5
    scores = []
    for k in range(1,22,2):
        fold_scores = []
        for fold in range(5):
            ref = folds != fold
            prediction = knn_predict(x[ref], y[ref], x[~ref], k)
            fold_scores.append(metrics(confusion(y[~ref], prediction))['f1'])
        scores.append((float(np.mean(fold_scores)), k))
    return max(scores, key=lambda pair: (pair[0], -pair[1]))[1]
