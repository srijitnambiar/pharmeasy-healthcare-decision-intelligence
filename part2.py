"""Part 2 classification and cost evaluation."""
from pathlib import Path
import numpy as np
from shared import (read_data, split_standardise, sigmoid, bce, train_logistic,
                    confusion, safe_ratio, metrics, knn_predict, cost, write_csv)
OUTPUT = Path("outputs")

def main():
    OUTPUT.mkdir(exist_ok=True)
    x, y, _ = read_data()
    x_train, y_train, x_test, y_test = split_standardise(x, y)
    results = {}
    results['train_rows'] = len(y_train)
    results['train_positives'] = int(y_train.sum())
    results['test_rows'] = len(y_test)
    results['test_positives'] = int(y_test.sum())

    weights, bias, history = train_logistic(x_train, y_train, 0.1, 1000,
                                            snapshots=(1, 10, 100, 1000))
    for epoch in (0, 1, 10, 100, 1000):
        results[f'loss_epoch_{epoch}'] = history[epoch]
    for i, weight in enumerate(weights, 1):
        results[f'weight_X{i}'] = float(weight)
    results['bias'] = bias
    results['test_bce'] = bce(x_test, y_test, weights, bias)
    for eta in (0.01, 0.1, 1.0):
        _, _, hist = train_logistic(x_train, y_train, eta, 200, snapshots=(200,))
        results[f'loss_epoch_200_eta_{eta}'] = hist[200]
    for i, weight in enumerate(weights, 1):
        results[f'odds_ratio_X{i}'] = float(np.exp(weight))

    test_prob = sigmoid(x_test @ weights + bias)
    train_prob = sigmoid(x_train @ weights + bias)
    for threshold in (0.5, 0.3):
        for key, value in metrics(confusion(y_test, test_prob >= threshold)).items():
            results[f'tau_{threshold}_{key}'] = value
    results['blind_accuracy'] = float(np.mean(y_test == 0))
    threshold_rows = []
    for k in range(1, 10):
        tau = k / 10
        c = confusion(y_test, test_prob >= tau)
        threshold_rows.append((tau, c['tp'], c['fp'], c['tn'], c['fn'],
                               safe_ratio(c['tp'], c['tp'] + c['fn']),
                               safe_ratio(c['fp'], c['fp'] + c['tn'])))
    write_csv(OUTPUT / 'part2_thresholds.csv',
              ('tau', 'tp', 'fp', 'tn', 'fn', 'tpr', 'fpr'), threshold_rows)

    cv_rows = []
    folds = np.arange(len(y_train)) % 5
    for k in range(1, 22, 2):
        f1s = []
        for fold in range(5):
            reference = folds != fold
            validation = ~reference
            predictions = knn_predict(x_train[reference], y_train[reference],
                                      x_train[validation], k)
            f1s.append(metrics(confusion(y_train[validation], predictions))['f1'])
        cv_rows.append((k, float(np.mean(f1s))))
    write_csv(OUTPUT / 'part2_knn_cv.csv', ('K', 'mean_cv_f1'), cv_rows)
    best_k = max(cv_rows, key=lambda item: item[1])[0]  # first wins ties
    results['knn_best_k'] = best_k
    knn_test = knn_predict(x_train, y_train, x_test, best_k)
    for key in ('tp', 'fp', 'tn', 'fn', 'accuracy', 'precision', 'recall', 'f1'):
        results[f'knn_{key}'] = metrics(confusion(y_test, knn_test))[key]

    # Select threshold exclusively using training predictions and training labels.
    candidate_costs = [(cost(confusion(y_train, train_prob >= k / 20)), k / 20)
                       for k in range(1, 20)]
    best_train_cost, best_tau = min(candidate_costs)  # lower tau wins ties
    results['cost_tau'] = best_tau
    results['cost_train_at_tau'] = best_train_cost
    results['cost_test_at_tau'] = cost(confusion(y_test, test_prob >= best_tau))
    results['cost_test_at_0.5'] = cost(confusion(y_test, test_prob >= 0.5))
    results['cost_test_flag_nobody'] = cost(confusion(y_test, np.zeros_like(y_test)))

    with (OUTPUT / 'part2_results.txt').open('w', encoding='utf-8') as file:
        for key, value in results.items():
            if isinstance(value, (int, np.integer)):
                formatted = str(value)
            else:
                formatted = f'{value:.4f}'
            file.write(f'{key} = {formatted}\n')
    print('Part 2 complete:', len(results), 'result keys; best K:', best_k,
          '; cost threshold:', f'{best_tau:.2f}')
    return results


if __name__ == '__main__':
    main()
