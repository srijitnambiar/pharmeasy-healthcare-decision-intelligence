Domain: Healthcare

# PharmEasy Operations Analytics Capstone — Parts 1–4

This repository implements all four Parts on synthetic, invented patient records; no actual PharmEasy patient data are used. Every result is generated offline by the committed Python scripts.

## How to run

Use Python 3.10 or newer. From the repository root, run these commands **in order** on a clean clone:

```bash
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

The project uses NumPy 2.3.5 and runs offline. The generator must print `800 10 170 e26b46e82f1a1af44dcabc133050802e`. Each Part script regenerates its committed text and CSV outputs; all graded calculations use handwritten code and NumPy array operations. Run commands from the repository root so relative data and output paths resolve correctly.

## Part 1 findings

### R1.1 Planning claim

The two-tailed test **rejects H0** at alpha = 0.05: `z_stat = 2.8930` exceeds the critical absolute value of `1.96`. The observed mean is `8.9382` thousand INR, versus the planning assumption of `8.0` thousand INR. The difference is `0.9382` thousand INR, approximately INR 938.20 per patient, or 11.73% above the budget assumption. This difference could materially affect an aggregate care-plan budget, but its operational importance depends on patient volume and financial tolerance. A Type I error here would mean concluding the true average care-plan cost differs from INR 8,000 when it actually equals INR 8,000.

### R1.2 Sample size

The required sample is `n_required = 1293` for a 95% confidence interval with a margin of error of INR 500, versus the `800` available observations. The existing data are enough for a preliminary estimate and test but do not meet the desired precision under the stated sample-size formula; the team would need 493 additional independent observations under the same variability assumption.

### R1.3 Gradient descent

`gd_slope_gap = 0.0000` at four-decimal precision. Gradient descent converges to the closed-form OLS solution because both optimize the same convex squared-error objective for a linear model, and 500 full-batch steps with the specified learning rate are sufficient.

### R1.4 Residual spread

The raw residual spread ratio is `resid_sd_ratio_raw = 3.7525`, which indicates a strong funnel: larger predicted costs have much larger residual variation. After logging both cost and billing amount, `resid_sd_ratio_log = 1.0396`, indicating that the log transformation substantially removes the funnel on the log scale. The raw and logged R² values are not directly comparable because they use different target scales.

### R1.5 Multiple regression

Applying the requested mechanical rule of absolute coefficient greater than 0.05 (excluding the intercept), the selected predictors are `ln(avg_bill_inr_thousands)` (`1.1576`), `missed_visit_count_12m` (`0.0940`), `copay_burden_ratio` (`0.1067`), and `chronic_condition_flag` (`0.1518`). This is not a ranking of importance or statistical significance: the predictors have different units and coefficient uncertainty is not assessed here. The multiple-regression adjusted R² is `mr_adj_r2 = 0.7403`, versus `loglog_adj_r2 = 0.6561` for the one-predictor log-log model. On the common log-cost target scale, the additional predictors improve explanatory fit even after the complexity penalty.

## Part 2 findings

### R2.1 Learning rate

After 200 updates the losses, from lowest to highest, are eta 1.0 (`0.3703`), eta 0.1 (`0.3776`), and eta 0.01 (`0.5091`). The smallest learning rate converges slowly, while the largest of these tested rates converges faster without instability on this dataset; larger rates are not universally safe.

### R2.2 Coefficients and odds

The three largest absolute standardised weights are X3 copay burden (`0.9013`), X1 months registered (`-0.5554`), and X2 missed visits (`0.5505`). The largest odds ratio is X3 (`2.4629`): for one training-SD increase in copay burden, holding the other standardised predictors fixed, predicted no-show odds multiply by about 2.46.

### R2.3 Class imbalance and thresholds

The always-attend baseline has `blind_accuracy = 0.7875`, while logistic at tau 0.5 has accuracy `0.8042`; accuracy obscures costly missed no-shows because most patients attend. Lowering tau to 0.3 increases recall from `0.4706` to `0.7451`, while precision falls from `0.5455` to `0.4634`.

### R2.4 KNN comparison

Logistic at tau 0.3 has F1 `0.5714` and recall `0.7451`, versus the cross-validated KNN (K=5) F1 `0.3696` and recall `0.3333`. Logistic captures substantially more missed appointments.

### R2.5 Deployment threshold

Deploy logistic regression with training-selected cost threshold `0.1500`: test cost `INR 6,050`, versus `INR 11,800` at tau 0.5 and `INR 20,400` for flagging nobody. Choosing the threshold on training rather than test rows protects the test evaluation from tuning bias.

## Part 3 findings

### R3.1 K-means elbow

The four successive WCSS reductions are `659.8282` (K=2 to 3), `108.0739` (3 to 4), `171.0575` (4 to 5), and `45.5466` (5 to 6). The specified half-previous-drop rule first holds at `elbow_k = 3`, because 108.0739 is less than half of 659.8282. The drops do not decrease monotonically: the pinned initial centroids make the solutions reproducible but do not guarantee the global WCSS minimum.

### R3.2 Patient profiles and service actions

- **A — Established, regular-service users**: 322 records; average tenure 47.5028 months, 1.9503 missed visits, copay burden 0.2250, and outcome rate 0.0186. Offer self-service scheduling and avoid unnecessary calls.
- **B — Higher-spend, frequent-contact users**: 255 records; mean bill 28.7192 thousand INR, 6.0706 missed visits and outcome rate 0.2353. Offer billing guidance and appointment confirmations.
- **C — Newer, higher-friction users**: 223 records; mean tenure 14.3892 months, copay burden 0.7118, adherence score 38.5215 and outcome rate **0.4664**, the highest. Prioritise reminders and help with rescheduling. These labels describe service usage, not medical status.

### R3.3 Hierarchical linkage

Single linkage has its largest merge-height gap after merge 2 and therefore cuts into four groups: **E1**, **E2+E3+E6**, **E4**, **E5**. Complete linkage has its largest gap after merge 4 and therefore cuts into two groups: **E1+E4+E5** and **E2+E3+E6**. Single linkage uses the closest cross-cluster pair and can join clusters by a short chain, while complete linkage measures the most distant pair and postpones joining broad, dispersed groups. Their largest merge-height gaps therefore occur at different stages on this table.

### R3.4 Principal components

After standardisation, the five-column mean vector is approximately `(0, 0, 0, 0, 0)`; the data are centred again before forming the sample covariance matrix. PC1 has positive loadings for X1 (`0.5317`) and X5 (`0.4418`) and negative loadings for X2 (`-0.4100`), X3 (`-0.5187`) and X4 (`-0.2916`), so tenure and adherence oppose missed visits, copay burden and bill amount along this axis. **Three components** retain at least 80% variance (`0.8616` cumulative); two retain only `0.7593`. Mean PC1 by segment is A `1.7773`, B `-0.7831`, C `-1.6709`, showing strong separation between established users and newer higher-friction users.

## Part 4 findings

### R4.1 Decision split

S4 (months registered <= 24) is the strongest split by both information gain (`0.1252`) and Gini gain (`0.0606`), larger than the other three candidates. Growing a tree until every leaf is pure can memorise sampling noise and generalise poorly; a minimum leaf size or maximum depth prevents excessive growth.

### R4.2 Voting comparison

Ship logistic regression: at the common evaluation threshold 0.3, its F1 is `0.5714` and recall `0.7451`. KNN achieves F1 `0.3696`, recall `0.3333`; stump `0.5532`, `0.7647`; hard voting `0.5344`, `0.6863`; and soft voting `0.5507`, `0.7451`. Neither voting method improves F1 over logistic, though the stump alone slightly improves recall. For production, use the separately training-selected logistic cost threshold 0.15.

### R4.3 Boosting weights

After round 1, the two incorrectly classified points (3 and 7) each have weight `0.2500`, for total `0.5000`. Round 2's h2 receives greater voting weight (`alpha = 0.8047`, versus h1's `0.5493`) because its weighted error is smaller (`0.1667` versus `0.2500`).

### R4.4 Architecture sizing

Model A CNN output shapes and parameter counts (ReLU and pooling have no learned parameters):

| Layer | Output shape | Parameters |
|---|---|---:|
| Input | 28 × 28 × 1 | 0 |
| Conv1, 5×5, 6 filters, pad 2 | 28 × 28 × 6 | 156 |
| ReLU1 | 28 × 28 × 6 | 0 |
| Max-pool1, 2×2 | 14 × 14 × 6 | 0 |
| Conv2, 5×5, 16 filters, pad 0 | 10 × 10 × 16 | 2,416 |
| ReLU2 | 10 × 10 × 16 | 0 |
| Max-pool2, 2×2 | 5 × 5 × 16 | 0 |
| Flatten | 400 | 0 |
| Dense1, ReLU | 32 | 12,832 |
| Dense2 | 10 | 330 |

The CNN has `15,734` parameters; the 784→16→16→10 MLP has `13,002`. At 128×128×3 input, a dense layer with 1,024 neurons has `50,332,672` parameters versus `1,792` for 64 convolution filters of size 3×3, because convolution shares filter weights across spatial positions.

Model B's matrix shapes are W_xh **128×64**, W_hh **128×128**, W_hy **16×128**, with bias vectors of lengths 128 and 16, totalling `26,768` parameters. Sequence length does not change the parameter count because the same recurrent weights are reused at every step. Use ReLU in hidden CNN layers and 10-way softmax with categorical cross-entropy at the output for mutually exclusive scan queues. A very deep plain network can suffer vanishing gradients, leaving early layers with negligible updates. A residual skip connection creates a shorter gradient path and makes optimisation easier.

## Model-choice memo

**To: Head of Scheduling Operations**

I recommend deploying the from-scratch logistic no-show model first, with an alert threshold of **0.15** selected on the training set to minimise the stated cost of false reminders and missed slots. On the untouched 240-row test set, this threshold produces a total estimated cost of **INR 6,050**, compared with **INR 11,800** at the default threshold of 0.5 and **INR 20,400** when nobody is flagged.

At the common evaluation threshold of 0.3, logistic regression achieves **0.5714 F1** and **0.7451 recall**. Hard voting reaches only **0.5344 F1** and **0.6863 recall**, and soft voting reaches **0.5507 F1** and **0.7451 recall**. The extra ensemble complexity therefore has no demonstrated F1 benefit. I would deploy logistic with operational monitoring and periodic threshold review, rather than a voting ensemble.

The evidence comes from synthetic data and a single fixed holdout split; real reminder effectiveness, costs, calibration, fairness and drift still require validation before production use. Neither the scan-routing CNN (Model A) nor the referral-note RNN (Model B) should be built ahead of validating and deploying the structured scheduling model.

## Limitations

1. **Synthetic dataset:** The 800 records are invented and cannot establish real PharmEasy patient behaviour, reminder effectiveness, or costs.
2. **Single held-out split:** The 240-row test set has only 51 positive outcomes; results may vary with another sample, time period, or operational population.
3. **Fixed cost assumptions:** INR 50 per false alarm and INR 400 per missed no-show are scenario assumptions; actual call costs, recovered appointments, and capacity effects may differ.
4. **No external validation:** The logistic model and segments have not been prospectively tested, calibrated, audited for fairness, or monitored for drift.
5. **Simplified architecture exercise:** CNN and RNN parameter counts do not establish training feasibility, routing accuracy, security, or production suitability; neither network was trained.

All findings concern scheduling, cost planning and routing only; they are not clinical recommendations.
