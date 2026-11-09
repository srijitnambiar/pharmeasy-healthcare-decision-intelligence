"""Part 1: estimation, inference, regression, and gradient descent.
Run from the repository root after generate_data.py.
All reported quantities are calculated from data/records.csv.
"""
import csv
import math
from pathlib import Path
import numpy as np

DATA = Path('data/records.csv')
OUT = Path('outputs/part1_results.txt')


def fit_line(x, y):
    """OLS line using explicitly calculated sample covariance and variance."""
    n = len(x)
    xbar, ybar = np.mean(x), np.mean(y)
    covariance = np.sum((x - xbar) * (y - ybar)) / (n - 1)
    variance = np.sum((x - xbar) ** 2) / (n - 1)
    slope = covariance / variance
    intercept = ybar - slope * xbar
    fitted = intercept + slope * x
    residuals = y - fitted
    tss = np.sum((y - ybar) ** 2)
    ess = np.sum((fitted - ybar) ** 2)
    rss = np.sum(residuals ** 2)
    return dict(slope=slope, intercept=intercept, fitted=fitted,
                residuals=residuals, tss=tss, ess=ess, rss=rss,
                r2=1-rss/tss, rmse=np.sqrt(rss/n),
                mae=np.mean(np.abs(residuals)))


def split_residual_sds(model):
    order = np.argsort(model['fitted'], kind='stable')
    half = len(order) // 2
    residuals = model['residuals']
    lower = np.std(residuals[order[:half]], ddof=1)
    upper = np.std(residuals[order[half:]], ddof=1)
    return lower, upper, upper/lower


def main():
    with DATA.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    def column(name):
        return np.array([float(r[name]) for r in rows], dtype=float)

    x1 = column('months_registered')
    x2 = column('missed_visit_count_12m')
    x3 = column('copay_burden_ratio')
    x4 = column('avg_bill_inr_thousands')
    x5 = column('adherence_score')
    x6 = column('chronic_condition_flag')
    x7 = column('insured_flag')
    y = column('care_plan_cost_inr_thousands')
    results = {}

    # 1.1: estimate mean and normal-approximation confidence interval.
    mean = np.mean(y)
    sd = np.std(y, ddof=1)
    se = sd / math.sqrt(n)
    results.update(n=n, mean_value=mean, sd_value=sd, se_value=se,
                   ci95_low=mean-1.96*se, ci95_high=mean+1.96*se)

    # 1.2: two-sided Z test of the INR 8,000 planning assumption.
    z_stat = (mean - 8.0)/se
    results.update(z_stat=z_stat, reject_h0='yes' if abs(z_stat)>1.96 else 'no')

    # 1.3: minimum sample size for margin of error 0.5 thousand INR.
    results['n_required'] = math.ceil((1.96*sd/0.5)**2)

    # 1.4: closed-form simple OLS regression.
    raw = fit_line(x4, y)
    results.update(reg_slope=raw['slope'], reg_intercept=raw['intercept'],
                   tss=raw['tss'], ess=raw['ess'], rss=raw['rss'],
                   tss_equals_ess_plus_rss=('yes' if abs(raw['tss']-raw['ess']-raw['rss']) < 1e-6*raw['tss'] else 'no'),
                   r2=raw['r2'], rmse=raw['rmse'], mae=raw['mae'])

    # 1.5: full-batch gradient descent on standardized X4.
    xbar = np.mean(x4)
    xsd = np.std(x4, ddof=1)
    z = (x4-xbar)/xsd
    m, c = 0.0, 0.0
    def cost(m, c):
        return np.mean((y-(m*z+c))**2)
    results['gd_J0'] = cost(m, c)
    for step in range(1, 501):
        e = y-(m*z+c)
        dm = -2.0*np.mean(z*e)
        dc = -2.0*np.mean(e)
        m -= 0.1*dm
        c -= 0.1*dc
        if step in (1, 10, 100, 500):
            results[f'gd_J{step}'] = cost(m, c)
    gd_slope = m/xsd
    gd_intercept = c-m*xbar/xsd
    results.update(gd_slope=gd_slope, gd_intercept=gd_intercept,
                   gd_slope_gap=abs(gd_slope-raw['slope']))

    # 1.6: residual funnel and log-log transformation.
    low, high, ratio = split_residual_sds(raw)
    results.update(resid_sd_lower_raw=low, resid_sd_upper_raw=high,
                   resid_sd_ratio_raw=ratio)
    logged = fit_line(np.log(x4), np.log(y))
    results.update(loglog_intercept=logged['intercept'], loglog_slope=logged['slope'],
                   loglog_r2=logged['r2'], loglog_rmse=logged['rmse'])
    low, high, ratio = split_residual_sds(logged)
    results.update(resid_sd_lower_log=low, resid_sd_upper_log=high,
                   resid_sd_ratio_log=ratio)

    # 1.7: normal-equation OLS with 7 predictors and a constant.
    X = np.column_stack([np.ones(n), np.log(x4), x1, x2, x3, x5, x6, x7])
    Y = np.log(y)
    beta = np.linalg.solve(X.T @ X, X.T @ Y)
    fitted = X @ beta
    mr_rss = np.sum((Y-fitted)**2)
    mr_tss = np.sum((Y-np.mean(Y))**2)
    mr_r2 = 1-mr_rss/mr_tss
    for i, b in enumerate(beta):
        results[f'mr_beta_{i}'] = b
    results.update(mr_r2=mr_r2,
                   mr_adj_r2=1-(1-mr_r2)*(n-1)/(n-7-1),
                   loglog_adj_r2=1-(1-logged['r2'])*(n-1)/(n-1-1))

    OUT.parent.mkdir(exist_ok=True, parents=True)
    with OUT.open('w', encoding='utf-8', newline='\n') as f:
        for key, value in results.items():
            if isinstance(value, str):
                shown = value
            elif isinstance(value, (int, np.integer)):
                shown = str(value)
            else:
                shown = f'{value:.4f}'
            f.write(f'{key} = {shown}\n')
    print(f'Wrote {OUT} ({len(results)} keys)')


if __name__ == '__main__':
    main()
