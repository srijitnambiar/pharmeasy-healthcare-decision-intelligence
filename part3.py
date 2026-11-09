"""Part 3: deterministic K-means, agglomerative linkage and PCA, NumPy only."""
import csv
from pathlib import Path
import numpy as np

OUT = Path('outputs')
FEATURES = ['months_registered','missed_visit_count_12m','copay_burden_ratio','avg_bill_inr_thousands','adherence_score']

def fmt(x):
    return f'{float(x):.4f}'

def kmeans(z, k):
    n = len(z)
    centroids = z[np.arange(k) * (n // k)].copy()
    previous = np.full(n, -1, dtype=int)
    passes = 0
    while True:
        distance2 = ((z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(distance2, axis=1)
        passes += 1
        if np.array_equal(labels, previous):
            break
        for j in range(k):
            members = z[labels == j]
            if len(members):
                centroids[j] = members.mean(axis=0)
        previous = labels
    wcss = float(((z - centroids[labels]) ** 2).sum())
    return labels, centroids, wcss, passes

D = np.array([
[0,7.9,7.4,2.9,2.9,6.1],
[7.9,0,1.4,5.9,5.3,2.0],
[7.4,1.4,0,5.8,4.6,1.3],
[2.9,5.9,5.8,0,2.8,4.5],
[2.9,5.3,4.6,2.8,0,3.4],
[6.1,2.0,1.3,4.5,3.4,0]], dtype=float)

def linkage_distance(a, b, mode):
    distances = D[np.ix_(a,b)].ravel()
    return {'single': np.min, 'complete': np.max, 'average': np.mean}[mode](distances)

def label(cluster):
    return '+'.join('E'+str(i+1) for i in cluster)

def hierarchical(mode):
    clusters = [(i,) for i in range(6)]
    rows, heights, snapshots = [], [], []
    for merge_no in range(1,6):
        candidates = [(float(linkage_distance(a,b,mode)),i,j) for i,a in enumerate(clusters) for j,b in enumerate(clusters) if i<j]
        height,i,j = min(candidates)
        a,b = clusters[i], clusters[j]
        if a[0]>b[0]: a,b=b,a
        rows.append([mode,merge_no,label(a),label(b),fmt(height)])
        heights.append(height)
        clusters = [c for q,c in enumerate(clusters) if q not in (i,j)] + [tuple(sorted(a+b))]
        clusters.sort(key=lambda c:c[0])
        snapshots.append(clusters[:])
    gap_index = int(np.argmax(np.diff(heights)))
    # Gap after merge (gap_index+1): retain partition after that many merges.
    return rows, 6-(gap_index+1), snapshots[gap_index], heights

def main():
    with Path('data/records.csv').open(newline='',encoding='utf-8') as f:
        records=list(csv.DictReader(f))
    x=np.array([[float(r[c]) for c in FEATURES] for r in records])
    outcome=np.array([int(r['no_show']) for r in records])
    value=np.array([float(r['care_plan_cost_inr_thousands']) for r in records])
    z=(x-x.mean(axis=0))/x.std(axis=0,ddof=1)
    results={}
    wcss={}
    for k in range(2,7):
        labels,centers,score,passes=kmeans(z,k)
        wcss[k]=score
        results[f'wcss_k{k}']=score
        print(f'K={k}: WCSS={score:.4f}, assignment passes={passes}')
    elbow=next((k for k in (3,4,5) if wcss[k]-wcss[k+1]<0.5*(wcss[k-1]-wcss[k])),None)
    if elbow is None: raise RuntimeError('No elbow under the specified rule')
    results['elbow_k']=elbow
    labels,centers,_,_=kmeans(z,3)
    ordered=sorted(range(3),key=lambda j:x[labels==j,2].mean())
    segment_labels={name:labels==j for name,j in zip('ABC',ordered)}
    segment_rows=[]
    for name,mask in segment_labels.items():
        size=int(mask.sum())
        results[f'size_segment_{name}']=size
        segment_rows.append([name,size,*[fmt(v) for v in x[mask].mean(axis=0)],fmt(outcome[mask].mean()),fmt(value[mask].mean())])
    hierarchical_rows=[]
    for mode in ('single','complete','average'):
        rows,count,cut_clusters,heights=hierarchical(mode)
        hierarchical_rows.extend(rows)
        results[f'{mode}_clusters_at_cut']=count
        print(f'{mode}: heights={heights}; cut={list(map(label,cut_clusters))}')
    # PCA: centre standardised columns, form sample covariance, decompose, sort.
    centred=z-z.mean(axis=0)
    covariance=centred.T@centred/(len(z)-1)
    eigenvalues,eigenvectors=np.linalg.eigh(covariance)
    order=np.argsort(eigenvalues)[::-1]
    eigenvalues=eigenvalues[order]
    eigenvectors=eigenvectors[:,order]
    for j in range(eigenvectors.shape[1]):
        index=np.argmax(np.abs(eigenvectors[:,j]))
        if eigenvectors[index,j]<0: eigenvectors[:,j]*=-1
    ratios=eigenvalues/eigenvalues.sum()
    cumulative=np.cumsum(ratios)
    results['eigenvalue_sum']=float(eigenvalues.sum())
    for i in range(5):
        results[f'eigenvalue_{i+1}']=eigenvalues[i]
        results[f'explained_ratio_{i+1}']=ratios[i]
        results[f'cumulative_ratio_{i+1}']=cumulative[i]
    results['components_for_80pct']=int(np.searchsorted(cumulative,0.8)+1)
    for component in (1,2):
        for feature in range(1,6):
            results[f'pc{component}_loading_X{feature}']=eigenvectors[feature-1,component-1]
    projections=centred@eigenvectors[:,:2]
    for row in range(3):
        for component in (1,2):
            results[f'row{row+1}_pc{component}']=projections[row,component-1]
    for name,mask in segment_labels.items():
        results[f'mean_pc1_segment_{name}']=float(projections[mask,0].mean())
    OUT.mkdir(exist_ok=True)
    with (OUT/'part3_results.txt').open('w') as f:
        for key,v in results.items():
            f.write(f'{key} = {v if isinstance(v,(int,np.integer)) else fmt(v)}\n')
    with (OUT/'part3_segments.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['segment','size','mean_X1','mean_X2','mean_X3','mean_X4','mean_X5','outcome_rate','mean_value'])
        writer.writerows(segment_rows)
    with (OUT/'part3_hierarchical.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['linkage','merge_no','cluster_a','cluster_b','height'])
        writer.writerows(hierarchical_rows)
    print(f'Wrote {len(results)} results, {len(segment_rows)} segment rows, {len(hierarchical_rows)} merges')

if __name__=='__main__': main()
