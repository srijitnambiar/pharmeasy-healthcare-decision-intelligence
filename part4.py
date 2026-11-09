"""Part 4: handwritten split criteria, ensembles, AdaBoost and architecture sizing."""
import math
from pathlib import Path
import numpy as np
from shared import (read_data, split_standardise, sigmoid, train_logistic,
                    confusion, metrics, cost, write_csv, knn_probabilities,
                    best_k_from_training)

OUT = Path('outputs')

def impurity(labels, method):
    if len(labels) == 0:
        return 0.0
    p = float(np.mean(labels))
    if method == 'gini':
        return 2*p*(1-p)
    return -sum(v*math.log2(v) for v in (p,1-p) if v > 0)

def split_masks(x):
    return [x[:,5] == 1, x[:,6] == 1, x[:,2] > .5, x[:,0] <= 24]

def weighted_gain(y, mask, method):
    n = len(y)
    return impurity(y,method) - (mask.sum()*impurity(y[mask],method)
                                     + (~mask).sum()*impurity(y[~mask],method))/n

def ada_round(y, h, weights):
    error = float(np.sum(weights[h != y]))
    alpha = .5*math.log((1-error)/error)
    unnormalised = weights*np.exp(-alpha*y*h)
    return error, alpha, unnormalised, unnormalised/unnormalised.sum()

def main():
    OUT.mkdir(exist_ok=True)
    x, y, _ = read_data()
    xs, yt, xt, ytest = split_standardise(x,y)
    # Recover the exact file-order training rows for the unstandardised splits.
    selected = np.zeros(len(y),dtype=bool)
    for outcome in (0,1):
        indices = np.flatnonzero(y==outcome)
        selected[indices[:7*len(indices)//10]] = True
    xraw_train, xraw_test = x[selected], x[~selected]
    masks = split_masks(xraw_train)
    test_masks = split_masks(xraw_test)
    result = {'root_entropy':impurity(yt,'entropy'),
              'root_gini':impurity(yt,'gini')}
    split_rows = []
    for i,mask in enumerate(masks,1):
        split_rows.append((f'S{i}',int(mask.sum()),int(yt[mask].sum()),
                           int((~mask).sum()),int(yt[~mask].sum()),
                           weighted_gain(yt,mask,'entropy'),
                           weighted_gain(yt,mask,'gini')))
    write_csv(OUT/'part4_splits.csv',
              ('split','n_yes','positives_yes','n_no','positives_no','info_gain','gini_gain'),split_rows)
    best_info = max(range(4),key=lambda i:split_rows[i][5])
    best_gini = max(range(4),key=lambda i:split_rows[i][6])
    result['best_split_by_info_gain'] = f'S{best_info+1}'
    result['best_split_by_gini_gain'] = f'S{best_gini+1}'
    mask, test_mask = masks[best_info], test_masks[best_info]
    yes_rate = float(np.mean(yt[mask]))
    no_rate = float(np.mean(yt[~mask]))
    result['stump_rate_yes'],result['stump_rate_no'] = yes_rate,no_rate
    w,b,_ = train_logistic(xs,yt,.1,1000)
    logprob = sigmoid(xt@w+b)
    k = best_k_from_training(xs,yt)
    knnprob = knn_probabilities(xs,yt,xt,k)
    stumpprob = np.where(test_mask,yes_rate,no_rate)
    logistic = logprob >= .3
    knn = knnprob >= (k+1)/(2*k)
    stump = stumpprob >= .3
    hard = (logistic.astype(int)+knn.astype(int)+stump.astype(int))>=2
    soft = (logprob+knnprob+stumpprob)/3 >= .3
    votes = []
    for name,pred in [('logistic',logistic),('knn',knn),('stump',stump),
                      ('hard_vote',hard),('soft_vote',soft)]:
        m = metrics(confusion(ytest,pred))
        votes.append((name,*(m[key] for key in ('tp','fp','tn','fn','precision','recall','f1'))))
    write_csv(OUT/'part4_voting.csv',
              ('model','tp','fp','tn','fn','precision','recall','f1'),votes)
    actual = np.array([1,1,1,-1,-1,-1,1,-1])
    h1 = np.array([1,1,-1,-1,-1,-1,-1,-1])
    h2 = np.array([-1,1,1,-1,1,-1,1,-1])
    weights = np.full(8,1/8)
    ada_rows=[]
    alphas=[]
    for round_no,h in ((1,h1),(2,h2)):
        error,alpha,unnorm,new_weights = ada_round(actual,h,weights)
        result[f'round{round_no}_error']=error
        result[f'round{round_no}_alpha']=alpha
        alphas.append(alpha)
        ada_rows.extend((round_no,i+1,float(weights[i]),float(unnorm[i]),
                         float(new_weights[i])) for i in range(8))
        weights=new_weights
    write_csv(OUT/'part4_adaboost.csv',
              ('round','point','weight_before','weight_unnormalised','weight_after'),ada_rows)
    result['ensemble_accuracy']=float(np.mean(np.sign(alphas[0]*h1+alphas[1]*h2)==actual))
    result['oob_fraction']=(1-1/len(yt))**len(yt)
    arch=[('mlp_hidden1',(784+1)*16),('mlp_hidden2',(16+1)*16),
          ('mlp_output',(16+1)*10),('cnn_conv1',(5*5*1+1)*6),
          ('cnn_conv2',(5*5*6+1)*16),('cnn_dense1',(5*5*16+1)*32),
          ('cnn_dense2',(32+1)*10)]
    write_csv(OUT/'part4_architecture.csv',('layer','parameters'),arch)
    result['mlp_total']=sum(v for _,v in arch[:3])
    result['cnn_total']=sum(v for _,v in arch[3:])
    result['fc_layer_params']=(128*128*3+1)*1024
    result['conv_layer_params']=(3*3*3+1)*64
    result['rnn_total']=128*64+128*128+16*128+128+16
    with (OUT/'part4_results.txt').open('w',encoding='utf-8') as f:
        for key,val in result.items():
            f.write(f'{key} = {val if isinstance(val,str) else str(val) if isinstance(val,int) else f"{val:.4f}"}\n')
    print('Part 4 complete:',len(result),'result keys; best split:',result['best_split_by_info_gain'])
    return result,votes,ada_rows,arch

if __name__=='__main__':
    main()
