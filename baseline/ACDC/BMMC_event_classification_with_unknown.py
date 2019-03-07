from src import * 

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix
import phenograph
from sklearn.cross_validation import StratifiedKFold
import pickle

channels = ['CD45','CD45RA', 'CD19', 'CD11b', 'CD4', 'CD8', 'CD34',
           'CD20', 'CD33', 'CD123', 'CD38', 'CD90', 'CD3']

path = '/home/disij/projects/acdc/data/BMMC_benchmark/'

df = pd.read_csv(path + 'BMMC_benchmark.csv.gz', sep=',', header = 0, compression = 'gzip')
df = df[df.cell_type != 'NotGated']


table = pd.read_csv(path + 'BMMC_table.csv', sep=',', header=0, index_col=0)
table = table.fillna(0)

cts, channels = get_label(table)

X0= np.arcsinh((df[channels].values - 1.0)/5.0)

idx2ct = [key for idx, key in enumerate(table.index)]
idx2ct.append('unknown')

ct2idx = {key:idx for idx, key in enumerate(table.index)}
ct2idx['unknown'] = len(table.index)
        
ct_score = np.abs(table.as_matrix()).sum(axis = 1)

## compute manual gated label
y0 = np.zeros(df.cell_type.shape)

for i, ct in enumerate(df.cell_type):
    if ct in ct2idx:
        y0[i] = ct2idx[ct]
    else:
        y0[i] = ct2idx['unknown']
        
from sklearn.metrics import accuracy_score, confusion_matrix
import phenograph
from sklearn.cross_validation import StratifiedKFold

n_neighbor = 10
thres = 0.5

import time

skf = StratifiedKFold(y0, n_folds=5, shuffle=True, random_state=0)
result = []
score_final = []


process_time = []
c = 0
for tr, te in skf:
    print('%02d th batch' % c)
    if c == 1:
        break
    c += 1
    
    X = X0.copy()
    y_true = y0.copy()

    X = X[tr, :]
    y_true = y_true[tr]

    mk_model =  compute_marker_model(pd.DataFrame(X, columns = channels), table, 0.0)

    ## compute posterior probs
    tic = time.clock()
    score = get_score_mat(X, [], table, [], mk_model)
    score = np.concatenate([score, 1.0 - score.max(axis = 1)[:, np.newaxis]], axis = 1)    

    ## get indices     
    ct_index = get_unique_index(X, score, table, thres)
    
    ## baseline - classify events    
    y_pred_index = np.argmax(score, axis = 1)
    
    toc = time.clock()
    time0 = toc - tic
    
    
    
    ## running ACDC
    tic = time.clock()
    res_c = get_landmarks(X, score, ct_index, idx2ct, phenograph, thres)

    landmark_mat, landmark_label = output_feature_matrix(res_c, [idx2ct[i] for i in range(len(idx2ct))]) 

    landmark_label = np.array(landmark_label)

    lp, y_pred = rm_classify(X, landmark_mat, landmark_label, n_neighbor)

    process_time.append(toc-tic)
    
    res = phenograph.cluster(X, k=30, directed=False, prune=False, min_cluster_size=10, jaccard=True,
                        primary_metric='euclidean', n_jobs=-1, q_tol=1e-3)
    
    toc = time.clock()
    time1 = toc - tic
    
    
    ## running phenograph classification
    tic = time.clock()
    y_pred_oracle = np.zeros_like(y_true)
    for i in range(max(res[0])+1):
        ic, nc = Counter(y_true[res[0] == i]).most_common(1)[0]
        y_pred_oracle[res[0] == i] = ic
        
    score_final.append([accuracy_score(y_true, [ct2idx[c] for c in y_pred]), 
                    accuracy_score(y_true, y_pred_index), 
                    accuracy_score(y_true, y_pred_oracle)])
    
    toc = time.clock()
    time2 = toc - tic   
    
    
    result.append((y_true, y_pred, y_pred_index, y_pred_oracle))
    process_time.append((time0, time1, time2))
    
    pickle.dump(result, open('output/BMMC/event_classidication_BMMC_with_unknown.p', 'wb'))

print("score of ACDC, score-based classification, phenograph classification:", np.mean(score_final, axis = 0)) # score of ACDC, score-based classification, phenograph classification