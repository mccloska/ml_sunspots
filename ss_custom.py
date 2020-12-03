# Skill Scores
import numpy as np
from sklearn.metrics import brier_score_loss

def tss_calc(cont_table):
    TP = cont_table[1,1]
    TN = cont_table[0,0]
    FP = cont_table[0,1]
    POD = TP/(TP+FP)
    POFD = FP/(TN+FP)
    TSS = POD - POFD
    return TSS

# Calculate Brier Skill Score
def bss_calc(Y_val, predict_probs):
    clim_arr = np.full(len(Y_val), np.mean(Y_val))
    bs_metric = brier_score_loss(Y_val, predict_probs)
    bs_metric_clim = brier_score_loss(Y_val, clim_arr)

    return 1. - bs_metric/bs_metric_clim

