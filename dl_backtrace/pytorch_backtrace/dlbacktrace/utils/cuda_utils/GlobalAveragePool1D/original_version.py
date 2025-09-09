import numpy as np

def calculate_wt_gavgpool_1d(wts, inp):
    channels = wts.shape[0]
    wt_mat = np.zeros_like(inp)
    for c in range(channels):
        wt = wts[c]
        temp_wt = wt_mat[:, c]
        x = inp[:, c]
        p_mat = np.copy(x)
        n_mat = np.copy(x)
        p_mat[p_mat < 0] = 0
        n_mat[n_mat > 0] = 0
        p_sum = np.sum(p_mat)
        n_sum = np.sum(n_mat) * -1
        p_agg_wt = 0.0
        n_agg_wt = 0.0
        if p_sum + n_sum > 0.0:
            p_agg_wt = p_sum / (p_sum + n_sum)
            n_agg_wt = n_sum / (p_sum + n_sum)
        if p_sum == 0.0:
            p_sum = 1.0
        if n_sum == 0.0:
            n_sum = 1.0
        temp_wt = temp_wt + ((p_mat / p_sum) * wt * p_agg_wt)
        temp_wt = temp_wt + ((n_mat / n_sum) * wt * n_agg_wt * -1.0)
        wt_mat[:, c] = temp_wt
    return wt_mat