import numpy as np

def blockshaped(tx_data, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = tx_data.size
    """
    h, w = tx_data.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    
    chunks = (tx_data.reshape(h//nrows, nrows, -1, ncols).swapaxes(1,2).reshape(-1, nrows, ncols))
    return chunks



def get_features(window):
    """
    Bin FK window and calculate sum for each bin
    CURRENTLY: only for predefined length and frequencies
    """
    window_eff = np.abs(window[:100,:])
#     window_chunks = blockshaped(window_eff, 10, 5)
    window_chunks = blockshaped(window_eff, 10, 5)
    nFeatures = window_chunks.shape[0]
    features = np.zeros(nFeatures)

    for i in range(nFeatures):
        fk_bin = window_chunks[i,:,:]
        features[i] = sum(map(sum, fk_bin)) # Sums up all entries in one fk bin
    
    return features