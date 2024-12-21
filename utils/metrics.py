import numpy as np

def mse_error(x, y):
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return np.nan
    return ((x - y)**2).mean()

def compute_firing_rate(V_record, threshold=0.0):
    # Compute firing rate as fraction of time steps membrane pot > threshold
    # V_record: [time, nodes]
    if not np.isfinite(V_record).all():
        return np.nan
    spikes = (V_record > threshold).sum(axis=0)
    firing_rate = spikes / V_record.shape[0]
    return firing_rate

def compute_synchronization(V_record):
    # A simple synchronization measure: 
    # Average the instantaneous phase coherence of spikes.
    # For simplicity, let's define: 
    # Sync = std(mean(V(t)) over neurons)
    # Lower std could mean more synchronization in a simplistic sense.
    if not np.isfinite(V_record).all():
        return np.nan
    mean_v = V_record.mean(axis=1)  # average across neurons at each time
    sync = mean_v.std()
    return sync

