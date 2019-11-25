import numpy as np
import pandas as pd
import pickle

from torch.utils.data import Dataset

def load_user_data(user, data_directory = '/mnt/array/valse_data/DeepLearning/Project/Pickle', load_web_mercator = False):
    
    with open(f'{data_directory}/TS_{user}.pickle', 'rb') as f:
        ts = pd.to_datetime(pickle.load(f))      
    
    with open(f'{data_directory}/label_{user}.pickle', 'rb') as f:
        labels = np.array(pickle.load(f)).astype(int)
   
    with open(f'{data_directory}/pos_lon_lat_UTM_{user}.pickle', 'rb') as f:
        pos_utm = np.array(pickle.load(f))

    if load_web_mercator:
        with open(f'{data_directory}/webM_user_{user}.pickle', 'rb') as f:
            pos_webm = np.array(pickle.load(f))
    else:
        pos_webm = None
                            
    return user, ts, pos_utm, pos_webm, labels

def create_data_frame(user, ts, pos_utm, pos_webm, labels, segmentation = False, seq_cutoff_time = 300, seq_cutoff_speed = 42):
    
    df = pd.DataFrame({
        'user': user,
        'ts': ts,
        'x': pos_utm[:, 0],
        'y': pos_utm[:, 1],
        'label': labels,
        'image_ix': np.arange(0, ts.shape[0])
    })
        
    if pos_webm is not None:
        df['x_web'] = pos_webm[:, 0]
        df['y_web'] = pos_webm[:, 1]
            
    df.sort_values('ts', inplace = True)
    
    df['delta_t'] = np.concatenate([[0], (df['ts'].values[1:] - df['ts'].values[:-1]) / pd.to_timedelta('1s')], axis = 0)    
    df['delta_d'] = np.concatenate([[0], np.linalg.norm(df[['x', 'y']][1:] - df[['x', 'y']][1:], axis = 1)], axis = 0)    
    df['speed'] = df['delta_d'] / df['delta_t'] 
    
    df = df[lambda x: x['delta_t'] > 0].copy()
    
    seq_bins = np.cumsum((df['delta_t'] >= seq_cutoff_time) | (df['speed'] > seq_cutoff_speed))
    seq_bin_ids, seq_bin_counts = np.unique(seq_bins, return_counts=True)
    
    if segmentation:
        df['segment_id'] = seq_bins
        df['segment_ix'] = np.concatenate([np.arange(seq_bin_counts[i]) for i in range(len(seq_bin_counts))])
        df['segment_point_count'] = np.repeat(seq_bin_counts, seq_bin_counts)
    
    return df
