import numpy as np
import pandas as pd
import pickle

from torch.utils.data import Dataset

#requirements for random pick of users
import random

#requirements for plot_filtered()
import datashader.transfer_functions as dtf
from colorcet import fire
import datashader as ds
import holoviews as hv
import geoviews as gv
from holoviews.operation.datashader import datashade
hv.extension('bokeh')
#Interactive image
from datashader.bokeh_ext import InteractiveImage
import bokeh.plotting as bp

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
    df['delta_d'] = np.concatenate([[0], np.linalg.norm(df[['x', 'y']].values[1:] - df[['x', 'y']].values[:-1], axis = 1)], axis = 0)    
    df['bearing'] = np.concatenate([[0], np.arctan2(df[['y']].values[1:] - df[['y']].values[:-1], df[['x']].values[1:] - df[['x']].values[:-1]).reshape(-1)], axis = 0)
    df['speed'] = df['delta_d'] / df['delta_t'] 
    
    df = df[lambda x: x['delta_t'] > 0].copy()
    
    seq_bins = np.cumsum((df['delta_t'] >= seq_cutoff_time) | (df['speed'] > seq_cutoff_speed))
    seq_bin_ids, seq_bin_counts = np.unique(seq_bins, return_counts=True)
    
    if segmentation:
        df['segment_id'] = seq_bins
        df['segment_ix'] = np.concatenate([np.arange(seq_bin_counts[i]) for i in range(len(seq_bin_counts))])
        df['segment_point_count'] = np.repeat(seq_bin_counts, seq_bin_counts)
    
    return df

def train_test_data_split(u=12, random = False):
    
    if random:        
        #Pick random users
        users = range(0,u)
        #Split Test/Train according to random pick
        train_val = random.sample(population=users, k=9) 

        train = train_val[:-1]
        val = train_val[-1:]
        test  = [x for x in users if x not in train_val]
    else:
        train, val, test = ([8, 6, 4, 5, 9, 1, 11, 7], [2], [0, 3, 10])
    
    return train, val, test

import bokeh.plotting as bp
from bokeh.models.tiles import WMTSTileSource
bp.output_notebook()

# Default plot ranges:
def create_image_wrap(fdf, col, w=1000, h=900, x_range = (1373757.1102773394, 1412506.1502695908), y_range = (7478418.9895278225, 7520786.118694777), background = 'black'):
    
    def create_image(x_range=x_range, y_range=y_range, w=w, h=h):
        cvs = ds.Canvas(x_range=x_range, y_range=y_range, plot_height=h, plot_width=w)
    
    
        agg = cvs.points(fdf, 'x_web', 'y_web', agg=ds.mean(col))
        image = dtf.shade(agg, cmap=fire)
        return dtf.dynspread(image, threshold=0.75, max_px=8)

    return create_image

def base_plot(tools='pan,wheel_zoom,reset', w=1000, h=900, x_range = (1373757.1102773394, 1412506.1502695908), y_range = (7478418.9895278225, 7520786.118694777), background = 'black'):
    p = bp.figure(tools=tools
                  , plot_width=int(w)
                  , plot_height=int(h)
                  , x_range=x_range, y_range=y_range
                  , outline_line_color=None,
        min_border=0, min_border_left=0, min_border_right=0,
        min_border_top=0, min_border_bottom=0,
                 x_axis_type="mercator", y_axis_type="mercator")
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    return p

def plot_filtered(col, fdf, background = 'black'):
    p = base_plot()
    url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{Z}/{Y}/{X}.png"
    #url="http://tile.stamen.com/toner-background/{Z}/{X}/{Y}.png"
    tile_renderer = p.add_tile(WMTSTileSource(url=url))
    tile_renderer.alpha=1.0 if background == "black" else 0.15
    return InteractiveImage(p, create_image_wrap(col, fdf))
