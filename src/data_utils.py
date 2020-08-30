import numpy as np
import pandas as pd
import pickle

from torch.utils.data import Dataset

#requirements for random pick of users
import random


#######################################
#PLOT
#requrements for compute_roc() and other plots
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
#######################################
#PLOT ON MAPS
#requirements for plot_filtered()
from datashader.bokeh_ext import InteractiveImage
import bokeh.plotting as bp
import datashader.transfer_functions as dtf
from colorcet import fire, glasbey_warm, glasbey_dark, glasbey_light, bkr
import datashader as ds
import holoviews as hv
import geoviews as gv
from holoviews.operation.datashader import datashade
hv.extension('bokeh')
import bokeh.plotting as bp
from bokeh.models.tiles import WMTSTileSource

#SKLEARN
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_curve, auc

scaler = MinMaxScaler()
#scaler = StandardScaler()

def scale_by_ID(df, ixx):
    """
    """
    #ixxStd = [xx+'std' for xx in ixx]
    #df[ixxStd]=0
    for ID in df['user'].unique():
        #df.loc[ df['user'][(df['user']==ID)].index, (ixxStd)] = scaler.fit_transform(df[ixx][(df['user']==ID)])
        df.loc[ df['user'][(df['user']==ID)].index, (ixx)] = scaler.fit_transform(df[ixx][(df['user']==ID)])
        
    return df

def load_user_data(user, data_directory = '/mnt/array/valse_data/DeepLearning/Project/Pickle', load_web_mercator = False, load_GPS=False, load_Dummies=False):
    """
    """
    with open(f'{data_directory}/TS_{user}.pickle', 'rb') as f:
        ts = pd.to_datetime(pickle.load(f))      
    
    with open(f'{data_directory}/label_{user}.pickle', 'rb') as f:
        labels = np.array(pickle.load(f)).astype(int)
   
    with open(f'{data_directory}/pos_lon_lat_UTM_{user}.pickle', 'rb') as f:
        pos_utm = np.array(pickle.load(f))
    
    if load_GPS:
        with open(f'{data_directory}/pos_lon_lat_{user}.pickle', 'rb') as f:
            pos_GPS = np.array(pickle.load(f))
    else:
        pos_GPS = None

    if load_web_mercator:
        with open(f'{data_directory}/webM_user_{user}.pickle', 'rb') as f:
            pos_webm = np.array(pickle.load(f))
    else:
        pos_webm = None
    
    if load_Dummies:
        with open(f'{data_directory}/dummies_array_{user}.pickle', 'rb') as f:
            dummy_array = pickle.load(f)
    else:
        dummy_array = None
    
                            
    return user, ts, pos_utm, pos_webm, pos_GPS, dummy_array, labels

def create_dummies_pickle_from_images(user, data_directory = '/mnt/array/valse_data/DeepLearning/Project/Pickle'):
    """
    """
    with open(f'{data_directory}/images_list_{user}.pickle', 'rb') as f:
        image_data = np.stack(pickle.load(f), axis = 0).astype(float)
    
    dummy = np.stack([np.stack([image_data[j,:,:k].sum() for k in range(0,image_data.shape[3])], axis = 0 ) for j in range(0,image_data.shape[0])], axis = 0 )
    dummy = np.where(dummy > 0, 1, 0)
    with open(f'{data_directory}/dummies_array_{user}.pickle', 'wb') as f:
        dummy.dump(f)
    #print(dummy.shape)

def create_data_frame(user, ts, pos_utm, pos_webm, pos_GPS, dummy_array, labels, segmentation = False, seq_cutoff_time = 300, seq_cutoff_speed = 42, Standardize = False):
    """
    """
    df = pd.DataFrame({
        'user': user,
        'ts': ts,
        'image_ix': np.arange(0, ts.shape[0]),
        'x': pos_utm[:, 0],
        'y': pos_utm[:, 1],
        'label': labels
    })
    
    if dummy_array is not None:
        
        df['f_highway_motorway'] = dummy_array[:,0]
        df['f_traffic_signals'] = dummy_array[:,1]
        df['f_bus_stops'] = dummy_array[:,2]
        df['f_landuse_meadow'] = dummy_array[:,3]
        df['f_landuse_residential'] = dummy_array[:,4]
        df['f_landuse_industrial'] = dummy_array[:,5]
        df['f_landuse_commercial'] = dummy_array[:,6]
        df['f_shop'] = dummy_array[:,7]
        df['f_railways'] = dummy_array[:,8]
        df['f_railways_station'] = dummy_array[:,9]
        df['f_subway'] = dummy_array[:,10]
        
    if pos_webm is not None:
        df['x_web'] = pos_webm[:, 0]
        df['y_web'] = pos_webm[:, 1]
        
    if pos_GPS is not None:
        df['lon'] = pos_GPS[:, 0]
        df['lat']= pos_GPS[:, 1]
        
    df.sort_values('ts', inplace = True)
    
    df['delta_t'] = np.concatenate([[0], (df['ts'].values[1:] - df['ts'].values[:-1]) / pd.to_timedelta('1s')], axis = 0)    
    df['delta_d'] = np.concatenate([[0], np.linalg.norm(df[['x', 'y']].values[1:] - df[['x', 'y']].values[:-1], axis = 1)], axis = 0)    
    df['bearing'] = np.concatenate([[0], np.arctan2(df[['y']].values[1:] - df[['y']].values[:-1], df[['x']].values[1:] - df[['x']].values[:-1]).reshape(-1)], axis = 0)
    df['speed'] = df['delta_d'] / df['delta_t'] 
    
    cut_labels_6 = [0, 1, 2, 3, 4]
    cut_bins = [0, 6, 10, 14, 18, 24]
    df['tod'] = pd.cut(df['ts'].dt.hour, bins=cut_bins, labels=cut_labels_6, right = False).astype(int)
    
    df = df[lambda x: x['delta_t'] > 0].copy()
    
    seq_bins = np.cumsum((df['delta_t'] >= seq_cutoff_time) | (df['speed'] > seq_cutoff_speed))
    seq_bin_ids, seq_bin_counts = np.unique(seq_bins, return_counts=True)
    
    if segmentation:
        df['segment_id'] = seq_bins
        df['segment_ix'] = np.concatenate([np.arange(seq_bin_counts[i]) for i in range(len(seq_bin_counts))])
        df['segment_point_count'] = np.repeat(seq_bin_counts, seq_bin_counts)
    
    if Standardize:
        df = scale_by_ID(df, ['delta_d','bearing','speed'])
    
    return df

def train_test_data_split(u=12, Random = False, k=9):
    """
    """
    if Random:        
        #Pick random users
        users = range(0,u)
        #Split Test/Train according to random pick
        train_val = random.sample(population=users, k=k) 
        train = train_val[:-1]
        val = train_val[-1:]
        test  = [x for x in users if x not in train_val]
    elif k==9:
        #this sequence comes form a random sampling
        #to make it consistent across multiple contributors, and installaitons, 
        #the seed is not enough. Hence we store it as a statc variable.
        train, val, test = ([8, 6, 4, 5, 9, 1, 11, 7], [2], [0, 3, 10])
    
    elif k==10:
        train, val, test = ([3, 1, 7, 5, 4, 9, 8, 10, 11], [2], [0, 6])
    
    return train, val, test


bp.output_notebook()

# Utility visualization functions
def hex_to_rgb(hex):
    """
    """
    hex = hex.lstrip('#')
    hlen = len(hex)
    return tuple(int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))

# Default plot ranges:
def create_image_wrap(fdf, col, w=1000, h=900, x_range = (1373757.1102773394, 1412506.1502695908), y_range = (7478418.9895278225, 7520786.118694777), background = 'black'):
    """
    """
    def create_image(x_range=x_range, y_range=y_range, w=w, h=h):
        """
        """
        cvs = ds.Canvas(x_range=x_range, y_range=y_range, plot_height=h, plot_width=w)
    
        if len(fdf[col].unique())>10:
            colormap = fire
        else:
            colormap = bkr
        agg = cvs.points(fdf, 'x_web', 'y_web', agg=ds.mean(col))
        image = dtf.shade(agg, cmap=colormap)
        ds.utils.export_image(image,filename=col+'.png')
        return dtf.dynspread(image, threshold=0.75, max_px=8)

    return create_image

def base_plot(tools='pan,wheel_zoom,reset', w=1000, h=900, x_range = (1373757.1102773394, 1412506.1502695908), y_range = (7478418.9895278225, 7520786.118694777), background = 'black'):
    """
    """
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

def plot_filtered(col, fdf, background = 'black', Toner=True, x_range = (1373757.1102773394, 1412506.1502695908), y_range = (7478418.9895278225, 7520786.118694777)):
    """
    
    """
    p = base_plot(x_range = x_range, y_range = y_range)
    if Toner:
        url="http://tile.stamen.com/toner-background/{Z}/{X}/{Y}.png"
    else:
        url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{Z}/{Y}/{X}.png"
    tile_renderer = p.add_tile(WMTSTileSource(url=url))
    tile_renderer.alpha=1.0 if background == "black" else 0.15
    return InteractiveImage(p, create_image_wrap(col, fdf, x_range = x_range, y_range = y_range))

def compute_roc(y_true, y_pred, phase_name, plot=False):
    """
    TODO
    :param y_true: ground truth
    :param y_pred: predictions
    :param plot:
    :return:
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title(f"ROC Curve {phase_name}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score 