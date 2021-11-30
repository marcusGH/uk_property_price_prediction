from .config import *

from matplotlib.colors import LogNorm

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
import seaborn as sbn
import warnings

"""These are the types of import we might expect in this file
import pandas
import bokeh
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

warnings.filterwarnings("ignore", message= ".*Geometry is in a geographic CRS.*")

"""Place commands in this file to assess the data you have downloaded. How are
missing values encoded, how are outliers encoded? What do columns represent,
makes rure they are correctly labeled. How is the data indexed. Crete
visualisation routines to assess the data (e.g. in bokeh). Ensure that date
formats are correct and correctly timezoned."""

def plot_geographical_heatmap(ax, gdf, val, bins, transform='mean', useLog=False):
  if 'geometry' not in gdf and ('longitude' not in gdf or 'latitude' not in gdf):
    raise Exception('The provided dataframe needs some column indicating positions')

  gdf_copy = gpd.GeoDataFrame(gdf.copy(deep=True))

  if 'longitude' not in gdf_copy or 'latitude' not in gdf_copy:
    gdf_copy['longitude'] = gdf_copy.centroid.map(lambda p : p.x)
    gdf_copy['latitude'] = gdf_copy.centroid.map(lambda p : p.y)


  bin_size_x = (np.max(gdf_copy.longitude) - np.min(gdf_copy.longitude)) / bins
  bin_size_y = (np.max(gdf_copy.latitude) - np.min(gdf_copy.latitude))  / bins

  # assign quantile ID, and then scale back up to get accurate ticks
  gdf_copy['bin_x'] =  (((gdf_copy['longitude']) / bin_size_x).astype(int).astype(float) * float(bin_size_x)).round(decimals=2)
  gdf_copy['bin_y'] =  (((gdf_copy['latitude']) / bin_size_y).astype(int).astype(float) * float(bin_size_y)).round(decimals=2)

  # mean value of specified column
  gdf_copy['bin_value'] = gdf_copy.groupby(['bin_x', 'bin_y'])[val].transform(transform)

  piv = gdf_copy.drop_duplicates(subset=['bin_x', 'bin_y']).pivot(index='bin_y', columns='bin_x', values='bin_value')
  if useLog:
    sbn.heatmap(piv, ax=ax, fmt='g', cmap='viridis', norm=LogNorm())
  else:
    sbn.heatmap(piv, ax=ax, fmt='g', cmap='viridis')

  ax.set_xlabel("Longitude")
  ax.set_ylabel("Latitude")


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError
