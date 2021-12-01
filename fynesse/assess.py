from .config import *

from matplotlib.colors import LogNorm
from shapely.geometry import Polygon, LineString, Point
from fynesse.access import km_to_crs, crs_to_km
from fynesse import access

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

"""Place commands in this file to assess the data you have downloaded. How are
missing values encoded, how are outliers encoded? What do columns represent,
makes rure they are correctly labeled. How is the data indexed. Crete
visualisation routines to assess the data (e.g. in bokeh). Ensure that date
formats are correct and correctly timezoned."""

"""
############################# Plotting utilities ###########################
"""

def plot_housing_density_against_price_and_amt_data(gdf, loc, osm_accommodation_types):
  lat, lon = access.config[loc]

  pois = access.get_pois_around_point(lon, lat, 10, [], { "building": osm_accommodation_types})
  print(f"Found {len(pois)} accommodations in {loc}")

  fig, axs = plt.subplots(ncols=3,figsize=(30, 10))

  fig.suptitle(f"Log-plot of housing price, houses sold, and accommodation density in {loc}", fontsize=24)

  axs[0].set_title(f"House price", fontsize=16)
  plot_geographical_heatmap(axs[0], gdf,
                            'price', 50, useLog=True, transform='mean')

  axs[1].set_title(f"Houses sold in {loc}", fontsize=16)
  # value field doesn't matter when 'count' is used
  plot_geographical_heatmap(axs[1], gdf,
                            'postcode', 50, useLog=True, transform='count')

  axs[2].set_title(f"Number of OSM accomoodations in {loc}", fontsize=16)
  plot_geographical_heatmap(axs[2], pois, 'building', 50, 'count', useLog=True)

  plt.show()

def geoplot(title, figsize=(12,12)):
  fig, ax = plt.subplots(figsize=figsize)
  ax.set_xlabel("Longitude")
  ax.set_ylabel("Latitude")
  ax.set_title(title, fontsize=22)
  return fig, ax

def plot_geographical_heatmap(ax, gdf, val, bins, transform='mean', useLog=False):
  if 'geometry' not in gdf and ('longitude' not in gdf or 'latitude' not in gdf):
    raise Exception('The provided dataframe needs some column indicating positions')

  gdf_copy = gpd.GeoDataFrame(gdf.copy(deep=True))

  if 'longitude' not in gdf_copy or 'latitude' not in gdf_copy:
    warnings.filterwarnings("ignore", message= ".*Geometry is in a geographic CRS.*")
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

def plot_pois_around_area(longitude, latitude, dist_in_km, figName, keys):
  fig, ax = plt.subplots(figsize=(10,10))
  # get POIs from OSM
  pois_keys = { k : v for (k, (_, v, _)) in keys.items() }
  pois = access.get_pois_around_point(longitude, latitude, dist_in_km, keys=pois_keys)
  # plot them using specified colours
  for i, k in enumerate(pois.drop(columns=['geometry']).columns):
    pois[pois[k].notna()].plot(ax=ax, color=keys[k][0], alpha=keys[k][2])

  err = access.km_to_crs(dist_in_km)
  ax.set_xlim([longitude-err, longitude+err])
  ax.set_ylim([latitude-err, latitude+err])
  ax.set_title(f"POIs around {figName}", fontsize=22)
  ax.set_xlabel("Longitude")
  ax.set_ylabel("Latitude")
  plt.tight_layout()

"""
#################################### Sanitisation ################################
"""

def make_geodataframe(df):
  """

  When a geodataframe is saved to a CSV file, the 'geometry'
  and 'date' fields are not recovered properly when importing
  the file into a gpd.GeoDataFrame. This function fixes
  those two columns if they are present.

  Addtionally, a regular pd.DataFrame can be passed and a
  geometry column will be added based on the longitude and
  latitude columns.
  """
  gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
  if 'db_id' in gdf:
    gdf = gdf.set_index('db_id')
  gdf.crs = "EPSG:4326"
  return gdf

def recover_df_from_file(filename, upload_required=True):
  """
  There is some strange things happening when saving
  a gdf as csv and uploading it, so I've abstracted
  away the task of fixing this
  """
  if upload_required:
    files.upload()
  df = pd.read_csv(filename)
  if 'geometry' in df:
    df.drop(columns=['geometry'])
    df = make_geodataframe(df)
  if 'date_of_transfer' in df:
    df['date_of_transfer'] = pd.to_datetime(df['date_of_transfer']).dt.date

  return df

def find_num_missing_values(table_name, columns):
  df = pd.DataFrame(columns=columns)
  num_nans = []
  for c in columns:
    num_nans.append(access.db_select(f"SELECT count(*) AS count FROM {table_name} WHERE {c} IS NULL OR {c} = ''")['count'].iloc[0])
  df.loc[0] = num_nans
  return df

"""
################################### Dataframe aggregation functions ###########################
"""

def highlight_topn(n):
  return lambda s, props = '': np.where((s != 1) &
        (np.abs(s) >= np.partition(np.abs(s.values).flatten(), -n-1)[-n-1]), props, '')

def _add_spatial_aggregate_column(gdf, pois, dist_in_km, col_name, agg_f, nafill=np.nan, cacheBy='postcode'):
  """
  Adds a new column to GeoDataFrame `gdf` and returns this modified GeoDataFrame.
  The value $v$ in row $i$ for this new column is determined as follows:
  * All the POIs in pois with amenity key equal to `amenity_type`, that are
    closer than `dist_in_km` to entry $i$ in `gdf`, are put in a dataframe
  * The value $v$ is the result of calling provided aggregation function `agg_f`
    on this dataframe

  :param gdf GeoDataFrame
  :param pois GeoDataFrame
  :param dist_in_km positive number
  :param amenity_type string that is a valid amenity key in OSM
  :param col_name string
  :param agg_f a lambda function that takes a dataframe and does some aggregation
               that produces a single value
  :param nafill some single value
  """
  # earth_circum = 40075
  # dist = (dist_in_km / earth_circum) * 360
  dist = km_to_crs(dist_in_km)

  # make a copy of the passed dataframes, so that the arguments
  #   are not modified when we do spatial joins later
  gdf_copy = gpd.GeoDataFrame(gdf.copy(deep=True))
  pois_copy = gpd.GeoDataFrame(pois).copy(deep=True)
  pois_copy.crs = 'EPSG:4326'
  # we have a copy of the geometry such that we look at it after sjoin
  pois_copy['geometry_copy'] = pois_copy['geometry']

  # add a approximated circular buffer around each point in gdf
  gdf_copy['geometry'] = gdf_copy.geometry.buffer(dist)

  # count pois within the buffer of each point
  join_left_df = gdf_copy.sjoin(pois_copy, predicate='contains', how='left')
  # We have a group with each price_paid row, so aggregate after regrouping
  #   we also cache results by the postcode because that gives the longitude/latitude info
  agg_f_cache = {}
  def agg_f_with_cache(x):
    k = None
    if 'postcode' not in x:
      k = str(x['postcode_left'].iloc[0])
    else:
      k = str(x['postcode'].iloc[0])
    if k not in agg_f_cache:
      agg_f_cache[k] = agg_f(x)
    return agg_f_cache[k]
  # We do very short distances, so curvature of the earth is unlikely
  #  to affect accuracy
  warnings.filterwarnings("ignore", message= ".*Geometry is in a geographic CRS.*")
  nums_series = join_left_df.groupby('db_id').apply(agg_f_with_cache).fillna(nafill)

  # Merge on the index, which is db_id
  result_gdf = pd.merge(gdf_copy,
                  pd.DataFrame(nums_series, columns=[col_name]),
                  left_index=True, right_index=True)
  # change the geometry column back
  result_gdf['geometry'] = (gdf['geometry']).copy(deep=True)
  return result_gdf

def get_num_pois_within_radius(gdf, pois, dist_in_km, col_name):
  """
  Takes a GeoDataFrame gdf and a GeoDataFrame of POIs and adds a new
  column to the provided gdf that contains the number of POIs located
  within a dist_in_km radius of the gdf entry.
  """
  # we can't do len(x) because when there is no POI, we will
  #  still have one entry because we do LEFT_JOINs, so we count
  #  number of non-NaNs entries in a column that definitely exists
  #  in our POIs, `geometry_copy`
  agg_f = lambda x: x.geometry_copy.notna().sum()
  return _add_spatial_aggregate_column(
      gdf, pois, dist_in_km, col_name, agg_f
    )

def get_closest_poi_within_radius(gdf, pois, dist_in_km, col_name):
  """
  Takes a GeoDataFrame gdf and a GeoDataFrame of POIs and adds a new
  column to the provided gdf that contains the distance in km
  to the cloest POIs located within a dist_in_km radius of the gdf entry.

  In case there are no POI within `dist_in_km`, the
  distance is set to `dist_in_km`, which is a lower bound on the true
  distance to nearest POI.
  """
  agg_f = lambda x: crs_to_km(x.set_geometry('geometry_copy')
                     # all entries have the same longitude and latitude info
                     .distance(Point(x.longitude.iloc[0], x.latitude.iloc[0]))
                     .dropna()
                     .min())
  return _add_spatial_aggregate_column(
      gdf, pois, dist_in_km, col_name, agg_f, dist_in_km
    )

def get_average_dist_to_poi_within_radius(gdf, pois, dist_in_km, col_name):
  """
  Takes a GeoDataFrame gdf and a GeoDataFrame of POIs and adds a new
  column to the provided gdf that contains the average distance in km
  to the POIs located within a dist_in_km radius of the gdf entry.

  In case there are no POI within `dist_in_km`, the
  distance is set to `dist_in_km`, which is a lower bound on the true
  distance to nearest POI.
  """
  agg_f = lambda x: crs_to_km(x.set_geometry('geometry_copy')
                     # all entries have the same longitude and latitude info
                     .distance(Point(x.longitude.iloc[0], x.latitude.iloc[0]))
                     .dropna()
                     .mean())
  return _add_spatial_aggregate_column(
      gdf, pois, dist_in_km, col_name, agg_f, dist_in_km
    )

# TODO: remove below functions

# def data():
#     """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
#     df = access.data()
#     raise NotImplementedError
#
# def query(data):
#     """Request user input for some aspect of the data."""
#     raise NotImplementedError
#
# def view(data):
#     """Provide a view of the data that allows the user to verify some aspect of its quality."""
#     raise NotImplementedError
#
# def labelled(data):
#     """Provide a labelled set of data ready for supervised learning."""
#     raise NotImplementedError

# vim: set shiftwidth=2 :
