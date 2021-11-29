# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

from sklearn.model_selection import train_test_split
from fynesse.access import km_to_crs, crs_to_km, make_geodataframe

import statsmodels.api as sm
import sklearn.metrics as skm
import pandas as pd
import geopandas as gpd
import numpy as np
import warnings

warnings.filterwarnings("ignore", message= ".*Geometry is in a geographic CRS.*")

"""Address a particular question that arises from the data"""

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
  # nums_series = join_left_df.groupby('db_id').apply(agg_f).fillna(nafill)
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

def add_feature_from_pois(gdf, pois, **feature_kwargs):
  augmented_gdf = None
  if feature_kwargs['func'] == 'closest':
    augmented_gdf = get_closest_poi_within_radius(gdf, feature_kwargs['pois_cond'](pois), feature_kwargs['dist'], feature_kwargs['name'])
  elif feature_kwargs['func'] == 'count':
    augmented_gdf = get_num_pois_within_radius(gdf, feature_kwargs['pois_cond'](pois), feature_kwargs['dist'], feature_kwargs['name'])
  elif feature_kwargs['func'] == 'avg_dist':
    augmented_gdf = get_average_dist_to_poi_within_radius(gdf, feature_kwargs['pois_cond'](pois), feature_kwargs['dist'], feature_kwargs['name'])
  else:
    raise NotImplemented(f"Function {feature_kwargs['func']} is not implemented.")
  return augmented_gdf[feature_kwargs['name']]

def build_prices_coordinates_features_dataset(latitude, longitude, date,
                                              property_type, bb_size_km,
                                              pois_bb_size_km, year_range_size,
                                              pois_keys, features, 
                                              logging=False):
  """
  TODO: docs
  """

  if logging:
    print((f"Building prices coordinates with features dataset at point"
      f" ({longitude}, {latitude}) with bounding box size {bb_size_km}km"
      f" with POIs bounding box size {pois_bb_size_km}km "
      f"with features: {', '.join([f['name'] for f in features])}"))

  bb_half_size = km_to_crs(bb_size_km) / 2
  year_diff    = int(year_range_size / 2)

  # Fetch prices_coordinate rows and 
  #   cache SQL query for location and date range, so can be reused later
  cache_key_sql = (latitude, longitude, date, bb_size_km, year_range_size)
  if cache_key_sql not in build_prices_coordinates_features_dataset.cache:
    if logging:
      print(f"The cache key ({cache_key_sql}) is not in cache, running SQL query...")
    build_prices_coordinates_features_dataset.cache[cache_key_sql] = make_geodataframe(
      prices_coordinates_range_query_fast(longitude-bb_half_size,
                                      longitude+bb_half_size,
                                      latitude-bb_half_size,
                                      latitude+bb_half_size,
                                      [date-year_diff, date+year_diff])
      )
  prices_coordinates_gdf = build_prices_coordinates_features_dataset.cache[cache_key_sql]
  if logging:
    print(f"Found {len(prices_coordinates_gdf)} prices_coordinate rows")

  # fetch the POIs around the bounding box
  cache_key_osm = (latitude, longitude, pois_bb_size_km)
  if cache_key_osm not in build_prices_coordinates_features_dataset.cache:
    if logging:
      print(f"The cache key ({cache_key_osm}) is not in cache, querying OSM...")
    build_prices_coordinates_features_dataset.cache[cache_key_osm] = get_pois_around_point(longitude, latitude, pois_bb_size_km / 2, [], pois_keys)
  pois = build_prices_coordinates_features_dataset.cache[cache_key_osm]
  if logging:
    print(f"Found {len(pois)} POIs in the area")
  
  # add the features
  for f in features:
    cache_key_f = (cache_key_sql, cache_key_osm, f.values())
    if cache_key_f not in build_prices_coordinates_features_dataset.cache:
      build_prices_coordinates_features_dataset.cache[cache_key_f] = add_feature_from_pois(prices_coordinates_gdf, pois, **f)
    prices_coordinates_gdf = prices_coordinates_gdf.assign(**{f['name']: build_prices_coordinates_features_dataset.cache[cache_key_f].to_numpy()})
    print(f"Added feature {f['name']} to dataset")

  # return
  return prices_coordinates_gdf, pois
build_prices_coordinates_features_dataset.cache = {}
