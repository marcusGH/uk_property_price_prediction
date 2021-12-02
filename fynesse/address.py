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


from fynesse import access
from fynesse import assess
from fynesse.access import km_to_crs, crs_to_km, flatten
from shapely.geometry import Polygon, LineString, Point
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics as skm
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore", message= ".*Geometry is in a geographic CRS.*")

def add_feature_from_pois(gdf, pois, larger_gdf=None, **feature_kwargs):
  """
  Returns the geodataframe gdf, but with a new column with values for the feature
  specified in the feature_kwargs dictionary, which should contain entries

      func : name indicating aggregation function to calculate the feature's value
      pois_cond : lambda taking a pois and extracting the relevant POIs entries
      dist : number of kilometers to search for POIs
      name : name of the feature, and what to name the column

  The pois geodataframe should contain POIs from OSM where all the columns mentioned in
  pois_cond exists in pois. The POIs in `pois` should also have been fetched from an area
  around all the entries in `gdf`.
  """
  augmented_gdf = None
  if feature_kwargs['func'] == 'closest':
    augmented_gdf = assess.get_closest_poi_within_radius(gdf, feature_kwargs['pois_cond'](pois), feature_kwargs['dist'], feature_kwargs['name'])
  elif feature_kwargs['func'] == 'count':
    augmented_gdf = assess.get_num_pois_within_radius(gdf, feature_kwargs['pois_cond'](pois), feature_kwargs['dist'], feature_kwargs['name'])
  elif feature_kwargs['func'] == 'avg_dist':
    augmented_gdf = assess.get_average_dist_to_poi_within_radius(gdf, feature_kwargs['pois_cond'](pois), feature_kwargs['dist'], feature_kwargs['name'])
  elif feature_kwargs['func'] == 'num_houses':
    if larger_gdf is None:
      raise Exception("Must specify larger_gdf if using features 'num_houses'")
    augmented_gdf = assess.get_num_pois_within_radius(gdf, larger_gdf, feature_kwargs['dist'], feature_kwargs['name'])
  else:
    raise NotImplemented(f"Function {feature_kwargs['func']} is not implemented.")
  return augmented_gdf

def add_many_features_from_pois(gdf, pois, larger_gdf=None, **feature_kwargs):
  """
  A generalised version of add_feature_from_pois. The only difference is that
  the 'func' and 'name' keys in the feature specification may contain a list
  of functions to apply, e.g. 'func' : ['count', 'avg_dist']
  """
  if isinstance(feature_kwargs['func'], str):
    assert isinstance(feature_kwargs['name'], str), "If func is singular, name must be as well"
    return add_feature_from_pois(gdf, pois, larger_gdf, **feature_kwargs)
  else:
    # run the single function on all the provided funcs
    for func, name in zip(feature_kwargs['func'], feature_kwargs['name']):
      gdf = add_feature_from_pois(gdf, pois, larger_gdf=larger_gdf, **{
          'func': func,
          'pois_cond' : feature_kwargs['pois_cond'],
          'dist' : feature_kwargs['dist'],
          'name' : name
      })
    return gdf

def build_prices_coordinates_features_dataset(latitude, longitude, date,
                                              property_type, bb_size_km,
                                              pois_bb_size_km, year_range_size,
                                              pois_keys, features,
                                              logging=False):
  """
  Returns a pandas dataframe ... TODO TODO

  :param latitude, longitude   geoposition around which to build dataset
  :param date                  integer specifying year around which data is fetched
  :param property_type         a valid property type in pp_data
  :param bb_size_km            size of the sides of the geographical bounding box in
                               which prices_coordinates data is fetched
  :param pois_bb_size_km       size of the side of the bounding box in which
                               which OSM data is fetched
  :param year_range_size       temporal bounding box, specified as integer indicating number of years
  :param pois_keys             OSMNX POIs keys to use when fetching POIs. These keys must be a superset
                               of the keys mentioned in any of the features
  :param features              A list of feature specifications @see add_feature_from_pois
  """

  if logging:
    print((f"Building prices coordinates with features dataset at point"
      f" ({longitude}, {latitude}) with bounding box size {bb_size_km}km"
      f" with POIs bounding box size {pois_bb_size_km}km "
      f"with features: {', '.join(flatten([f['name'] for f in features]))}"))

  bb_half_size = km_to_crs(bb_size_km) / 2
  year_diff    = int(year_range_size / 2)

  # Fetch prices_coordinate rows and
  #   cache SQL query for location and date range, so can be reused later
  cache_key_sql = ("prices_coordinates", latitude, longitude, date, bb_size_km, year_range_size)
  if cache_key_sql not in build_prices_coordinates_features_dataset.cache:
    if logging:
      print(f"The cache key {cache_key_sql} is not in cache, running SQL query...")
    build_prices_coordinates_features_dataset.cache[cache_key_sql] = assess.make_geodataframe(
      access.prices_coordinates_range_query_fast(longitude-bb_half_size,
                                      longitude+bb_half_size,
                                      latitude-bb_half_size,
                                      latitude+bb_half_size,
                                      [date-year_diff, date+year_diff])
      )
  elif logging:
    print(f"The cache key {cache_key_sql} is in cache, skipping SQL query...")
  prices_coordinates_gdf = build_prices_coordinates_features_dataset.cache[cache_key_sql]

  # filter out properties of different types, unless that would give too little data
  if len(prices_coordinates_gdf[prices_coordinates_gdf['property_type'] == property_type]) > 100:
    print(f"Found enough properties of type {property_type}. Filtering out other types for predictions...")
    prices_coordinates_gdf = prices_coordinates_gdf[prices_coordinates_gdf['property_type'] == property_type]
    print(f"Which leaves {len(prices_coordinates_gdf)} rows of property type {property_type}")
  else:
    print("Did not find enough properties of appropriate type. Using all types...")
    print("WARNING: This will likely cause the prediction to be very inaccurate!")

  # if no data is found, we must fail
  if len(prices_coordinates_gdf) == 0:
    return None, None, None

  if logging:
    print(f"Using {len(prices_coordinates_gdf)} prices_coordinate rows")

  # Fetch prices_coordinate rows that is used for num_houses feature and
  #   cache SQL query for location and date range, so can be reused later
  houses_dist = 0
  for f in features:
    if f['func'] == 'num_houses':
      houses_dist = max(f['dist'], houses_dist)
    elif f['func'] == ['num_houses']:
      houses_dist = max(f['dist'], houses_dist)
  larger_prices_gdf = None
  # we look at number of houses sold within some dist of houses in our dataset,
  #   so we need an extra dataset of houses sold which cover a larger region
  if houses_dist > 0:
    # Fetch prices_coordinates rows for larger range and put result in larger_prices_gdf, and
    #   use caching to speed up computation time
    cache_key_sql = ("prices_coordinates", latitude, longitude, date, bb_size_km + houses_dist, year_range_size)
    bb_larger_half_size = km_to_crs(bb_size_km + houses_dist)
    if cache_key_sql not in build_prices_coordinates_features_dataset.cache:
      if logging:
        print(f"The cache key {cache_key_sql} is not in cache, running SQL query...")
      build_prices_coordinates_features_dataset.cache[cache_key_sql] = assess.make_geodataframe(
          access.prices_coordinates_range_query_fast(longitude-bb_larger_half_size,
            longitude+bb_larger_half_size,
            latitude-bb_larger_half_size,
            latitude+bb_larger_half_size,
            [date-year_diff, date+year_diff])
          )
    elif logging:
      print(f"The cache key {cache_key_sql} is in cache, skipping SQL query...")
    larger_prices_gdf = build_prices_coordinates_features_dataset.cache[cache_key_sql]
    if logging:
      print(f"Found {len(larger_prices_gdf)} prices_coordinate rows of any property type for larger area")

  # fetch the POIs around the bounding box
  cache_key_osm = ("OSM", latitude, longitude, pois_bb_size_km)
  if cache_key_osm not in build_prices_coordinates_features_dataset.cache:
    if logging:
      print(f"The cache key {cache_key_osm} is not in cache, querying OSM...")
    build_prices_coordinates_features_dataset.cache[cache_key_osm] = \
      access.get_pois_around_point(longitude, latitude, pois_bb_size_km / 2, keys=pois_keys)
  else:
    print(f"The cache key {cache_key_osm} is in cache, skipping OSM query...")
  pois = build_prices_coordinates_features_dataset.cache[cache_key_osm]

  # The osmx API will not add column for specified key if it doesn't find any
  #   entities of that type, so we need to add it explicitly and fill it with NaNs
  #   to avoid ValueErrors later down the line when adding features
  for k in pois_keys:
    if k not in pois.columns:
      pois[k] = np.nan
  # print(pois.to_string()[0:10000000])

  # fail if nothing found
  if len(pois) == 0:
    return None, None, None

  if logging:
    print(f"Found {len(pois)} POIs in the area")

  # add the features
  for f in features:
    if (f['func'] == 'num_houses' or f['func'] == ['num_houses']) and len(prices_coordinates_gdf) > 1800:
      print(f"There was a lot of data entries found ({len(prices_coordinates_gdf)}). "
        + "Considering removing the feature num_houses to speed up computation")

    # Add a new column with name f['name'] using the values computed for the feature
    # prices_coordinates_gdf = prices_coordinates_gdf.assign(**{f['name']: build_prices_coordinates_features_dataset.cache[cache_key_f].to_numpy()})
    prices_coordinates_gdf = add_many_features_from_pois(prices_coordinates_gdf, pois, larger_prices_gdf, **f)
    print(f"Added feature {f['name']} to dataset")

  # return
  return prices_coordinates_gdf, pois, larger_prices_gdf
build_prices_coordinates_features_dataset.cache = {}

"""
##################### predict_price #################################
"""


def predict_price(latitude, longitude, date, property_type, build_dataset_kwargs, design, printSummary=False, plotAx=None):
  """
  The date is in years
  axis can be specified to plotFit
  """

  # add some missing keys
  build_dataset_kwargs['date'] = date # year of date
  build_dataset_kwargs['property_type'] = property_type

  # create dataset with columns for all the features
  dataset_gdf, pois, gdf_for_num_houses = build_prices_coordinates_features_dataset(
      latitude, longitude, logging=True, **build_dataset_kwargs)
  
  if dataset_gdf is None:
    print(f"No UK properties"
     + f"were found within a {build_dataset_kwargs['bb_size_km']/2} km radius of ({latitude}, {longitude})")
    return None, None

  if len(dataset_gdf) < 100:
    print(f"Only {len(dataset_gdf)} entires were found.")
    print("No accruate price predictions can be made with fewer than 100 entries.")
    print("Try again with a larger bounding box for location or year")
    return None, None

  print(f"Size of dataset is {len(dataset_gdf)}. Example rows:")
  print(dataset_gdf.drop_duplicates(subset=['postcode']).head().to_string()[0:100000])

  # split
  training_data, testing_data = train_test_split(dataset_gdf, test_size=0.2)

  # create the linear model
  basis = sm.GLM(training_data['price'], design(training_data), family=sm.families.NegativeBinomial())
  results = basis.fit()

  # run predictions on test set to calculate various metrics
  predicts = results.predict(design(testing_data))
  mae = skm.mean_absolute_error(testing_data['price'].to_numpy(), predicts)
  mse = np.sqrt(skm.mean_squared_error(testing_data['price'].to_numpy(), predicts))
  mape = skm.mean_absolute_percentage_error(testing_data['price'].to_numpy(), predicts)

  if printSummary:
    print(results.summary())
    print("=" * 100)
    print(f"Average housing price in training data: {int(np.mean(training_data['price']))} ± {int(np.std(training_data['price']))}σ")
    print(f"Average predicted house price: {int(np.mean(predicts))} ± {int(np.std(predicts))}σ")
    print(f"MAE for price predictions on testing data set is: {int(mae)}")
    print(f"sqrt(MSE) for price predictions on testing data set is: {int(mse)}")
    print(f"Mean absolute percentage error for price predictions on testing data set is: {mape:.4}")
    print("=" * 100)

  if plotAx is not None:
    plotAx.scatter(testing_data['price'].to_numpy(), predicts)
    xs = np.linspace(0,max(testing_data['price'].to_numpy()), 100)
    plotAx.plot(xs, xs, alpha=0.4, color='green')
    plotAx.set_xlabel("True price")
    plotAx.set_ylabel("Predicted price")
    plotAx.set_title(f"True vs. predicted price on held-out training data (n={len(predicts)})")


  # Now run the prediction on the provided data row

  # create the gdf with a single row
  predict_df = pd.DataFrame(columns=['db_id', 'date_of_transfer', 'postcode', 'property_type', 'longitude', 'latitude']).set_index('db_id')
  predict_df.loc[0] = [f'{date}-01-01', 'PLACEHOLDER POSTCODE', property_type, longitude, latitude]
  predict_gdf = assess.make_geodataframe(predict_df)
  # add the features
  for f in build_dataset_kwargs['features']:
    predict_gdf = add_many_features_from_pois(predict_gdf, pois, gdf_for_num_houses, **f)

  print("Feature values:")
  f_names = flatten([f['name'] for f in build_dataset_kwargs['features']])
  print(f_names)
  print(predict_gdf[flatten([f['name'] for f in build_dataset_kwargs['features']])].to_markdown())
  print("Feature values after applying design matrix:")
  print(design(predict_gdf))

  pred_val = results.predict(design(predict_gdf))[0]

  return pred_val, pred_val * mape

"""
###################################### Evaluations ###########################
"""

def do_pca(design_matrix_fun, data_gdf, features):

  X = design_matrix_fun(data_gdf, features)

  print(f"Shape of design matrix applied to features: {X.shape}")

  # normalize by column, such that variance is 1 and mean is 0
  X = (X - X.mean(axis=0)) / X.std(axis=0)

  # We now do PCA

  pca = PCA()
  pca.fit(X)

  # Transform
  transformed_pca = pca.transform(X)

  # Print how much each feature contribute
  for i, _ in enumerate(features):
    var_ex = pca.explained_variance_ratio_[i]
    var_name = pca.components_[i]
    print(f"Feature {var_name} explains {var_ex:.4} of the variance")

# vim: set shiftwidth=2 :
