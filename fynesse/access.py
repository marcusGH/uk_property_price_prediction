from .config import *

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import mongodb
import sqlite"""

"""
 Python module dependencies
"""
from google.colab import files
from ipywidgets import interact_manual, Text, Password
from shapely.geometry import Polygon, LineString, Point
from collections import Iterable

import geopandas as gpd
import datetime
import io
import numpy as np
import osmnx as ox
import pandas as pd
import pymysql
import sys
import traceback
import urllib.request
import yaml
import zipfile
import warnings

warnings.filterwarnings("ignore", message= ".*Geometry is in a geographic CRS.*")

"""
 Place commands in this file to access the data electronically. Don't remove
 any missing values, or deal with outliers. Make sure you have legalities
 correct, both intellectual property and personal data privacy rights. Beyond
 the legal side also think about the ethical issues around this data.
"""

"""
################################# Database set up ############################
"""

# Commented out IPython magic to ensure Python compatibility.

# Database set up and connection utility functions
def get_and_store_credentials():
  @interact_manual(username=Text(description="Username:"),
                  password=Password(description="Password:"))
  def write_credentials(username, password):
      with open("credentials.yaml", "w") as file:
          credentials_dict = {'username': username,
                              'password': password}
          yaml.dump(credentials_dict, file)

def get_database_details():
  @interact_manual(url=Text(description="Database endpoint:"),
                  port=Text(description="Port number:"))
  def write_credentials(url, port):
      with open("database-details.yaml", "w") as file:
          database_details_dict = {'url': url,
                                   'port': port}
          yaml.dump(database_details_dict, file)

def get_database_connection(database, get_connection_url=False):
  """
  If get_connection_url is set to True, returns a string that can
  be used to connect to the MariaDB server using %sql magic.
  Otherwise, a connection object is returned.
  """
  # read stored information
  with open("credentials.yaml") as file:
    credentials = yaml.safe_load(file)
  with open("database-details.yaml") as file:
    database_details = yaml.safe_load(file)
  username = credentials["username"]
  password = credentials["password"]
  url = database_details["url"]
  port = int(database_details["port"])
  # for use if doing %sql mariadb+pymysql://$return_value
  if get_connection_url:
    return f"{username}:{password}@{url}?local_infile=1"
  else:
    conn = None
    try:
      conn = pymysql.connect(user=username,
                        passwd=password,
                        host=url,
                        port=port,
                        local_infile=1,
                        db=database
                        )
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn

def create_database(db_name):
  # first connect to our MariaDB server
  connect_string = get_database_connection(db_name, True)

  print("See the source code. I couldn't get %sql and %load_ext sql to work in my package...")
  # %sql mariadb+pymysql://$connect_string
  # %sql SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
  # %sql SET time_zone = "+00:00";
  #
  # %sql CREATE DATABASE IF NOT EXISTS `$db_name` DEFAULT CHARACTER SET utf8 COLLATE utf8_bin;
  #
  # %sql USE `$db_name`

"""
################################# Database queries ############################
"""

def db_query(query, database="property_prices"):
  """
  Runs specified query on specified database, and commits the changes if the
  query caused any changes to the database. Nothing is returned, so this
  function is more making changes to the tables.
  """
  # conn = get_database_connection("property_prices")
  conn = get_database_connection(database)
  try:
    cur = conn.cursor()
    cur.execute(query)
    conn.commit()
    conn.close()
  except:
    conn.close()
    print("Received interruption and successfully closed the connection")

def db_select(query, database="property_prices", n=None, one_every=None):
  """
  Perform specified SQL query and return the result as a pandas dataframe
  with appropriate column names.
  
  :param query: The SQL query
  :param database: The database to do the query on
  :param n: number of rows (optional)
  :param one_every: only fetch one row of every one_every rows returned
  """
  conn = get_database_connection(database)
  cur = conn.cursor()
  if n is not None:
    query = f"{query} LIMIT {n}"

  try:
    cur.execute(query)
    # This cool line extracts the column names from the result
    #   of the most recent query executed
    column_names = list(map(lambda x: x[0], cur.description))
    result_sql = cur.fetchall()
    conn.close()
    # convert result to pandas dataframe
    df = pd.DataFrame(columns=column_names, data=result_sql)
    if 'db_id' in column_names:
      try:
        return df.set_index('db_id')
      except ValueError as e:
        print(f"Could not set index to 'db_id'\n:{traceback.format_exc()}")
        return df
    else:
      return df
  except KeyboardInterrupt:
    conn.close()
    print("Received interruption and successfully closed the connection")
  # This does not include KeyboardInterrupt
  except Exception as e:
    conn.close()
    print(traceback.format_exc())
    print(f"Tried to do query:")
    print(query)

def inner_join_sql_query(minYear=None,maxYear=None,minLatitude=None,
                         maxLatitude=None,minLongitude=None,maxLongitude=None,
                         oneEvery=None):
  """
  Returns a string of an sql query that joins the `pp_data` and `postcode_data`
  on the fly, selecting only those rows within the specified spatial and temporal
  subrange, as specified by the first 6 arguments.
  :param oneEvery: integer i specifying that only every ith row should
                   be returned (optional)
  """
  query = """
    SELECT
      -- Columns from pp_data
      pp.county,
      pp.date_of_transfer,
      pp.db_id,
      pp.district,
      pp.locality,
      pp.new_build_flag,
      pp.postcode,
      pp.price,
      pp.property_type,
      pp.tenure_type,
      pp.town_city,
      -- Columns from postcode data
      pd.lattitude AS latitude,
      pd.longitude,
      pd.country
    FROM
      postcode_data AS pd
    INNER JOIN
      pp_data as pp
    ON
      pp.postcode = pd.postcode
    WHERE
      TRUE
  """
  # add constraint on years
  if minYear is not None:
    query = f"{query} AND YEAR(pp.date_of_transfer) >= {minYear}"
  if maxYear is not None:
    query = f"{query} AND YEAR(pp.date_of_transfer) < {maxYear}"
  # constraints on longitude
  if minLongitude is not None:
    query = f"{query} AND pd.longitude > {minLongitude}"
  if maxLongitude is not None:
    query = f"{query} AND pd.longitude < {maxLongitude}"
  # constraints on latitude
  if minLatitude is not None:
    query = f"{query} AND pd.lattitude > {minLatitude}"
  if maxLatitude is not None:
    query = f"{query} AND pd.lattitude < {maxLatitude}"
  # only select some rows
  if oneEvery is not None:
    query = f"{query} AND pp.db_id MOD {oneEvery} = 0"

  return query

def prices_coordinates_range_query_fast(minLon, maxLon, minLat, maxLat, yearRange=None):
  """
  Optimized query to get rows from a certain geographical range and year range, after
  joining `pp_data` and `postcode_data`.

  Returns a pandas dataframe with the results.
  """

  # Cache the postcode_data table
  if prices_coordinates_range_query_fast.all_postcodes_df is None:
    prices_coordinates_range_query_fast.all_postcodes_df = \
      db_select("SELECT postcode, longitude, lattitude FROM postcode_data")

  # filter out postcodes outside bounding box
  #  this is done in pandas instead of SQL because it's faster
  all_pd = prices_coordinates_range_query_fast.all_postcodes_df
  postcode_list = list((all_pd[
      (all_pd['longitude'] > minLon) &
      (all_pd['lattitude'] > minLat) &
      (all_pd['longitude'] < maxLon) &
      (all_pd['lattitude'] < maxLat)
    ])['postcode'])

  postcodes = list(map(lambda x: f"'{x}'", postcode_list))
  if len(postcodes) == 0:
    raise Exception(f"There are no postcodes within the bounding box ({minLon}, {maxLon}, {minLat}, {maxLat})")
  # The pp_data is indexed by postcodes, so this gives efficient retrieval
  cond_sql = f" AND pp.postcode IN (" + ",".join(postcodes) + f")"
  if yearRange is not None:
      cond_sql = f"{cond_sql} AND pp.date_of_transfer BETWEEN '{yearRange[0]}-01-01' AND '{yearRange[1]}-12-30'"
  return db_select(inner_join_sql_query() + cond_sql)

# only compute this dataframe on first call
prices_coordinates_range_query_fast.all_postcodes_df = None

"""
################################# Schema setup ###############################
"""

# schema set up for `pp_data`
def create_pp_data_table():
  # --
  # -- Table structure for table `pp_data`
  # --
  sql_schema_queries = [(" "
    "DROP TABLE IF EXISTS `pp_data`; "),
    (" "
    "CREATE TABLE IF NOT EXISTS `pp_data` ( "
    "  `transaction_unique_identifier` tinytext COLLATE utf8_bin NOT NULL, "
    "  `price` int(10) unsigned NOT NULL, "
    "  `date_of_transfer` date NOT NULL, "
    "  `postcode` varchar(8) COLLATE utf8_bin NOT NULL, "
    "  `property_type` varchar(1) COLLATE utf8_bin NOT NULL, "
    "  `new_build_flag` varchar(1) COLLATE utf8_bin NOT NULL, "
    "  `tenure_type` varchar(1) COLLATE utf8_bin NOT NULL, "
    "  `primary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL, "
    "  `secondary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL, "
    "  `street` tinytext COLLATE utf8_bin NOT NULL, "
    "  `locality` tinytext COLLATE utf8_bin NOT NULL, "
    "  `town_city` tinytext COLLATE utf8_bin NOT NULL, "
    "  `district` tinytext COLLATE utf8_bin NOT NULL, "
    "  `county` tinytext COLLATE utf8_bin NOT NULL, "
    "  `ppd_category_type` varchar(2) COLLATE utf8_bin NOT NULL, "
    "  `record_status` varchar(2) COLLATE utf8_bin NOT NULL, "
    "  `db_id` bigint(20) unsigned NOT NULL "
    ") DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1 ; ")]

  # --
  # -- Indexes for table `pp_data`
  # --
  index_sql_queries = [(" "
    "ALTER TABLE `pp_data` "
    "ADD PRIMARY KEY (`db_id`); "),
    (" "
    "ALTER TABLE `pp_data` "
    "MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=1; "),
    (" "
    "CREATE INDEX `pp.postcode` USING HASH "
    "ON `pp_data` "
    "(postcode); "),
    (" "
    "CREATE INDEX `pp.date` USING HASH "
    "ON `pp_data`  "
    "(date_of_transfer); ")]

  # run the SQL commands
  for query in sql_schema_queries:
    db_query(query)
  for query in index_sql_queries:
    db_query(query)

# schema setup for `postcode_data`
def create_postcode_data_table():

  # --
  # -- Table structure for table `postcode_data`
  # --
  sql_schema_queries = [(
    "DROP TABLE IF EXISTS `postcode_data`; "),
    (" "
    "CREATE TABLE IF NOT EXISTS `postcode_data` ( "
    "  `postcode` varchar(8) COLLATE utf8_bin NOT NULL, "
    "  `status` enum('live','terminated') NOT NULL, "
    "  `usertype` enum('small', 'large') NOT NULL, "
    "  `easting` int unsigned, "
    "  `northing` int unsigned, "
    "  `positional_quality_indicator` int NOT NULL, "
    "  `country` enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL, "
    "  `lattitude` decimal(11,8) NOT NULL, "
    "  `longitude` decimal(10,8) NOT NULL, "
    "  `postcode_no_space` tinytext COLLATE utf8_bin NOT NULL, "
    "  `postcode_fixed_width_seven` varchar(7) COLLATE utf8_bin NOT NULL, "
    "  `postcode_fixed_width_eight` varchar(8) COLLATE utf8_bin NOT NULL, "
    "  `postcode_area` varchar(2) COLLATE utf8_bin NOT NULL, "
    "  `postcode_district` varchar(4) COLLATE utf8_bin NOT NULL, "
    "  `postcode_sector` varchar(6) COLLATE utf8_bin NOT NULL, "
    "  `outcode` varchar(4) COLLATE utf8_bin NOT NULL, "
    "  `incode` varchar(3)  COLLATE utf8_bin NOT NULL, "
    "  `db_id` bigint(20) unsigned NOT NULL "
    ") DEFAULT CHARSET=utf8 COLLATE=utf8_bin; ")]

  # --
  # -- Indexes for table `postcode_data`
  # --
  index_sql_queries = [(
    "ALTER TABLE `postcode_data` "
    "ADD PRIMARY KEY (`db_id`); "),
    (" "
    "ALTER TABLE `postcode_data` "
    "MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=1; "),
    (" "
    "CREATE INDEX `po.postcode` USING HASH "
    "ON `postcode_data` "
    "(postcode); ")]

  # run the SQL commands
  for query in sql_schema_queries:
    db_query(query)
  for query in index_sql_queries:
    db_query(query)

"""
################################# Data import ################################
"""

# Commented out IPython magic to ensure Python compatibility.

# utility functions for downloading csv datasets
def upload_csv(csv_file, table_name):
  quote = '"'
  # upload to database
  upload_sql = (f" "
    f"LOAD DATA LOCAL INFILE '{csv_file}' INTO TABLE `{table_name}` "
    f"FIELDS TERMINATED BY ',' "
    # NOTE: we add this to specify " as quote character
    f"OPTIONALLY ENCLOSED BY '{quote}' "
    # f-strings in python don't interpret \n as newline
    f"LINES STARTING BY '' TERMINATED BY '\n'; " )
  db_query(upload_sql)
  print(f"Uploaded file {csv_file} to database table `{table_name}`")

  print("See the source code. I couldn't get %rm and %load_ext sql to work in my package...")

  # delete the file
  deletion_errors = None # %rm $csv_file
  if deletion_errors is None:
    print(f"Successfully removed file {csv_file}")
  else:
    print(f"Failed to delete file {csv_file}")

def download_file_from_url(url):
  try:
    file_name = url.rsplit('/', 1)[-1]
    urllib.request.urlretrieve(url, file_name)
    print(f'Successfully download file {file_name}.')
    return file_name
  except Exception as e:
    print('The server couldn\'t fulfill the request.')
    print('Error code: ', e.code)
    return None

def download_and_upload_price_paid_data(ymin=1995,ymax=2022):
  """
  Uploads UK price paid data from the year ymin inclusive to the
  year ymax exclusive. This data is loaded into the table named
  `pp_data`, assumed to be in a database called "poperty_prices"
  """
  base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-{}-part{}.csv"
  parts = [1,2]
  for y in range(ymin, ymax):
    for p in parts:
      # get url
      url = base_url.format(y, p)
      # download the file
      file_name = download_file_from_url(url)
      # upload to database
      upload_csv(file_name, "pp_data")

def download_and_upload_postcode_data():
  """
  Downloads and uploads UK postcode data.
  This data is loaded into the table named
  `postcode_data`, assumed to be in a database called "poperty_prices"
  """
  base_url = "https://www.getthedata.com/downloads/open_postcode_geo.csv.zip"
  file_name = download_file_from_url(base_url)
  with zipfile.ZipFile(file_name,"r") as zip_ref:
    zip_ref.extractall()
    upload_csv("open_postcode_geo.csv", "postcode_data")
  # delete the zip file
  print("See the source code. I couldn't get %rm and %load_ext sql to work in my package...")
#   %rm $file_name

"""
################## Open Street Map (OSM) Access functions ###############
"""

# geometry warnings when doing `.distance` with geographical CRS coordinates:
#   We can safely ignore this since the distance calculations are between
#   relatively close points, so the earth's curvature is minimal
# warnings.filterwarnings("ignore", message= ".*Geometry is in a geographic CRS.*")

def km_to_crs(dist_km, latitude=53.71, longitude=-2.03):
  """
  ~~Takes a distance in kilometers and creates a bounding box in EPSG:4326 latitude and~~
  ~~longitude coordinates. Returns a tuple of four elements:~~
  ~~north south west east~~
  """
  return (dist_km / 40075) * 360
  # h = dist_km / 110.574
  # w = dist_km / (math.cos(latitutde * math.pi/180) * 111.320)

def crs_to_km(dist_crs): # two args, lat and lon
  return (dist_crs / 360) * 40075
  # h = 110.574 * lat_diff
  # w = 111.320 * math.cos(lat * math.pi / 180) * lon_diff

  # return math.sqrt(h**2 + w**2)

def get_pois_around_point(longitude, latitude, dist_in_km,
    required_tags=[], keys={}, dropCols=True):
  """
  Returns a geopandas GeoDataFrame of POIs that are located inside a bounding
  box around specified point with sides of length `dist_in_km`.

  These POIs will either have a non-NaN value in a OSM key specified in
  `required_tags` or a valid OSM "key : value" combination specified in `keys`.
  If a key is in both `required_tags` and `keys` the latter will take priority.

  If dropCols is set, the columns not specified in neither required_tags
  nor keys is removed from the returned GeoDataFrame

  :param keys: A dictionary where the keys are valid OSM keys and the values
               are either True or a list of OSM values for that key
  """
  # don't override what is specified in keys
  tags = { k: True for k in required_tags if k not in keys }
  tags = {**tags, **keys}
  # make bb
  delta_dist = km_to_crs(dist_in_km)
  north = latitude + delta_dist
  south = latitude - delta_dist
  west = longitude - delta_dist
  east = longitude + delta_dist

  pois = ox.geometries_from_bbox(north, south, east, west, tags)

  gdf = None
  # don't use non-interesting keys
  # if tags is not None:
  present_keys = [key for key in tags if key in pois.columns]
  if 'geometry' not in present_keys:
    present_keys.append('geometry')
  if dropCols:
    gdf = gpd.GeoDataFrame(pois[present_keys])
  else:
    gdf = gpd.GeoDataFrame(pois)
  gdf.crs = "EPSG:4326"
  return gdf

def get_pois_with_amenity_value(longitude, latitude, dist_in_km, amenities):
  """
  Returns a dataframe of POIs around specified bounding box where the amenity key
  is set to any value in amenities
  """
  pois = get_pois_around_point(longitude, latitude,
                               dist_in_km, ["amenity"])
  # remove columns with all NaNs, and filter out rows not interesting
  return pois[pois['amenity'].isin(amenities)].dropna(axis=1, how='all')

def point_to_bounding_box(longitude, latitude, size):
  return {
      "minLongitude" : longitude - size/2,
      "maxLongitude" : longitude + size/2,
      "minLatitude"  : latitude  - size/2,
      "maxLatitude"  : latitude  + size/2
  }

def flatten(coll):
  """
  Returns a flatten list when given lists of lists, but the strings
  in that list are not flattened.

  See: https://stackoverflow.com/questions/17864466/flatten-a-list-of-strings-and-lists-of-strings-and-lists-in-python
  """
  for i in coll:
    if isinstance(i, Iterable) and not isinstance(i, str):
      for subc in flatten(i):
        yield subc
    else:
      yield i

# vim: shiftwidth=2 tabstop=2
