from .config import *

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import mongodb
import sqlite"""

"""
 Pip dependencies
"""
%pip install ipython-sql
%pip install PyMySQL
%load_ext sql

"""
 Python module dependencies
"""
from google.colab import files
from ipywidgets import interact_manual, Text, Password

import pymysql
import urllib.request
import yaml

"""
 Place commands in this file to access the data electronically. Don't remove
 any missing values, or deal with outliers. Make sure you have legalities
 correct, both intellectual property and personal data privacy rights. Beyond
 the legal side also think about the ethical issues around this data.
"""

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

def db_query(conn, query, n=None):
  """
  Perform specified SQL query using provided connection
  and limit the result to the first n rows if n is
  specified.
  :param conn: the Connection object
  :param query: The SQL query
  :param n: number of rows (optional)
  """
  cur = conn.cursor()
  if n is not None:
    query = f"{query} LIMIT {n}"
  cur.execute(query)
  return cur.fetchall()

def create_database(db_name):
  # first connect to our MariaDB server
  connect_string = get_database_connection(db_name, True)

  %sql mariadb+pymysql://$connect_string
  %sql SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
  %sql SET time_zone = "+00:00";

  %sql CREATE DATABASE IF NOT EXISTS `$db_name` DEFAULT CHARACTER SET utf8 COLLATE utf8_bin;

  %sql USE `$db_name`

def create_pp_data_table(conn):
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
    db_query(conn, query)
  for query in index_sql_queries:
    db_query(conn, query)

def download_and_upload_csv(conn, csv_url):
  # download the file
  file_name = csv_url.rsplit('/', 1)[-1]
  urllib.request.urlretrieve(csv_url, file_name)
  print(f"Downloaded file: {file_name}")

  # upload to database
  upload_sql = (f" "
      f"LOAD DATA LOCAL INFILE '{file_name}' INTO TABLE `pp_data` "
      f"FIELDS TERMINATED BY ',' "
      # note the additional \ to pass '\n' to the SQL command
      f"LINES STARTING BY '' TERMINATED BY '\\n'; " )
  db_query(conn, upload_sql)
  print(f"Uploaded file {file_name} to database table `pp_data`")

  # delete the file
  deletion_errors = %rm $file_name
  if deletion_errors is None:
    print(f"Successfully removed file {file_name}")
  else:
    print(f"Failed to delete file {file_name}")

def download_and_upload_price_paid_data():
  base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-{}-part{}.csv"
  parts = [1,2]
  for y in range(1995, 2022):
    for p in parts:
      url = base_url.format(y, p)
      print(url)

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

def db_query(conn, query, n=None):
  """
  Perform specified SQL query using provided connection
  and limit the result to the first n rows if n is
  specified.
  :param conn: the Connection object
  :param query: The SQL query
  :param n: number of rows (optional)
  """
  cur = conn.cursor()
  if n is not None:
    query = f"{query} LIMIT {n}"
  cur.execute(query)
  return cur.fetchall()

# Sample function
def data():
  """Read the data from the web or local file, returning structured format
  such as a data frame"""
  raise NotImplementedError

# vim: shiftwidth=2 tabstop=2
