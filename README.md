# Fynesse

This repo contains a set of utility functions associated with the tasks of:
* _Access_ing prices paid data, postcode data and Open Street Map (OSM) data
* _Assess_ing this data
* _Address_ing the property price prediction questions.

As such, the functions in this repo is organised into three sections:
1. [Access](##Access)
2. [Assess](##Assess)
3. [Address](##Address)

## Access

The functions `access.py` are split into the following sections:

1. [Database setup](###Database-setup)
2. [Database queries](###Database-queries) (TODO: move to assess)
3. [Schema setup](###Schema-setup)
4. [Data import](###Data-import)
5. [OSM access utilities](###OSM-access-utilities)

### Database setup

``


get_and_store_credentials():
get_database_details():
get_database_connection(database, get_connection_url=False):
create_database(db_name):
get_database_connection(database, get_connection_url=False):
db_query(query, database="property_prices"):
db_select(query, database="property_prices", n=None, one_every=None):
inner_join_sql_query(minYear=None,maxYear=None,minLatitude=None,
prices_coordinates_range_query_fast(minLon, maxLon, minLat, maxLat, yearRange=None):
create_pp_data_table():
create_postcode_data_table():
upload_csv(csv_file, table_name):
download_file_from_url(url):
download_and_upload_price_paid_data(ymin=1995,ymax=2022):
download_and_upload_postcode_data():
km_to_crs(dist_km, latitude=53.71, longitude=-2.03):
crs_to_km(dist_crs): # two args, lat and lon
get_pois_around_point(longitude, latitude, dist_in_km,
get_pois_with_amenity_value(longitude, latitude, dist_in_km, amenities):
point_to_bounding_box(longitude, latitude, size):
flatten(coll):

### Database queries

### Schema setup

### Data import

### OSM access utilities

* `get_pois_around_point(longitude, latitude, dist_in_km, required_tags, keys, dropCols=True)`
  * This function makes a bounding box around specified coordinates and returns a GeoDataFrame
    of the POIs found on OSM within this bounding box for the specified keys
Accessing various points of interests (POIs) from OSM is not specific to the
task of house price predictions, so I've made a general utility function
`get_pois_around_point` which makes a bounding box around specified
coordinate and returns all the POIs in this box containing a non-`NaN` value
for the `keys` specified as interesting. Below is an example of this use:

## Assess.py

### Database queries

### Plotting utilities

## Address.py

Below I give an example of adding a column with the number of houses being sold within a 1km radius, for houses in 4𝑘𝑚2 area around Cambridge (interestingly, the get_num_pois_within_radius, that was initially built to a different purpose, can be reused for this computation):

wider_gdf ....
