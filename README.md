# Fynesse

1. [Access](#Access)
    1. [Database setup](#Database-setup)
    2. [Database queries](#Database-queries)
    3. [Schema setup](#Schema-setup)
    4. [Data import](#Data-import)
    5. [OSM access utilities](#OSM-access-utilities)
    6. [General utility functions](#General-utility-functions)
2. [Assess](#Assess)
    1. [Visualisation utilities](#Visualisation-utilities)
    2. [Sanitsation](#Sanitsation)
    3. [Dataframe aggregation](#Dataframe-aggregation)
3. [Address](#Address)
    1. [Adding features](#Adding-features)
    2. [House price prediction](#House-price-prediction)
    3. [Dimensionality reduction](#Dimensionality-reduction)

This repo contains a set of utility functions associated with the tasks of:
* _Access_ ing prices paid data, postcode data and Open Street Map (OSM) data
* _Assess_ ing this data
* _Address_ ing the property price prediction questions.

As such, the functions in this repo is organised into three sections: _Access_, _Assess_ and _Address_ .

## Access

### Database setup

* `get_and_store_credentials()`
* `get_database_details()`
* `create_database(db_name)`
* `get_database_connection(database, get_connection_url=False)`

These functions handle everything from interactively connecting to
the database to getting a `PyMySQL` connection object for doing
queries to it.

Throughout _accessing_ data from our tables, we will do a lot of database
queries and it's more suitable to do this with the `PyMySQL` API than through
magic line commands because this allows more flexibility. For example, we can
interrupt the query manually if we see it is taking forever, without leaving a
connection open. We also create a new connection each time we do a query such
that we don't need to pass a connection object around everywhere. Additionally,
our `db_query` and `db_select` functions allow manually interrupting the SQL
query if it takes an unexpected long time to complete, and safely closing the
connection such that future queries are not affected.


### Database queries

* `db_query(query, database="property_prices")`
* `db_select(query, database="property_prices", n=None, one_every=None)`
* `inner_join_sql_query(minYear=None,maxYear=None,minLatitude=None`
* `prices_coordinates_range_query_fast(minLon, maxLon, minLat, maxLat, yearRange=None)`

The `db_query` and `db_select` functions are abstraction of the process of
getting a database connection, running the query, and then closing the
connection. They also handles `KeyboardInterrupt`s nicely. The former
is used to make changes and the latter is for fetching data
using `SELECT` statements and return the result as a pandas dataframe.

The last two functions are for joining the `pp_data` and `postcode_data` tables
on the fly, and the last one return a dataframe with columns as specified in
the joined schema listed in _Task D._ The `inner_join_sql_query` is just for
getting the query string, because we may want to modify this slightly by e.g.
adding additional conditions. For example:

    db_select(inner_join_sql_query(...) + " AND country = 'England'")

**Note:** The last function is an artefact of me waiting long periods of
time for AWS to finish its queries, so it's a less flexible version
of `db_select(inner_join_sql_query(...))` optimised for amortized speed
using caching.

_All of these functions seem appropriate for putting in "assess" as they can
extract subsets of the accessible data that is interesting to user, and put it
into pandas dataframes which are nice to work with. However, this process of
putting some of the data into pandas dataframe is also a way of making the data
accessible in the notebook. As a result, it's not clear which phase they belong
to._

### Schema setup

* `create_pp_data_table()`
* `create_postcode_data_table()`

These functions simply create new tables in the database
with the schemas specified for `pp_data` and `postcode_data`.

### Data import

* `upload_csv(csv_file, table_name)`
* `download_file_from_url(url)`
* `download_and_upload_price_paid_data(ymin=1995,ymax=2022)`
* `download_and_upload_postcode_data()`

These functions are used for loading `.csv` files from URLs
into the created database tables, more specifically the
`pp_data` and `postcode_data` tables.

### OSM access utilities

* `get_pois_around_point(longitude, latitude, dist_in_km, required_tags=[], keys={}, dropCols=True)`
  * This function makes a bounding box around specified coordinates and returns
    a GeoDataFrame of the POIs found on OSM within this bounding box for the
    specified keys and/or tags.
* `get_pois_with_amenity_value(longitude, latitude, dist_in_km, amenities)`
  * This function has similar functionality, but finds POI with key `amenity` equal to any value
    listed in `amenities`

These were implemented using `osmnx`'s `geometries_from_bbox` API call.

### General utility functions

* `km_to_crs(dist_km, latitude=53.71, longitude=-2.03)`
* `crs_to_km(dist_crs)`
* `flatten(coll)`
* `point_to_bounding_box(longitude, latitude, size)`

These functions do not really belong in the Fynesse framework,
but they are procedures I found myself repeating very often
all the way from the access phase to the address phase, so I
put them in `access.py`.

The `crs` and `km` conversion functions translates between distances
specified in number of `EPSG:4326` degrees and number of kilometres.
**Note:** these functions have some inaccuracies since they don't take
into account the latitude of the location where the translation
happens. Instead, they assume the translation is happening at the
equator. I decided not to fix this because it would require major
refactoring and the inaccuracies were only minor.

## Assess

### Visualisation utilities

* `geoplot(title, figsize=(12,12))`
* `plot_housing_density_against_price_and_amt_data(gdf, loc, osm_accommodation_types)`
* `plot_geographical_heatmap(ax, gdf, val, bins, transform='mean', useLog=False)`
* `plot_pois_around_area(longitude, latitude, dist_in_km, figName, keys)`

* `highlight_topn(n)`
* `highlight_aboven(n)`

The first set of functions are used for various visualisations of either POIs in a
geographical area or for plotting heatmaps with specified aggregation functions.

The last two are for getting `lambda`s that can be passed to `DataFrame.corr().style.apply`
to better visualise feature correlation.

For the heatmap visualisation function, I start by creating a square grid of bins,
and then place each of the rows in `gdf` in a bin
based on their `Point` or centroid location. I then group together the entries in the same
x and y bins and run specified `transform` function on it, using column `val`.
If `transform='count'`, the heatmap will visualise the number of each entry located in
each square bin.

### Sanitsation

* `make_geodataframe(df)`
* `recover_df_from_file(filename, upload_required=True)`
* `find_num_missing_values(table_name, columns)`

These are utility functions for assessing incompleteness in existing data
or for recovering lost information. The main motivation for `recover_df_from_file`
is that when converting a `GeoDataFrame` or pandas `Dataframe` to a `.csv` and then
reimporting it, some of the data in columns like `geometry` and `date_of_transfer`
have their types changed. This functions reencodes such data.

### Dataframe aggregation

* `_add_spatial_aggregate_column(gdf, pois, dist_in_km, col_name, agg_f, nafill=np.nan, cacheBy='postcode')`
* `get_num_pois_within_radius(gdf, pois, dist_in_km, col_name)`
* `get_closest_poi_within_radius(gdf, pois, dist_in_km, col_name)`
* `get_average_dist_to_poi_within_radius(gdf, pois, dist_in_km, col_name)`

These functions take a `GeoDataFrame` and return the same dataframe, but with a new column
with values that are calculated through some specific aggregation after a spatial join
with the second argument `pois`.

The last three functions all use `_add_spatial_aggregate_column` which works as follows:
* Each entry in `gdf` is put in a group together with all the entries in `pois` that are within `dist_in_km` of the `gdf` entry's spatial position.
* A specified aggregation function is run on this group, producing a value (e.g. the number of `pois` entries in the group)
* This value is added to a new column with name `col_name`

When programming this, I also realised how similar
counting the number of POIs within an area and finding the closest POI
within that area are, so there's another hidden utility function which
handles the aggregation done in the most general way: `_add_spatial_aggregate_column`.

I have added these utility function in _Address_ because they enrich
my dataframes with aggregation data that is useful when evaluating how
suitable a feature is for a specific problem, and the utility functions can give
aggregation data that can be used to decide on features for many other problems that other people may address as well

## Address

### Adding features

* `add_feature_from_pois(gdf, pois, larger_gdf=None, **feature_kwargs)`
* `add_many_features_from_pois(gdf, pois, larger_gdf=None, **feature_kwargs)`

These functions take a feature specification dictionary item e.g. on the form:

    {
      'func' : 'closest',
      'pois_cond' : lambda pois : pois[pois['public_transport'].notna()],
      'dist' : 3, # km
      'name' : 'closest_transport'
    }

and adds a new column to the provided dataframe `gdf` with values for this feature.

The implementation of these functions is based on the pois-aggregation functions
implemented in `assess.py`. The functions simply inspect the `func` field and pass
the rest of the arguments onto the relevant pois aggregation function.

### House price prediction

* `build_prices_coordinates_features_dataset(latitude, longitude, date, property_type, bb_size_km, pois_bb_size_km, year_range_size, pois_keys, features, logging=False)`
* `predict_price(latitude, longitude, date, property_type, build_dataset_kwargs, design, printSummary=False, plotAx=None)`
* `predict_price_simple(latitude, longitude, date, property_type)`

The first function fetches all the relevant `prices_coordinates` data and adds a new column
to the dataframe for each of the specified features. This function is encapsulated in
the `predict_price` function and the arguments to it are encoded in the `build_dataset_kwargs` parameter.
I abstracted away the process of building the dataset with all the features because I found it useful
to just build such a dataframe, without doing price predictions, for other tasks done in the notebook.

The `predict_price` function takes in all the basic information about the
property from `latitude` to `property_type`.  It also takes a specification of
a `design` matrix. This specification is a function from a `GeoDataFrame` to a
2-dimensional `numpy` array. The `build_dataset_kwargs` also have to be
specified (specific structure in the function docs). The `predict_price`
function starts by building the dataset using the above utility function.  It
then trains a `statsmodel` GLM on 80% of this data and the model is tested on
the remaining 20%, allowing us to plot the predictions against the true price,
as well as computing various metrics. When making a prediction for the property
type specified through the function parameters, we make use of the
`add_many_features_from_pois` utility function to calculate the feature values
for the single property we are making a prediction for. This requires providing
the `pois` dataframe and the larger `gdf_for_num_houses` dataframe that was
used in the computation of the features for the training set. Therefore, these
dataframes are returned by `build_prices_coordinates_features_dataset` such
that we can pass them on to `add_many_features_from_pois`. All in all, we are
running the functionality to add features twice: One batch run for all of our
training+testing data, and once for the single datum we want to predict the price
for. The former happens in `build_prices_coordinates_features_dataset` and the
latter in `predict_price`, but in both instances `add_many_features_from_pois`
is used.

The `predict_price_simple` simply makes a call to `predict_price`, but with a specified `build_dataset_kwargs`
and `design` matrix. If the user does not want to create their own feature specification or design
matrix, this is the more accessible function.

### Dimensionality reduction

* `do_pca(design_matrix_fun, data_gdf, features)`

This function takes a design matrix a dataset with already evaluated features and
a list of features. It then does PCA on these features and reports how much of the
variance in the data each feature describes.
