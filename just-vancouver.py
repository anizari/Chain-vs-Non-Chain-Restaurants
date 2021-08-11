# Limit OSM data to just greater Vancouver
# Typical invocation:
# spark-submit just-vancouver.py amenities amenities-vancouver
# hdfs dfs -cat amenities-vancouver/* | gzip -d - | gzip -c > amenities-vancouver.json.gz

import sys
import unicodedata

import numpy as np
# from PIL import Image
# from PIL.ExifTags import GPSTAGS, TAGS
from pyspark.sql import Row, SparkSession, functions, types

assert sys.version_info >= (3, 5)  # make sure we have Python 3.5+

spark = SparkSession.builder.appName(
    'OSM point of interest extracter').getOrCreate()
assert spark.version >= '2.4'  # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
# sc = spark.sparkContext


amenity_schema = types.StructType([
    types.StructField('lat', types.DoubleType(), nullable=False),
    types.StructField('lon', types.DoubleType(), nullable=False),
    types.StructField('timestamp', types.TimestampType(), nullable=False),
    types.StructField('amenity', types.StringType(), nullable=False),
    types.StructField('name', types.StringType(), nullable=True),
    types.StructField('tags', types.MapType(
        types.StringType(), types.StringType()), nullable=False),
    types.StructField('wikidata', types.StringType(), nullable=False),
])


def filterData(poi):
    poi = poi.filter((poi['lon'] > -123.5) & (poi['lon'] < -122))
    poi = poi.filter((poi['lat'] > 49) & (poi['lat'] < 49.5))

    # Filter amenities to remove uninteresting places
    # Amenity tag values taken from https://wiki.openstreetmap.org/wiki/Key:amenity#Values
    # Can be placed into seperate function later

    # Food
    foodArray = ['bar', 'biergarten', 'cafe', 'fast_food',
                 'food_court', 'ice_cream', 'pub', 'restaurant']

    poi = poi.filter(poi['amenity'].isin(foodArray)).cache()
    poi.show()

    brands = poi.select(poi['tags']['brand']).distinct().dropna().collect()

    for item in brands:
        normal_item = unicodedata.normalize('NFKD', str(item[0]))
        print(normal_item.encode('ASCII', 'ignore'))

    return poi


def distance(city, stations):
    lat1 = city['latitude']
    lon1 = city['longitude']
    lat2 = stations['latitude']
    lon2 = stations['longitude']
    R = 6371
    p = np.pi / 180  # Pi/180
    a = 0.5 - np.cos((lat2 - lat1) * p) / 2 + np.cos(lat1 * p) * \
        np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    distance = (2 * R * np.arcsin(np.sqrt(a)))
    return distance


def main(inputs, output):
    poi = spark.read.json(inputs, schema=amenity_schema)
    poi = filterData(poi)

    # poi.show()
    poi = poi.coalesce(1)  # ~1MB after the filtering
    poi.write.json(output, mode='overwrite', compression='gzip')


if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    # imgFolder = sys.argv[3]
    main(inputs, output)
