# Limit OSM data to just greater Vancouver
# Typical invocation:
# spark-submit just-vancouver.py amenities amenities-vancouver
# hdfs dfs -cat amenities-vancouver/* | gzip -d - | gzip -c > amenities-vancouver.json.gz

import sys

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS
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


def cleanCode(poi):
    poi = poi.filter((poi['lon'] > -123.5) & (poi['lon'] < -122))
    poi = poi.filter((poi['lat'] > 49) & (poi['lat'] < 49.5))

    # Filter amenities to remove uninteresting places
    # Amenity tag values taken from https://wiki.openstreetmap.org/wiki/Key:amenity#Values
    # Can be placed into seperate function later

    # Food
    foodArray = ['bar', 'biergarten', 'cafe', 'fast_food',
                 'food_court', 'ice_cream', 'pub', 'restaurant']

    # Education
    educationArray = ['library']

    # Transportation
    transportationArray = ['bicycle_rental', 'boat_rental', 'bus_station',
                           'car_rental', 'charging_station', 'ferry_terminal', 'parking']

    # Financial
    financialArray = ['atm', 'burea_de_change']

    # Healthcare
    healthcareArray = ['clinic', 'hospital', 'pharmacy']

    # Entertainment, Arts, Culture
    entertainmentArray = ['arts_centre', 'casino', 'cinema', 'community_centre',
                          'conference_centre', 'fountain', 'nightclub', 'planetarium', 'theatre']

    # Public Services
    publicArray = ['post_box', 'post_office']

    # Facilities
    facilitiesArray = ['bench']

    # Others
    othersArray = ['clock', 'internet_cafe', 'marketplace']

    poi = poi.filter(poi['amenity'].isin(foodArray)
                     | poi['amenity'].isin(educationArray)
                     | poi['amenity'].isin(transportationArray)
                     | poi['amenity'].isin(financialArray)
                     | poi['amenity'].isin(healthcareArray)
                     | poi['amenity'].isin(entertainmentArray)
                     | poi['amenity'].isin(publicArray)
                     | poi['amenity'].isin(facilitiesArray)
                     | poi['amenity'].isin(othersArray))

    poi = poi.filter(~((poi['amenity'] == 'bench') & (poi['name'].isNull())))
    poi = poi.filter(
        ~((poi['amenity'] == 'fountain') & (poi['name'].isNull())))

    return poi


def get_exif(filename):
    # Code from https://developer.here.com/blog/getting-started-with-geocoding-exif-image-metadata-in-python3
    image = Image.open(filename)
    image.verify()
    return image._getexif()


def get_geotagging(exif):
    # Code from https://developer.here.com/blog/getting-started-with-geocoding-exif-image-metadata-in-python3
    if not exif:
        raise ValueError("No EXIF metadata found")

    geotagging = {}
    for (idx, tag) in TAGS.items():
        if tag == 'GPSInfo':
            if idx not in exif:
                raise ValueError("No EXIF geotagging found")

            for (key, val) in GPSTAGS.items():
                if key in exif[idx]:
                    geotagging[val] = exif[idx][key]

    return geotagging


def get_decimal_from_dms(dms, ref):
    # Code from https://developer.here.com/blog/getting-started-with-geocoding-exif-image-metadata-in-python3
    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0

    if ref in ['S', 'W']:
        degrees = -degrees
        minutes = -minutes
        seconds = -seconds

    return round(degrees + minutes + seconds, 5)


def get_coordinates(geotags):
    # Code from https://developer.here.com/blog/getting-started-with-geocoding-exif-image-metadata-in-python3
    lat = get_decimal_from_dms(
        geotags['GPSLatitude'], geotags['GPSLatitudeRef'])

    lon = get_decimal_from_dms(
        geotags['GPSLongitude'], geotags['GPSLongitudeRef'])

    return (lat, lon)


def main(inputs, output):
    poi = spark.read.json(inputs, schema=amenity_schema)
    poi = cleanCode(poi)

    exif = get_exif('img/car.jpg')
    geotags = get_geotagging(exif)
    print(get_coordinates(geotags))

    # poi.show()
    # poi = poi.coalesce(1) # ~1MB after the filtering
    poi.write.json(output, mode='overwrite', compression='gzip')


if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    main(inputs, output)
