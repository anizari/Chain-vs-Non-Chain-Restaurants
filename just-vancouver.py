# Limit OSM data to just greater Vancouver
# Typical invocation:
# spark-submit just-vancouver.py amenities amenities-vancouver
# hdfs dfs -cat amenities-vancouver/* | gzip -d - | gzip -c > amenities-vancouver.json.gz

import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

from pyspark.sql import SparkSession, functions, types, Row
spark = SparkSession.builder.appName('OSM point of interest extracter').getOrCreate()
assert spark.version >= '2.4' # make sure we have Spark 2.4+
spark.sparkContext.setLogLevel('WARN')
#sc = spark.sparkContext


amenity_schema = types.StructType([
    types.StructField('lat', types.DoubleType(), nullable=False),
    types.StructField('lon', types.DoubleType(), nullable=False),
    types.StructField('timestamp', types.TimestampType(), nullable=False),
    types.StructField('amenity', types.StringType(), nullable=False),
    types.StructField('name', types.StringType(), nullable=True),
    types.StructField('tags', types.MapType(types.StringType(), types.StringType()), nullable=False),
    types.StructField('wikidata', types.StringType(), nullable=False),
])


def main(inputs, output):
    poi = spark.read.json(inputs, schema=amenity_schema)
    poi = poi.filter((poi['lon'] > -123.5) & (poi['lon'] < -122))
    poi = poi.filter((poi['lat'] > 49) & (poi['lat'] < 49.5))
    
    #Filter amenities to remove uninteresting places
    #Amenity tag values taken from https://wiki.openstreetmap.org/wiki/Key:amenity#Values
    #Can be placed into seperate function later

    #Transportation
    poi = poi.filter((poi['amenity'].startswith('bicycle') == False) & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'].startswith('boat') == False) & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'].startswith('bus') == False) & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'].startswith('car') == False) & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'].startswith('parking') == False) & (poi['name'].isNull()))
    
    poi = poi.filter((poi['amenity'] != 'vehicle_inspection') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'charging_station') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'ferry_terminal') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'fuel') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'grit_bin') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'motorcycle_parking') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'taxi') & (poi['name'].isNull()))
    
    #Financial
    poi = poi.filter((poi['amenity'] != 'atm') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'bank') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'bureau_de_change') & (poi['name'].isNull()))
    
    #Healthcare
    poi = poi.filter((poi['amenity'] != 'baby_hatch') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'clinic') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'dentist') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'doctors') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'hospital') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'nursing_home') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'pharmacy') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'social_facility') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'veterinary') & (poi['name'].isNull()))
    
    #Public Service
    poi = poi.filter((poi['amenity'] != 'fire_station') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'police') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'ranger_station') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'].startswith('post') == False) & (poi['name'].isNull()))
    
    #Facilities
    poi = poi.filter((poi['amenity'] != 'bench') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'dog_toilet') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'drinking_water') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'give_box') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'shelter') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'shower') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'telephone') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'toilets') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'water_point') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'watering_place') & (poi['name'].isNull()))
    
    #Waste Management
    poi = poi.filter((poi['amenity'] != 'sanitary_dump_station') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'recycling') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'waste_basket') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'waste_disposal') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'waste_transfer_station') & (poi['name'].isNull()))
    
    #Others
    poi = poi.filter((poi['amenity'].startswith('animal') == False) & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'baking_oven') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'childcare') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'clock') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'crematorium') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'dive_centre') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'funeral_hall') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'hunting_stand') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'internet_cafe') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'kitchen') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'kneipp_water_cure') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'lounger') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'marketplace') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'monastery') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'photo_booth') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'place_of_mourning') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'place_of_worship') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'public_bath') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'refugee_site') & (poi['name'].isNull()))
    poi = poi.filter((poi['amenity'] != 'vending_machine') & (poi['name'].isNull()))
    

    #poi.show()
    #poi = poi.coalesce(1) # ~1MB after the filtering 
    poi.write.json(output, mode='overwrite', compression='gzip')


if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    main(inputs, output)
