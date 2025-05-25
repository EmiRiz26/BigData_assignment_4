from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, TimestampType
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import math
from pyspark.sql.functions import udf

# Start Spark session
spark = SparkSession.builder \
    .appName("Assignment_4") \
    .getOrCreate()

# Load the CSV file
df = spark.read.csv("aisdk-2024-05-04.csv", header=True, inferSchema=True)

# Select only the necessary columns
df = df.select(
    "MMSI", "Latitude", "Longitude", "`# Timestamp`"
)

# Convert columns to appropriate types
df = df.withColumn("Latitude", F.col("Latitude").cast(DoubleType())) \
       .withColumn("Longitude", F.col("Longitude").cast(DoubleType())) \
       .withColumn("Timestamp", F.to_timestamp(F.col("`# Timestamp`"), "dd/MM/yyyy HH:mm:ss"))

# Filter out rows with invalid latitude or longitude values
df = df.filter((df["Latitude"].between(-90, 90)) & (df["Longitude"].between(-180, 180)))

# Filter out rows with any missing values in the key columns
df = df.filter(df["Latitude"].isNotNull() & df["Longitude"].isNotNull() & df["Timestamp"].isNotNull())

# Define the Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    if None in [lat1, lon1, lat2, lon2]:
        return None  
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    R = 6371  # Radius of the Earth in kilometers
    distance = R * c  # Distance in kilometers
    return distance

# UDF to compute distance
haversine_udf = udf(haversine, DoubleType())

# Create a window to get the previous row's latitude and longitude within each MMSI group
windowSpec = Window.partitionBy("MMSI").orderBy("Timestamp")

# Add columns for previous latitude and longitude
df = df.withColumn("prev_lat", F.lag(df["Latitude"]).over(windowSpec)) \
       .withColumn("prev_lon", F.lag(df["Longitude"]).over(windowSpec))

# Calculate the distance traveled between consecutive points
df = df.withColumn("distance", haversine_udf(df["Latitude"], df["Longitude"], df["prev_lat"], df["prev_lon"]))

# Filter out rows with null or zero distance
df = df.filter(df["distance"].isNotNull())

## There were a lot of "ships" that had were travelling by impossible speed and thus their travelled distance was just too big to make sense
# Filter out rows where the calculated distance exceeds a threshold (in this case, I took a max cargo vessel speed of 50 knots and researched what is the longest distance a vessel can travel in a day, which was less than 1000km)
threshold = 1000  # in kilometers
df = df.filter(df["distance"] <= threshold)

# Add time difference in hours
df = df.withColumn("prev_time", F.lag("Timestamp").over(windowSpec))
df = df.withColumn("time_diff_hours", (F.unix_timestamp("Timestamp") - F.unix_timestamp("prev_time")) / 3600)

# Calculate speed (km/h)
df = df.withColumn("speed", df["distance"] / df["time_diff_hours"])

# Filter out speeds that are too high (e.g., > 50 knots ≈ 93 km/h)
df = df.filter((df["speed"] < 100) & (df["speed"].isNotNull()))

# Sum the distance for each MMSI to get total distance traveled
df_total_distance = df.groupBy("MMSI").agg(F.sum("distance").alias("total_distance"))

# Sort by the total distance to find the vessel with the longest route
df_longest_route = df_total_distance.orderBy("total_distance", ascending=False).limit(1)

# Show the result
df_longest_route.show()

print("The longest route was traveled by MMSI 219133000 - a total of more than 787km")
