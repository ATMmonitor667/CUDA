import sys
import numpy as np
sys.path.append('./build')
import pandas as pd
import haversine_library
import timeit
import matplotlib.pyplot as plt

def combineAll():
  allData = []
  string = "/tmp/tlcdata/yellow_tripdata_2009-"
  for i in range(1,13):
    currString = string
    if i < 10:
      currString += "0"
    currString += str(i) + ".parquet"
    taxi = pd.read_parquet(currString)
    allData.append(taxi)
  return pd.concat(allData, ignore_index=True)

def cleanData(taxi):
  taxi = taxi[(taxi['Start_Lon'] != 0) & (taxi['Start_Lat'] != 0) & (taxi['End_Lon'] != 0) & (taxi['End_Lat'] != 0)]
  taxi = taxi.dropna(subset=['Start_Lon','Start_Lat','End_Lon','End_Lat'])
  min_lon, min_lat, max_lon, max_lat = -74.15, 40.5774, -73.7004, 40.9176
  taxi = taxi[
    (taxi['Start_Lon'] >= min_lon) & (taxi['Start_Lon'] <= max_lon) &
    (taxi['Start_Lat'] >= min_lat) & (taxi['Start_Lat'] <= max_lat) &
    (taxi['End_Lon'] >= min_lon) & (taxi['End_Lon'] <= max_lon) &
    (taxi['End_Lat'] >= min_lat) & (taxi['End_Lat'] <= max_lat)
  ]
  return taxi

def haversineReimplenetation(size,x1,y1,x2,y2,dist):
  R = 6371.0
  for i in range(size):
      lat1 = np.radians(y1[i])
      lon1 = np.radians(x1[i])
      lat2 = np.radians(y2[i])
      lon2 = np.radians(x2[i])
      dlat = lat2 - lat1
      dlon = lon2 - lon1
      a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
      c = 2 * np.arcsin(np.sqrt(a))
      dist[i] = R * c
  return dist

data = combineAll()
taxi_clean = cleanData(data)
x1_clean = taxi_clean['Start_Lon'].to_numpy()
y1_clean = taxi_clean['Start_Lat'].to_numpy()
x2_clean = taxi_clean['End_Lon'].to_numpy()
y2_clean = taxi_clean['End_Lat'].to_numpy()
size_clean = len(x1_clean)
dist_cuda = np.zeros(size_clean)
dist_python = np.zeros(size_clean)

cuda_time = timeit.timeit(
    lambda: haversine_library.haversine_distance(size_clean, x1_clean, y1_clean, x2_clean, y2_clean, dist_cuda),
    number=1
)

python_time = timeit.timeit(
    lambda: haversineReimplenetation(size_clean, x1_clean, y1_clean, x2_clean, y2_clean, dist_python),
    number=1
)


print(f"Dataset size:               {size_clean:,} records")
print(f"Python execution time:      {python_time*1000:.3f} ms")
print(f"CUDA execution time:        {cuda_time*1000:.3f} ms")

max_diff = np.max(np.abs(dist_cuda - dist_python))
print(f"Max difference: {max_diff:.2e}\n")


#---------------------------------------------------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('NYC Yellow Cab Trip Data Distribution (2009)', fontsize=16, fontweight='bold')

axes[0, 0].hist(x1_clean, bins=50, color='blue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Start Longitude')
axes[0, 0].set_xlabel('Longitude')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].hist(y1_clean, bins=50, color='green', edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Start Latitude')
axes[0, 1].set_xlabel('Latitude')
axes[0, 1].set_ylabel('Frequency')

axes[0, 2].hist(x2_clean, bins=50, color='red', edgecolor='black', alpha=0.7)
axes[0, 2].set_title('End Longitude')
axes[0, 2].set_xlabel('Longitude')
axes[0, 2].set_ylabel('Frequency')

axes[1, 0].hist(y2_clean, bins=50, color='orange', edgecolor='black', alpha=0.7)
axes[1, 0].set_title('End Latitude')
axes[1, 0].set_xlabel('Latitude')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].hist(dist_cuda, bins=50, color='purple', edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Trip Distance (CUDA)')
axes[1, 1].set_xlabel('Distance (km)')
axes[1, 1].set_ylabel('Frequency')

axes[1, 2].hist(dist_cuda, bins=50, color='brown', edgecolor='black', alpha=0.7)
axes[1, 2].set_title('Trip Distance (Log Scale)')
axes[1, 2].set_xlabel('Distance (km)')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_yscale('log')

plt.tight_layout()
plt.savefig('histograms.png', dpi=150, bbox_inches='tight')
print("Histograms saved to 'histograms.png'")
plt.show()

print(f"Start Longitude: min={np.min(x1_clean):.4f}, max={np.max(x1_clean):.4f}, mean={np.mean(x1_clean):.4f}")
print(f"Start Latitude:  min={np.min(y1_clean):.4f}, max={np.max(y1_clean):.4f}, mean={np.mean(y1_clean):.4f}")
print(f"End Longitude:   min={np.min(x2_clean):.4f}, max={np.max(x2_clean):.4f}, mean={np.mean(x2_clean):.4f}")
print(f"End Latitude:    min={np.min(y2_clean):.4f}, max={np.max(y2_clean):.4f}, mean={np.mean(y2_clean):.4f}")
print(f"Distance (km):   min={np.min(dist_cuda):.4f}, max={np.max(dist_cuda):.4f}, mean={np.mean(dist_cuda):.4f}\n")

data_full = combineAll()
taxi_full = cleanData(data_full)

datetime_col = None
for col in ['Pickup_DateTime', 'Pickup_date', 'pickupDateTime', 'pickup_datetime', 'tpep_pickup_datetime']:
    if col in taxi_full.columns:
        datetime_col = col
        break

if datetime_col:
    taxi_full[datetime_col] = pd.to_datetime(taxi_full[datetime_col])
    taxi_full['day_of_week'] = taxi_full[datetime_col].dt.dayofweek
    taxi_full['is_weekend'] = taxi_full['day_of_week'].isin([5, 6])

    x1_full = taxi_full['Start_Lon'].to_numpy()
    y1_full = taxi_full['Start_Lat'].to_numpy()
    x2_full = taxi_full['End_Lon'].to_numpy()
    y2_full = taxi_full['End_Lat'].to_numpy()
    size_full = len(x1_full)
    dist_full = np.zeros(size_full)

    haversine_library.haversine_distance(size_full, x1_full, y1_full, x2_full, y2_full, dist_full)
    taxi_full['distance_km'] = dist_full

    weekday_trips = taxi_full[~taxi_full['is_weekend']]
    weekend_trips = taxi_full[taxi_full['is_weekend']]

    print("="*70)
    print("WEEKDAY/WEEKEND ANALYSIS")
    print("="*70)
    print(f"{'Metric':<30} {'Weekday':<20} {'Weekend':<20}")
    print(f"{'-'*70}")
    print(f"{'Number of trips':<30} {len(weekday_trips):<20,d} {len(weekend_trips):<20,d}")
    print(f"{'Avg distance (km)':<30} {np.mean(weekday_trips['distance_km']):<20.2f} {np.mean(weekend_trips['distance_km']):<20.2f}")
    print(f"{'Median distance (km)':<30} {np.median(weekday_trips['distance_km']):<20.2f} {np.median(weekend_trips['distance_km']):<20.2f}")
    print(f"{'-'*70}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(weekday_trips['distance_km'], bins=50, alpha=0.6, label='Weekday', color='blue', edgecolor='black')
    axes[0].hist(weekend_trips['distance_km'], bins=50, alpha=0.6, label='Weekend', color='red', edgecolor='black')
    axes[0].set_xlabel('Trip Distance (km)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Trip Distance: Weekday vs Weekend')
    axes[0].legend()

    categories = ['Weekday', 'Weekend']
    counts = [len(weekday_trips), len(weekend_trips)]
    avg_distances = [np.mean(weekday_trips['distance_km']), np.mean(weekend_trips['distance_km'])]

    x_pos = np.arange(len(categories))
    width = 0.35

    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    ax2.bar(x_pos - width/2, counts, width, label='Trip Count', color='skyblue', edgecolor='black')
    ax2_twin.bar(x_pos + width/2, avg_distances, width, label='Avg Distance', color='lightcoral', edgecolor='black')

    ax2.set_xlabel('Trip Type')
    ax2.set_ylabel('Number of Trips', color='skyblue')
    ax2_twin.set_ylabel('Average Distance (km)', color='lightcoral')
    ax2.set_title('Trip Count and Average Distance')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories)

    plt.tight_layout()
    plt.savefig('weekday_weekend_analysis.png', dpi=150, bbox_inches='tight')
    print("\nWeekday/Weekend analysis saved to 'weekday_weekend_analysis.png'")
    plt.show()
else:
    print("Datetime column not found. Skipping weekday/weekend analysis.")

