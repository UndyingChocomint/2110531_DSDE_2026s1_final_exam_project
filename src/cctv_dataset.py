"""CCTV dataset enrichment. Functions can be imported without running the pipeline."""
from pathlib import Path
from config import DATA_DIR, SRC_DIR
from urllib.parse import urljoin
import re
import time
import math

import pandas as pd
import requests
from bs4 import BeautifulSoup
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point

RAW = DATA_DIR / 'raw'
CAMERA_COL = 'camera_id'


def load_cctv(raw_dir=RAW):
    """Return one row per timestamped CCTV image, with camera metadata."""
    raw_dir = Path(raw_dir)
    info = pd.read_csv(raw_dir / 'cctv_info.csv')
    info.columns = info.columns.str.strip().str.lower()
    info[CAMERA_COL] = info[CAMERA_COL].astype(str)

    rows = []
    for image in (raw_dir / 'cctv_images').glob('*/*.jpg'):
        match = re.search(r'(\d{8}_\d{6})\.jpg$', image.name, re.I)
        if match:
            rows.append((image.parent.name, str(image), match.group(1)))
    if not rows:
        raise ValueError('No timestamped CCTV images were found.')

    df = pd.DataFrame(rows, columns=[CAMERA_COL, 'image_path', 'timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d_%H%M%S')
    df = df.merge(info.drop_duplicates(CAMERA_COL), on=CAMERA_COL, how='left')
    df['date'] = df['timestamp'].dt.normalize()
    return df


def get_longdo_data(start, end, raw_dir=RAW):
    """Return hourly traffic index and incident counts for the CCTV date range."""
    raw_dir = Path(raw_dir)
    page = requests.get('https://traffic.longdo.com/download', timeout=30)
    page.raise_for_status()
    soup = BeautifulSoup(page.text, 'html.parser')

    def download_csv(section, folder):
        links = soup.select(f'li:-soup-contains("{section}") > ul > li a[href]')
        parts = []
        for year in range(start.year, end.year + 1):
            link = next((a['href'] for a in links
                         if str(year) in a['href'] or str(year) in a.get_text()), None)
            if link is None:
                print('Longdo file not found:', folder, year)
                continue
            path = raw_dir / 'traffic_longdo' / folder / f'{year}.csv'
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                response = requests.get(urljoin(page.url, link), timeout=60)
                response.raise_for_status()
                path.write_bytes(response.content)
            part = pd.read_csv(path, encoding='utf-8-sig')
            part.columns = part.columns.str.strip().str.lower()
            parts.append(part)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    traffic = download_csv('ข้อมูลดัชนีรถติดย้อนหลัง', 'traffic_index')
    accidents = download_csv('ข้อมูลเหตุการณ์ย้อนหลัง', 'car_accident')

    if not traffic.empty:
        traffic['timestamp'] = pd.to_datetime(traffic['timestamp'], errors='coerce')
        traffic['index'] = pd.to_numeric(traffic['index'], errors='coerce')
        traffic = traffic[['timestamp', 'index']].dropna().copy()
        traffic['hour'] = traffic['timestamp'].dt.floor('h')
        traffic = (traffic.groupby('hour', as_index=False)['index'].mean()
                   .rename(columns={'index': 'traffic_index'}))
        traffic = traffic[traffic['hour'].between(start, end + pd.Timedelta(days=1))]

    if not accidents.empty:
        # Longdo incident files don't necessarily have a 'timestamp' column.
        # Use their actual event-start column rather than the CCTV filename date.
        date_column = next(
            (col for col in ('timestamp', 'start', 'start_time', 'datetime',
                             'date_time', 'startdate', 'start_date', 'starttime',
                             'event_start', 'date', 'created_at', 'time')
             if col in accidents.columns), None
        )
        if date_column is None:
            raise ValueError(
                f"Longdo incident date column not found. "
                f"Available columns: {accidents.columns.tolist()}"
            )

        # Some incident exports separate the date and time into two columns.
        values = accidents[date_column].astype(str)
        if date_column == 'date' and 'time' in accidents.columns:
            values = values + ' ' + accidents['time'].astype(str)
        if date_column == 'time' and not values.str.contains(r'[-/ ]|\d{8}', regex=True).any():
            raise ValueError('Longdo incidents have a time column but no usable date column.')
        accidents['timestamp'] = pd.to_datetime(values, errors='coerce')
        if accidents['timestamp'].notna().sum() == 0:
            raise ValueError(f'Could not parse Longdo incident dates from column {date_column!r}.')
        accidents['hour'] = accidents['timestamp'].dt.floor('h')
        accidents = (
            accidents.dropna(subset=['hour'])
            .groupby('hour').size()
            .rename('accident_count').reset_index()
        )
        accidents = accidents[
            accidents['hour'].between(start, end + pd.Timedelta(days=1))
        ]

    return traffic, accidents


def get_weather(start, end, latitude, longitude):
    """Return hourly ERA5 weather for a given location."""
    params = {
        'latitude': latitude, 'longitude': longitude,
        'start_date': start.strftime('%Y-%m-%d'),
        'end_date': end.strftime('%Y-%m-%d'),
        'hourly': 'temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,surface_pressure',
        'timezone': 'Asia/Bangkok', 'models': 'era5'
    }
    response = requests.get('https://archive-api.open-meteo.com/v1/archive',
                            params=params, timeout=60)
    response.raise_for_status()
    weather = pd.DataFrame(response.json()['hourly'])
    weather['timestamp'] = pd.to_datetime(weather.pop('time'))
    return weather


def get_roads(cctv_df, on_group=None):
    """Return road attributes per camera, sharing OSM queries for nearby cameras."""
    cameras = cctv_df[['camera_id', 'latitude', 'longitude']].drop_duplicates('camera_id').copy()
    cameras['latitude'] = pd.to_numeric(cameras['latitude'], errors='coerce')
    cameras['longitude'] = pd.to_numeric(cameras['longitude'], errors='coerce')
    cameras = cameras.dropna(subset=['latitude', 'longitude'])

    # # Group cameras into ~3 km cells; ERA5 grouping remains unchanged.
    # # Floor-based cells keep coordinates 100.51 and 100.52 together.
    # cameras['osm_lat'] = ((cameras['latitude'] / 0.03).round() * 0.03).round(2)
    # cameras['osm_lon'] = ((cameras['longitude'] / 0.03).round() * 0.03).round(2)
    
    # Group cameras into approximately 1 km cells
    cameras['osm_lat'] = cameras['latitude'].round(2)
    cameras['osm_lon'] = cameras['longitude'].round(2)
    print('OSM areas:', cameras[['osm_lat', 'osm_lon']].drop_duplicates().shape[0],
          'for', len(cameras), 'cameras')

    # OSMnx saves successful Overpass downloads for future script runs.
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(DATA_DIR / 'cache' / 'osmnx')
    ox.settings.overpass_rate_limit = False
    ox.settings.overpass_url = 'http://localhost:8080/api'
    
    def first(value):
        return value[0] if isinstance(value, list) and value else value

    def number(value):
        return pd.to_numeric(
            pd.Series([str(first(value))]).str.extract(r'(\d+(?:\.\d+)?)')[0],
            errors='coerce'
        ).iloc[0]

    rows = []
    groups = cameras.groupby(['osm_lat', 'osm_lon'])
    total_groups = groups.ngroups
    total_start = time.perf_counter()
    for group_index, ((lat, lon), group) in enumerate(groups):
        area_start = time.perf_counter()
        progress = f'{group_index + 1}/{total_groups}'
        print(f'{progress}: Processing OSM area ({lat:.2f}, {lon:.2f})', flush=True)
        # Center the query on the cameras, not the rounded grid coordinate.
        center_lat = group['latitude'].mean()
        center_lon = group['longitude'].mean()
        lat_m = (group['latitude'] - center_lat) * 111_000
        lon_m = ((group['longitude'] - center_lon) * 111_000
                 * math.cos(math.radians(center_lat)))
        radius = math.ceil((lat_m**2 + lon_m**2).pow(0.5).max() + 500)

        # Space out new area requests even when the previous area failed.
        if group_index:
            time.sleep(5)

        graph = None
        for attempt in range(3):
            try:
                # One shared road-network download covering the entire group.
                graph = ox.graph_from_point(
                    (center_lat, center_lon), dist=radius, network_type='drive'
                )
                break
            except Exception as exc:
                print(f'OSM request failed for ({lat}, {lon}), attempt {attempt + 1}/3: {exc}')
                if attempt < 2:
                    time.sleep(5 * (2 ** attempt))  # 5s, then 10s

        if graph is None:
            print(f'{progress}: Skipped after {time.perf_counter() - area_start:.2f}s (3 attempts)', flush=True)
            continue

        group_start = len(rows)
        try:
            _, edges = ox.graph_to_gdfs(graph)
            edges = edges.to_crs('EPSG:3857')

            for camera in group.itertuples(index=False):
                point = gpd.GeoSeries(
                    [Point(camera.longitude, camera.latitude)], crs='EPSG:4326'
                ).to_crs(edges.crs).iloc[0]
                road = edges.loc[edges.distance(point).idxmin()]
                bridge = first(road.get('bridge'))
                rows.append({
                    'camera_id': camera.camera_id,
                    'road_type': first(road.get('highway')),
                    'lane_count': number(road.get('lanes')),
                    'speed_limit_raw': first(road.get('maxspeed')),
                    'oneway_feature': road.get('oneway'),
                    'bridge_feature': int(pd.notna(bridge) and
                                          str(bridge).lower() not in ('no', 'false'))
                })
        except Exception as exc:
            print(f'{progress}: Road matching failed after {time.perf_counter() - area_start:.2f}s: {exc}', flush=True)

        # Save even partial results if road matching failed after some cameras.
        new_rows = rows[group_start:]
        if new_rows and on_group is not None:
            on_group(pd.DataFrame(new_rows))
        print(f'{progress}: Finished in {time.perf_counter() - area_start:.2f}s | '
              f'{len(new_rows)}/{len(group)} cameras', flush=True)

    print(f'OSM processing complete in {time.perf_counter() - total_start:.2f}s', flush=True)

    return pd.DataFrame(rows, columns=[
        'camera_id', 'road_type', 'lane_count', 'speed_limit_raw',
        'oneway_feature', 'bridge_feature'
    ])


def build_dataset(raw_dir=RAW, csv_path=None):
    """Build a new dataset, or retry missing OSM roads in an existing CSV."""
    road_cols = ['road_type', 'lane_count', 'speed_limit_raw',
                 'oneway_feature', 'bridge_feature']

    output = Path(csv_path) if csv_path is not None else DATA_DIR / 'processed/cctv_processed.csv'

    def save_checkpoint():
        # Write to a temporary file first, so an interrupted write cannot corrupt the CSV.
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_name(output.name + '.tmp')
        df.to_csv(temp, index=False)
        temp.replace(output)

    def save_roads(roads):
        roads = roads.drop_duplicates('camera_id').set_index('camera_id')
        for col in road_cols:
            if col not in df.columns:
                df[col] = pd.NA
            values = df['camera_id'].map(roads[col])
            missing = df[col].isna() | df[col].astype('string').str.strip().eq('').fillna(False)
            df[col] = df[col].where(~missing | values.isna(), values)
        save_checkpoint()
        print(f'Checkpoint saved: {output}', flush=True)

    if csv_path is not None:
        # Resume from a processed CSV. Longdo and ERA5 are NOT downloaded again.
        df = pd.read_csv(csv_path, dtype={'camera_id': str})
        df.columns = df.columns.str.strip().str.lower()
        df['camera_id'] = df['camera_id'].astype(str)

        # Missing road_type identifies cameras whose road matching was skipped.
        if 'road_type' not in df.columns:
            df['road_type'] = pd.NA
        missing = df['road_type'].isna() | df['road_type'].astype('string').str.strip().eq('').fillna(False)
        cameras = df.loc[missing, ['camera_id', 'latitude', 'longitude']].drop_duplicates('camera_id')
        print('Retrying OSM for', len(cameras), 'cameras')
        if not cameras.empty:
            get_roads(cameras, on_group=save_roads)
        return df

    df = load_cctv(raw_dir)
    start, end = df['date'].min(), df['date'].max()
    print('Date range:', start.date(), 'to', end.date())

    # Match external hourly features without modifying the original CCTV timestamp.
    df['hour'] = df['timestamp'].dt.floor('h')
    traffic, accidents = get_longdo_data(start, end, raw_dir)
    if not traffic.empty:
        df = df.merge(traffic, on='hour', how='left', validate='many_to_one')
    if not accidents.empty:
        df = df.merge(accidents, on='hour', how='left', validate='many_to_one')

    # Share one weather request among cameras with similar coordinates.
    cameras = df[['camera_id', 'latitude', 'longitude']].drop_duplicates('camera_id').copy()
    cameras['latitude'] = pd.to_numeric(cameras['latitude'], errors='coerce')
    cameras['longitude'] = pd.to_numeric(cameras['longitude'], errors='coerce')
    cameras['weather_lat'] = cameras['latitude'].round(1)
    cameras['weather_lon'] = cameras['longitude'].round(1)

    locations = cameras[['weather_lat', 'weather_lon']].dropna().drop_duplicates()
    print('ERA5 locations:', len(locations), 'for', len(cameras), 'cameras')

    weather_parts = []
    for location in locations.itertuples(index=False):
        weather = get_weather(start, end, location.weather_lat, location.weather_lon)
        weather['weather_lat'] = location.weather_lat
        weather['weather_lon'] = location.weather_lon
        weather_parts.append(weather)

    if weather_parts:
        weather = pd.concat(weather_parts, ignore_index=True)
        weather = weather.rename(columns={'timestamp': 'weather_hour'})
        df = df.merge(cameras[['camera_id', 'weather_lat', 'weather_lon']],
                      on='camera_id', how='left')
        df['weather_hour'] = df['hour']
        df = df.merge(weather, on=['weather_lat', 'weather_lon', 'weather_hour'], how='left')
        df = df.drop(columns=['weather_lat', 'weather_lon', 'weather_hour'])

    df = df.drop(columns='hour')
    for col in road_cols:
        if col not in df.columns:
            df[col] = pd.NA
    # Save CCTV, Longdo and ERA5 before starting OSM.
    save_checkpoint()
    get_roads(df, on_group=save_roads)
    return df


if __name__ == '__main__':
    df = build_dataset()
    output = DATA_DIR / 'processed/cctv_processed.csv'
    print('Saved:', output, '| shape:', df.shape)
