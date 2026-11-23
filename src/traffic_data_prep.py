"""Prepare CCTV images + traffic/weather/road features for PyTorch.

Install: pip install pandas numpy scikit-learn torch torchvision pillow joblib

Project structure:
    project/source/config.py
    project/source/traffic_data_prep.py
    project/data/interim/cctv_interim.csv
    project/data/raw/cctv_images/
    project/data/processed/{train,val,test}.pt

Run from the project root:
    python source/traffic_data_prep.py

Change CSV_PATH, IMAGE_ROOT, IMAGE_SIZE, or BATCH_SIZE in the settings below
if your project uses different locations or training settings.
The script saves tabular tensors, labels, and image paths (not image pixels).

In another Python file:
    from traffic_data_prep import prepare_datasets
    train_ds, val_ds, test_ds, preprocess = prepare_datasets(
        DATA_DIR / 'interim' / 'cctv_interim.csv', DATA_DIR / 'raw' / 'cctv_images'
    )
    sample = train_ds[0]  # image [3,224,224], features [N], target scalar

To reuse saved datasets in training without running Pandas preparation again:
    from traffic_data_prep import load_saved_datasets
    train_ds, val_ds, test_ds = load_saved_datasets()

Target: crash_next_hour is the verified spatially matched, future-window 0/1
label from cctv_dataset.py. Legacy citywide accident_count is NOT accepted.
"""

import json
from pathlib import Path, PureWindowsPath

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import DATA_DIR, SRC_DIR


# 1. Configuration: paths, settings, selected features, and target
CSV_PATH = DATA_DIR / 'interim' / 'cctv_interim.csv'
IMAGE_ROOT = DATA_DIR / 'raw' / 'cctv_images'
OUTPUT_DIR = DATA_DIR / 'processed'
IMAGE_SIZE = 224
BATCH_SIZE = 16

TARGET = 'crash_next_hour'
NUMERIC = [
    'latitude', 'longitude', 'temperature_2m', 'relative_humidity_2m',
    'precipitation', 'wind_speed_10m', 'surface_pressure',
    'lane_count', 'speed_limit_raw', 'oneway_feature', 'bridge_feature',
    'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
]
CATEGORICAL = ['road_type']

# Explicit whitelist: columns not listed in NUMERIC or CATEGORICAL are NOT
# model inputs. In particular, crash_next_hour, accident_count, camera_id,
# date, and other accident/crash fields are excluded from tabular features.
# Image pixels are separately loaded using image_path.
LEAKAGE_TERMS = ('accident', 'crash', 'collision', 'injury', 'fatality')


def check_feature_leakage():
    """Fail if a target-like field is accidentally added as a predictor."""
    selected = NUMERIC + CATEGORICAL
    if len(selected) != len(set(selected)):
        raise ValueError('Duplicate input feature names detected.')
    unsafe = [name for name in selected
              if name == TARGET or any(term in name.lower() for term in LEAKAGE_TERMS)]
    if unsafe:
        raise ValueError(f'Potential target leakage in input features: {unsafe}')


REQUIRED = ['image_path', 'timestamp', TARGET, *[x for x in NUMERIC if not x.endswith(('_sin', '_cos'))], *CATEGORICAL]


# 2. Inspect and clean the dataset

def load_and_clean(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    if TARGET not in df and 'accident_count' in df:
        raise ValueError('Legacy accident_count is citywide and cannot be used as a crash label. Rebuild the CSV using the updated cctv_dataset.py.')
    missing = sorted(set(REQUIRED + ['label_method', 'label_window_minutes', 'label_radius_m']) - set(df.columns))
    if missing:
        raise ValueError(f'Missing required CSV columns: {missing}')

    if not df['label_method'].eq('future_nearby_incident').all():
        raise ValueError('Unsupported or unverified crash-label method in input CSV.')
    if not df['label_window_minutes'].eq(60).all() or not df['label_radius_m'].eq(500).all():
        raise ValueError('This pipeline expects 60-minute, 500-meter crash labels.')
    print(f'Loaded {len(df):,} rows and {len(df.columns)} columns')
    print('Missing values (before cleaning):\n', df.isna().sum().loc[lambda s: s > 0])
    # Optional EDA:
    # print(df.describe(include='all'))
    # import matplotlib.pyplot as plt
    # df[['temperature_2m', 'precipitation', 'wind_speed_10m']].boxplot(); plt.show()
    # df[TARGET].value_counts().plot.bar(); plt.title('Crash labels'); plt.show()
    # import seaborn as sns
    # sns.heatmap(df.select_dtypes('number').corr(), cmap='coolwarm'); plt.show()

    df = df.drop_duplicates(subset=['image_path']).copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
    if df[TARGET].isna().any() or not df[TARGET].isin([0, 1]).all():
        raise ValueError('Target must contain verified 0/1 labels, without missing values.')
    df = df.dropna(subset=['image_path', 'timestamp'])
    df[TARGET] = df[TARGET].astype('float32')
    counts = df[TARGET].value_counts().sort_index()
    print('Binary target distribution (0=no crash, 1=crash):', counts.to_dict())
    if df[TARGET].nunique() < 2:
        raise ValueError('Only one crash-label class exists; do not train a binary classifier on these data.')

    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['road_type'] = df['road_type'].astype('string').str.strip().fillna('Unknown')
    df.loc[df['road_type'].eq(''), 'road_type'] = 'Unknown'
    return df.sort_values('timestamp').reset_index(drop=True)


# 3. Feature engineering: cyclic time features

def add_time_features(df):
    df = df.copy()
    hour = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60
    weekday = df['timestamp'].dt.day_of_week
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * weekday / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * weekday / 7)
    return df


# 4. Resolve and validate image paths

def resolve_image_path(raw_path, image_root):
    """Works with the original Windows paths when images are moved to a new machine."""
    raw_path = str(raw_path)
    original = Path(raw_path)
    if ('\\' not in raw_path or Path(raw_path).anchor) and original.is_file():
        return original

    parts = PureWindowsPath(raw_path).parts
    matching = [i for i, item in enumerate(parts) if item.lower() == 'cctv_images']
    if matching:
        relative = Path(*parts[matching[-1] + 1:])
        return Path(image_root) / relative
    # Fall back to image_root/<image filename> for other CSV exports.
    return Path(image_root) / PureWindowsPath(raw_path).name


def keep_existing_images(df, image_root):
    df = df.copy()
    df['resolved_image_path'] = df['image_path'].map(lambda p: str(resolve_image_path(p, image_root)))
    exists = df['resolved_image_path'].map(lambda p: Path(p).is_file())
    print(f'Images found: {exists.sum():,}/{len(df):,}; missing: {(~exists).sum():,}')
    if not exists.any():
        raise FileNotFoundError(
            'No CCTV images found. Set IMAGE_ROOT to the folder '
            'containing camera-ID subfolders (e.g. cctv_images/126/*.jpg). '
            f'Example searched: {df["resolved_image_path"].iloc[0] if len(df) else "(empty CSV)"}'
        )
    return df.loc[exists].reset_index(drop=True)


# 5. Split chronologically BEFORE fitting imputation/encoding/scaling

def chronological_split(df):
    """Hold out entire FUTURE dates, never random nearby CCTV frames."""
    dates = sorted(df['timestamp'].dt.normalize().unique())
    if len(dates) < 3:
        raise ValueError('Need at least 3 distinct dates to create train/validation/test splits.')
    train_end = max(1, int(len(dates) * 0.7))
    val_end = min(len(dates) - 1, max(train_end + 1, int(len(dates) * 0.85)))
    train_dates, val_dates, test_dates = dates[:train_end], dates[train_end:val_end], dates[val_end:]
    groups = tuple(df[df['timestamp'].dt.normalize().isin(days)].copy() for days in
                   (train_dates, val_dates, test_dates))
    if any(len(g) == 0 for g in groups):
        raise ValueError('Chronological split created an empty partition.')
    # Purge samples whose future-label window crosses the next split boundary.
    train = groups[0].loc[groups[0]['timestamp'] + pd.Timedelta(minutes=float(df['label_window_minutes'].iloc[0])) < groups[1]['timestamp'].min()].copy()
    val = groups[1].loc[groups[1]['timestamp'] + pd.Timedelta(minutes=float(df['label_window_minutes'].iloc[0])) < groups[2]['timestamp'].min()].copy()
    test = groups[2]
    groups = (train, val, test)
    if any(len(g) == 0 for g in groups):
        raise ValueError('Prediction-horizon boundary guard removed an entire partition.')
    # Guard against accidental changes that leak data between partitions.
    train, val, test = groups
    if not (train['timestamp'].max() < val['timestamp'].min()
            and val['timestamp'].max() < test['timestamp'].min()):
        raise AssertionError('Dates overlap or are not chronological.')
    for left, right in ((train, val), (train, test), (val, test)):
        if not set(left['image_path']).isdisjoint(right['image_path']):
            raise AssertionError('Duplicate image paths across data splits.')
    for name, part in zip(('Train', 'Validation', 'Test'), groups):
        print(f'{name}: {len(part):,} rows | {part.timestamp.min()} to {part.timestamp.max()}')
    return groups


# 6. Fill missing values and encode categories (fit TRAIN ONLY)

class TabularPreprocessor:
    def fit(self, train):
        numeric = train[NUMERIC].replace([np.inf, -np.inf], np.nan)
        self.medians = numeric.median().fillna(0.0)
        self.categories = sorted(train['road_type'].fillna('Unknown').astype(str).unique().tolist())
        self.feature_names = NUMERIC + [f'road_type={v}' for v in self.categories]
        self.scaler = StandardScaler()
        self.scaler.fit(numeric.fillna(self.medians).astype('float32'))
        return self

    def transform(self, df):
        numeric = df[NUMERIC].replace([np.inf, -np.inf], np.nan)
        numeric = numeric.fillna(self.medians).astype('float32')
        scaled = self.scaler.transform(numeric).astype('float32')
        # Unknown road types in validation/test get an all-zero one-hot encoding.
        road = df['road_type'].fillna('Unknown').astype(str)
        onehot = np.column_stack([(road == value).to_numpy(dtype='float32')
                                  for value in self.categories]).astype('float32')
        return np.concatenate([scaled, onehot], axis=1)


# 7. PyTorch Dataset: load each image lazily, with tabular features + target

class TrafficDataset(Dataset):
    def __init__(self, frame, features, image_size=224, augment=False):
        self.paths = frame['resolved_image_path'].tolist()
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(frame[TARGET].to_numpy(dtype='float32', copy=True))
        self.image_size = image_size
        self._set_transform(image_size, augment)

    def _set_transform(self, image_size, augment):
        ops = [transforms.Resize((image_size, image_size))]
        # Do not horizontally flip road CCTV footage by default: direction matters.
        if augment:
            ops.append(transforms.ColorJitter(brightness=0.15, contrast=0.15))
        ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.transform = transforms.Compose(ops)

    def save(self, path):
        """Save processed features, targets, and image paths (not pixels)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'paths': self.paths,
            'features': self.features,
            'targets': self.targets,
            'image_size': self.image_size,
        }, path)

    @classmethod
    def load(cls, path, augment=False):
        """Restore a Dataset without rerunning Pandas preprocessing."""
        data = torch.load(path, map_location='cpu', weights_only=True)
        dataset = cls.__new__(cls)
        dataset.paths = data['paths']
        dataset.features = data['features']
        dataset.targets = data['targets']
        dataset.image_size = data['image_size']
        dataset._set_transform(dataset.image_size, augment)
        return dataset

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        try:
            with Image.open(path) as im:
                image = self.transform(im.convert('RGB'))
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f'Could not load CCTV image: {path}') from exc
        return {
            'image': image,                      # Float32 [3,H,W]
            'features': self.features[index],    # Float32 [feature_count]
            'target': self.targets[index],      # Float32 scalar, 0=no crash / 1=crash
        }


# 8. End-to-end pipeline: create train/validation/test datasets

def prepare_datasets(csv_path=None, image_root=None, image_size=224, save_preprocessor=None):
    check_feature_leakage()
    csv_path = Path(csv_path) if csv_path is not None else CSV_PATH
    image_root = Path(image_root) if image_root is not None else IMAGE_ROOT
    frame = load_and_clean(csv_path)
    frame = add_time_features(frame)
    frame = keep_existing_images(frame, image_root)
    train, val, test = chronological_split(frame)

    preprocessor = TabularPreprocessor().fit(train)
    train_ds = TrafficDataset(train, preprocessor.transform(train), image_size, augment=True)
    val_ds = TrafficDataset(val, preprocessor.transform(val), image_size)
    test_ds = TrafficDataset(test, preprocessor.transform(test), image_size)

    if save_preprocessor:
        destination = Path(save_preprocessor)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, destination)
        print(f'Saved train-fitted preprocessing to {destination}')

    print(f'Tabular feature count: {len(preprocessor.feature_names)}')
    print('Leakage guard: verified future crash label is LABEL ONLY; fit on TRAIN ONLY.')
    return train_ds, val_ds, test_ds, preprocessor


# 9. Save and reload processed PyTorch datasets

def save_datasets(train_ds, val_ds, test_ds, output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    for name, dataset in (('train', train_ds), ('val', val_ds), ('test', test_ds)):
        destination = output_dir / f'{name}.pt'
        dataset.save(destination)
        print(f'Saved {name} dataset to {destination}')


def load_saved_datasets(processed_dir=OUTPUT_DIR):
    """Reuse saved tensors and image paths; actual images load on demand."""
    processed_dir = Path(processed_dir)
    train_ds = TrafficDataset.load(processed_dir / 'train.pt', augment=True)
    val_ds = TrafficDataset.load(processed_dir / 'val.pt')
    test_ds = TrafficDataset.load(processed_dir / 'test.pt')
    return train_ds, val_ds, test_ds


# 10. Example: feed batches into a multimodal PyTorch model

def main():
    train_ds, val_ds, test_ds, prep = prepare_datasets(
        csv_path=CSV_PATH,
        image_root=IMAGE_ROOT,
        image_size=IMAGE_SIZE,
        save_preprocessor=OUTPUT_DIR / 'preprocessor.joblib',
    )

    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'feature_names.json').write_text(json.dumps(prep.feature_names, indent=2), encoding='utf-8')
    save_datasets(train_ds, val_ds, test_ds, out_dir)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    batch = next(iter(train_loader))
    print('Example batch shapes:', {key: tuple(value.shape) for key, value in batch.items()})
    print(f'Datasets ready: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}')
    # Model example:
    # logits = model(batch['image'], batch['features']).squeeze(-1)  # [batch_size]
    # loss = torch.nn.functional.binary_cross_entropy_with_logits(
    #     logits, batch['target']
    # )
    # Image preview (before normalization is easiest):
    # import matplotlib.pyplot as plt
    # plt.imshow(Image.open(train_ds.paths[0])); plt.show()


if __name__ == '__main__':
    main()
