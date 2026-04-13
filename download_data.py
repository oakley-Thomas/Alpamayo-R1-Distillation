from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
DATA_DIR = Path("/workspace/data")
DATA_DIR.mkdir(exist_ok=True)

files_to_download = [
    # Index and feature map (required by PhysicalAIAVDatasetLocalInterface)
    "clip_index.parquet",
    "features.csv",
    # Metadata
    "metadata/data_collection.parquet",
    "metadata/feature_presence.parquet",
    # 4 cameras Alpamayo needs — kept as zips, the data loader reads them directly
    "camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0000.zip",
    "camera/camera_front_tele_30fov/camera_front_tele_30fov.chunk_0000.zip",
    "camera/camera_cross_left_120fov/camera_cross_left_120fov.chunk_0000.zip",
    "camera/camera_cross_right_120fov/camera_cross_right_120fov.chunk_0000.zip",
    # Egomotion labels — kept as zip, the data loader reads it directly
    "labels/egomotion/egomotion.chunk_0000.zip",
    # Calibration (parquet per chunk, not zipped)
    "calibration/camera_intrinsics/camera_intrinsics.chunk_0000.parquet",
    "calibration/sensor_extrinsics/sensor_extrinsics.chunk_0000.parquet",
]

for filename in files_to_download:
    print(f"Downloading {filename}...")
    hf_hub_download(
        repo_id=REPO,
        repo_type="dataset",
        filename=filename,
        local_dir=DATA_DIR,
    )

print("Done. Data directory:", DATA_DIR)
