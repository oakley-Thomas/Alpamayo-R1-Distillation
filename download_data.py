import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
DATA_DIR = Path("/workspace/data")
DATA_DIR.mkdir(exist_ok=True)

files_to_download = [
    # 4 cameras Alpamayo needs
    "camera/camera_front_wide_120fov/camera_front_wide_120fov.chunk_0000.zip",
    "camera/camera_front_tele_30fov/camera_front_tele_30fov.chunk_0000.zip",
    "camera/camera_cross_left_120fov/camera_cross_left_120fov.chunk_0000.zip",
    "camera/camera_cross_right_120fov/camera_cross_right_120fov.chunk_0000.zip",
    # Egomotion labels (trajectory ground truth)
    "labels/egomotion/egomotion.chunk_0000.zip",
    # Calibration
    "calibration/camera_intrinsics/camera_intrinsics.chunk_0000.zip",
    "calibration/sensor_extrinsics/sensor_extrinsics.chunk_0000.zip",
    # Metadata (single parquet files, not chunked)
    "metadata/data_collection.parquet",
    "metadata/feature_presence.parquet",
    "clip_index.parquet",
]

for filename in files_to_download:
    print(f"Downloading {filename}...")
    local_path = hf_hub_download(
        repo_id=REPO,
        repo_type="dataset",
        filename=filename,
        local_dir=DATA_DIR,
    )
    if local_path.endswith(".zip"):
        print(f"  Extracting...")
        with zipfile.ZipFile(local_path, "r") as z:
            z.extractall(DATA_DIR / Path(filename).parent)

print("Done.")
