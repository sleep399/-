from pathlib import Path
import logging
import urllib.request

logger = logging.getLogger(__name__)

MODELS = {
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    # CCPD RPNet fh02.pth (official pretrained, ECCV 2018)
    "fh02.pth": "https://drive.google.com/uc?export=download&id=1YYVWgbHksj25vV6bnCX_AWokFjhgIMhv&confirm=t",
    "vehicle_detector.pt": "",
}


def _download_file(url: str, dest: Path) -> None:
    logger.info("Downloading model %s -> %s", url, dest)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(dest, "wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    logger.info("Download complete: %s (%.1f MB)", dest.name, dest.stat().st_size / 1024 / 1024)


def get_model_path(name: str) -> str:
    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / name
    if path.exists() and path.stat().st_size > 1024:
        return str(path)
    url = MODELS.get(name)
    if not url:
        raise FileNotFoundError(f"Model file missing: {name}. Please place it in {path}")
    try:
        _download_file(url, path)
    except Exception as exc:
        if path.exists():
            path.unlink(missing_ok=True)
        raise FileNotFoundError(
            f"无法自动下载 {name}。请从 CCPD-master README 手动下载 fh02.pth 并放到: {path}"
        ) from exc
    return str(path)
