"""
download_model.py — Downloads the MediaPipe HandLandmarker model file.

Run this once before starting the app:
    python download_model.py

The model file (hand_landmarker.task) is ~29MB and is not included in
the repository. It is downloaded from Google's official MediaPipe model
registry and saved to the project root.
"""

import urllib.request
import os

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


def download_with_progress(url: str, save_path: str) -> None:
    """Download a file from url to save_path, showing a simple progress bar."""

    def _reporthook(count, block_size, total_size):
        if total_size <= 0:
            print(f"\r  Downloading... {count * block_size // 1024} KB", end="")
            return
        percent = min(count * block_size * 100 // total_size, 100)
        downloaded_mb = count * block_size / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        bar = "█" * (percent // 5) + "░" * (20 - percent // 5)
        print(f"\r  [{bar}] {percent:3d}%  {downloaded_mb:.1f} / {total_mb:.1f} MB", end="", flush=True)

    print(f"Downloading MediaPipe HandLandmarker model...")
    print(f"  Source : {url}")
    print(f"  Target : {save_path}\n")

    urllib.request.urlretrieve(url, save_path, reporthook=_reporthook)
    print()  # newline after progress bar


def main():
    if os.path.exists(SAVE_PATH):
        size_mb = os.path.getsize(SAVE_PATH) / (1024 * 1024)
        print(f"✅ Model already exists at '{SAVE_PATH}' ({size_mb:.1f} MB). Nothing to do.")
        return

    try:
        download_with_progress(MODEL_URL, SAVE_PATH)
        size_mb = os.path.getsize(SAVE_PATH) / (1024 * 1024)
        print(f"\n✅ Model downloaded successfully! ({size_mb:.1f} MB)")
        print(f"   Saved to: {SAVE_PATH}")
        print("\nYou can now start the app with: python app.py")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("Please check your internet connection and try again.")
        if os.path.exists(SAVE_PATH):
            os.remove(SAVE_PATH)  # Clean up partial download
        raise


if __name__ == "__main__":
    main()
