"""Download Google's versioned MediaPipe Holistic model into the ignored models directory."""
import argparse
from pathlib import Path
import urllib.request

URL = 'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('models/holistic_landmarker.task'))
    path = parser.parse_args().output
    if path.exists():
        print('Already present:', path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.download')
    try:
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open('wb') as target:
            import shutil
            shutil.copyfileobj(response, target)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print('Downloaded:', path)


if __name__ == '__main__':
    main()
