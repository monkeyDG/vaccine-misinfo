from pathlib import Path
import os 
import sys

home = Path(__file__).parent
FAVICON_PATH = os.path.join(home, "assets", "favicon.png")
print(FAVICON_PATH)
