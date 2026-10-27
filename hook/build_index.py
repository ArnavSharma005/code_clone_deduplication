"""Index all Java methods in a project: python hook/build_index.py <project_dir>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.hook_engine import build_index

if __name__ == "__main__":
    proj = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"Indexed {build_index(proj)} methods in {proj}/.clone_index")
