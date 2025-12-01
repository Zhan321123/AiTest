import sys
from pathlib import Path

Root = Path(__file__).resolve().parent  # root path
if Root not in sys.path:
    sys.path.append(str(Root))

if __name__ == '__main__':
    f = Root / './weight/all-MiniLM-L6-v2'
    print(f.exists())