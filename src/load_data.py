import sys
import shutil

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    shutil.copy(src, dst)