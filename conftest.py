import os
import sys

# Make sure that the application source directory (this directory's parent) is
# on sys.path.

there = os.path.dirname(os.path.dirname(os.path.abspath(__file__)) + "/src/main/app/")
sys.path.insert(0, there)
print(sys.path)
