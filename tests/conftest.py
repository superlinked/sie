import sys

# Remove paths that shadow the installed instructor package with the local repo copy
sys.path = [p for p in sys.path if "repos/instructor" not in p]
