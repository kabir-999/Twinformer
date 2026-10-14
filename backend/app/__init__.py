import os

# torch and faiss each bundle an OpenMP runtime; on macOS loading both aborts the process on the first FAISS search.
# This is the standard (if blunt) workaround. It must be set before torch or faiss are imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
