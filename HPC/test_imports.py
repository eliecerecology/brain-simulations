# save as test_imports.py
import numpy as np
import torch
import scipy
import networkx as nx
import tqdm
import matplotlib

print("numpy:", np.__version__)
print("torch:", torch.__version__)
print("scipy:", scipy.__version__)
print("networkx:", nx.__version__)
print("CUDA available:", torch.cuda.is_available())
print("All imports OK")
