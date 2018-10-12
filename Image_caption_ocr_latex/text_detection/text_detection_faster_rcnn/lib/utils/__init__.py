import sys
import os
sys.path.append(os.getcwd())

from . import bbox
from . import blob
import Cython
from . import boxes_grid
# from . import cython_nms
from . import cython_nms
from . import timer

try:
    from . import gpu_nms
except:
    gpu_nms = cython_nms
