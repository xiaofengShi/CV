
import sys
import os
from .config import cfg
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from lib.utils.cython_nms import nms as cython_nms

try:
    from lib.utils.gpu_nms import gpu_nms
    # from ..utils.gpu_nms import gpu_nms
except:
    gpu_nms = cython_nms
pass


def nms(dets, thresh):
    if dets.shape[0] == 0:
        return []
    if cfg.USE_GPU_NMS:
        try:
            return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
        except:
            return cython_nms(dets, thresh)
    else:
        return cython_nms(dets, thresh)
