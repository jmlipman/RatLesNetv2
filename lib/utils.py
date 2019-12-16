from datetime import datetime
import torch
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt as dist
from skimage.measure import label

def np2cuda(inp, dev):
    return torch.from_numpy(inp.astype(np.float32)).to(dev)

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S: ")

def he_normal(w):
    """ He Normal initialization.
    """
    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
    return torch.nn.init.normal_(w, 0, np.sqrt(2/fan_in))

def removeSmallIslands(masks, thr=20):
    """Post-processing operation. It removes small islands and holes if the
       size of the cluster is lower than the threshold.
    """
    for m in range(masks.shape[0]):
        mask = np.argmax(masks[m], axis=0)
        # Clean independent components from the background
        labelMap = label(mask)
        icc = len(np.unique(labelMap))

        for i in range(icc): # From 1 because we skip the background
            if np.sum(labelMap==i) < thr:
                mask[labelMap==i] = 0

        # Clean independent components from the lesion
        labelMap = label(1-mask)
        icc = len(np.unique(labelMap))
        for i in range(icc): # From 1 because we skip the background
            if np.sum(labelMap==i) < thr:
                mask[labelMap==i] = 1

        masks[m,:,:,:,:] = np.stack([mask==0, mask==1], axis=0)*1.0

    return masks
