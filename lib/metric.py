import numpy as np
from skimage import measure
from scipy import ndimage

def _border_np(y):
    """Calculates the border of a 3D binary map.
       From NiftyNet.
    """
    return y - ndimage.binary_erosion(y)

def _border_distance(y_pred, y_true):
    """Distance between two borders.
       From NiftyNet.
       y_pred and y_true are WHD
    """
    border_seg = _border_np(y_pred)
    border_ref = _border_np(y_true)
    distance_ref = ndimage.distance_transform_edt(1 - border_ref)
    distance_seg = ndimage.distance_transform_edt(1 - border_seg)
    distance_border_seg = border_ref * distance_seg
    distance_border_ref = border_seg * distance_ref
    return distance_border_ref, distance_border_seg


class Metric:
    def __init__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

    def dice(self):
        """This function calculates the Dice coefficient.

           Args:
            `y_pred`: batch containing the predictions. BDWHC.
            `y_true`: batch containing the predictions. BDWHC.

           Returns:
            Dice coefficient. BC (B: batch, C: classes)
        """
        num_samples = self.y_pred.shape[0]
        num_classes = self.y_pred.shape[1]
        results = np.zeros((num_samples, num_classes))
        y_pred = np.argmax(self.y_pred, axis=1)
        y_true = np.argmax(self.y_true, axis=1)

        for i in range(num_samples):
            for c in range(num_classes):
                a = y_pred[i] == c
                b = y_true[i] == c
                if np.sum(b) == 0: # If no lesion in the y_true
                    if np.sum(a) == 0: # No lesion predicted
                        result = 1.0
                    else:
                        result = (np.sum(b==0)-np.sum(a))*1.0 / np.sum(b==0)
                else: # Actual Dice
                    num = 2 * np.sum(a * b)
                    denom = np.sum(a) + np.sum(b)
                    result = num / denom
                results[i, c] = result
        return results

    def hausdorff_distance(self):
        """Hausdorff distance.
           From NiftyNet.
           2-classes only!
        """
        num_samples = self.y_pred.shape[0]
        results = np.zeros(num_samples)
        for i in range(num_samples):
            y_pred = np.argmax(self.y_pred[i], axis=0)
            y_true = np.argmax(self.y_true[i], axis=0)

            ref_border_dist, seg_border_dist = _border_distance(y_pred, y_true)
            results[i] = np.max([np.max(ref_border_dist), np.max(seg_border_dist)])

        return results

    def compactness(self):
        """surface^1.5 / volume
        """
        num_samples = self.y_pred.shape[0]
        results = np.zeros(num_samples)
        for i in range(num_samples):
            pred = np.argmax(self.y_pred[i], axis=0)
            surface = np.sum(_border_np(pred))
            volume = np.sum(pred)
            results[i] = (surface**1.5)/volume
        return results
