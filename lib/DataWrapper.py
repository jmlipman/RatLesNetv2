import torch, os, random
import nibabel as nib
import numpy as np
from lib.utils import np2cuda

class DataWrapper:
    def __init__(self, path, stage, dev, loadMemory):
        """This class finds and parses the data in the provided part.
           Scans will be zero-centered and normalized to have variance of 1.

           FORMAT of the files.
           > This script uses nibabel to open images. I recommend NIfTI
             files gzip compressed i.e. scan.nii.gz.
           > Images and labels must be in the same folder.
           > All images must have the same name (variable `scanName`).
           > All labels must have the same name (variable `labelName`).
           > Labels will have values of 0s (background) and 1s (lesion).
           > Image files will have the following size:
               Height x Width x Slices x Channels. For instance: 256x256x18x1
           > Labels will have the following size:
               Height x Width x Slices. For instance: 256x256x18

           Example of `path` structure:

           `path`
            └─Study 1
              └─24h (time-point)
                ├─32 (id of the scan)
                │ ├─scan.nii.gz (image)
                │ └─scan_lesion.nii.gz (label)
                └─35
                  ├─scan.nii.gz
                  └─scan_lesion.nii.gz

           Args:
            `path`: location of the training data. It must follow the format
             described above.
            `stage`: either "train" or "eval". In "eval", labels are optional.
            `dev`: device where the data will be loaded (cpu, gpu..)
            `loadMemory`: for training, loading the dataset into the RAM speeds
             up the execution time.
             Do this ONLY IF you know your data fits your RAM.

        """
        self.stage = stage
        self.dev = dev
        self.loadMemory = loadMemory

        self.scanName = "scan.nii.gz"
        self.labelName = "scan_lesion.nii.gz"

        self.list = []
        for root, subdirs, files in os.walk(path):
            if self.scanName in files:
                if self.stage == "train" and not self.labelName in files:
                    continue #REMOVE THIS
                    raise Exception("Parsing scans for training, but I couldn't find a label file called `"+self.labelName+"` in `"+root+"`")
                self.list.append(root + "/")
        if len(self.list) == 0:
            raise Exception("I couldn't find any `"+self.scanName+"` file in `"+path+"`")

        random.shuffle(self.list)

        if self.loadMemory and (self.stage == "train" or self.stage == "validation"):
            self.dataX = []
            self.dataY = []
            self.dataId = []
            for i in range(len(self.list)):
                X, Y, id_ = self._loadSubject(i)
                self.dataX.append(X)
                self.dataY.append(Y)
                self.dataId.append(id_)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        if self.loadMemory and (self.stage == "train" or self.stage == "validation"):
            return self.dataX[idx], self.dataY[idx], self.dataId[idx]
        else:
            X, Y, id_ = self._loadSubject(idx)
            return X, Y, id_


    def _loadSubject(self, idx):
        
        """This function loads a single subject.

           Args:
            `idx` Index of self.list of the subject that will be loaded.

           Returns:
            `X`: brain scan normalized to have 0-mean 1-std.
            `Y`: labels (0s are the background, 1s are the lesion).
            `id_`: id/name (path) of the scan.

        """
        target = self.list[idx]
        X = nib.load(target + self.scanName).get_data()
        X = (X-X.mean())/X.std() # 0-mean 1-std normalization
        X = np.moveaxis(X, -1, 0) # Move channels to the first axis.
        X = np.moveaxis(X, -1, 1) # Move "depth" to the second axis.
        X = np.expand_dims(X, axis=0) # BCDHW format
        X = np2cuda(X, self.dev)

        try:
            Y = nib.load(target +  self.labelName).get_data()
            Y = np.moveaxis(Y, -1, 0) # DHW format
            Y = np.stack([1.0*(Y==j) for j in range(2)], axis=0) # CDHW format
            Y = np.expand_dims(Y, 0) # BCDHW format
            Y = np2cuda(Y, self.dev)

        except FileNotFoundError:
            # This can only happen during "eval" mode.
            # If it happens, we cannot evaluate how well the predictions are
            # but the script will generate the predictions anyway.
            
            Y = None

        return X, Y, target

