from lib.RatLesNetv2 import RatLesNetv2
import itertools, os
import time
import numpy as np
from lib.utils import now, removeSmallIslands
from lib.DataWrapper import DataWrapper
import argparse
import torch
import nibabel as nib
from lib.metric import Metric


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
# Input folder
parser.add_argument("--input", dest="input", default=-1)
# Model file
parser.add_argument("--model", dest="model", default=-1)
# Output folder
parser.add_argument("--output", dest="output", default=-1)
# GPU. For CPU only use -1
parser.add_argument("--gpu", dest="gpu", default=0)
args = parser.parse_args()

### Check the user gave the mandatory arguments
# --input
if args.input == -1:
    raise Exception("Provide input folder where the data is located (--input FOLDER)")
else:
    if not os.path.isdir(args.input):
        raise Exception("The provided input folder '"+args.input+"' does not exist or it's invalid")

# --model
if args.model == -1:
    raise Exception("Provide the `model` file (--model FOLDER/RatLesNetv2.model)")
else:
    if not os.path.isfile(args.model):
        raise Exception("The provided input folder '"+args.model+"' does not exist or it's invalid")

# --output
if args.output == -1:
    raise Exception("Provide output forlder where the predictions will be saved (--output FOLDER)")
else:
    if not os.path.isdir(args.output):
        print("> Folder "+args.output+" does not exist. Creating folder...")
        os.makedirs(args.output)

# --gpu
if args.gpu >= torch.cuda.device_count():
    if torch.cuda.device_count() == 0:
        print("> No available GPUs. Add --gpu -1 to not use GPU. NOTE: This may take FOREVER to run.")
    else:
        print("> Available GPUs:")
        for i in range(torch.cuda.device_count()):
            print("    > GPU #"+str(i)+" ("+torch.cuda.get_device_name(i)+")")
    raise Exception("The GPU #"+str(args.gpu)+" does not exist. Check available GPUs.")

if args.gpu > -1:
    device = torch.device("cuda:"+str(args.gpu))
else:
    device = torch.device("cpu")

# Parsing the data
test_data = DataWrapper(args.input, "test", device, loadMemory=0)

### Creating a new folder for the current run
outputPath = os.path.join(args.output, str(max([0] + [int(c) for c in os.listdir(args.output) if c.isdigit()])+1)) + "/"
os.makedirs(outputPath)

# Architecture
filters = 32
modalities = 1

# Post-processing. Remove independent connected components. Use 20. If not, -1
removeSmallIslands_thr = 20

# Model
model = RatLesNetv2(modalities=modalities, filters=filters)
model.to(device)

# Loading model
model.load_state_dict(torch.load(args.model))

model.eval()
print(now() + "Start generating masks")
with torch.no_grad():
    for te_i in range(len(test_data)):
        if te_i % 10 == 0:
            print("Masks generated: {}/{}".format(te_i, len(test_data)))

        X, Y, id_ = test_data[te_i]

        output = model(X)
        pred = output[0].cpu().numpy() # BCDHW

        # Optional Post-processing
        if removeSmallIslands_thr != -1:
            pred = removeSmallIslands(pred, thr=removeSmallIslands_thr)

        if type(Y) != type(None):
            Y = Y.cpu().numpy()
            with open(outputPath + "stats.csv", "a") as f:
                measures = Metric(pred, Y)
                f.write("{},{},{},{}\n".format(id_, measures.dice()[0,1], measures.hausdorff_distance()[0], measures.islands()[0]))

        pred = pred[0] # CDHW
        pred = np.argmax(pred, axis=0) # DHW
        pred = np.moveaxis(pred, 0, -1) # HWD


        # The filename will be the name of the last 3 folders
        # Ideally Study_Timepoint_ScanID_predMask.nii.gz
        filename = "_".join(id_[:-1].split("/")[-3:])

        nib.save(nib.Nifti1Image(pred, np.eye(4)), outputPath + filename + "_predMask.nii.gz")



