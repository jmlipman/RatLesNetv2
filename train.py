from lib.RatLesNetv2 import RatLesNetv2
import itertools, os
import time
import numpy as np
from lib.losses import CrossEntropyDiceLoss
from lib.utils import he_normal, now
from lib.DataWrapper import DataWrapper
import argparse
import torch


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
# Input folder
parser.add_argument("--input", dest="input", default=-1)
# Validation folder
parser.add_argument("--validation", dest="validation", default=-1)
# Output folder
parser.add_argument("--output", dest="output", default=-1)
# Load training data into memory
parser.add_argument("--loadMemory", dest="loadMemory", default="-1")
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
# --output
if args.output == -1:
    raise Exception("Provide output forlder where the parameters and verbose will be saved (--output FOLDER)")
else:
    if not os.path.isdir(args.output):
        print("> Folder "+args.output+" does not exist. Creating folder...")
        os.makedirs(args.output)

# --validation (optional)
if args.validation != -1:
    if not os.path.isdir(args.validation):
        raise Exception("The provided validation folder '"+args.validation+"' does not exist or it's invalid")

# --loadMemory
if args.loadMemory != "1":
    print("> --loadMemory arg is not `1`. This means that the training data will not be stored in the memory")
    print("> In some cases, if the data is too much it cannot be saved into the memory, however this will")
    print("> cause the training to take considerably longer time.")
if args.loadMemory.isdigit():
    args.loadMemory = int(args.loadMemory)

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
print(now() + "Loading data")
train_data = DataWrapper(args.input, "train", device, loadMemory=args.loadMemory)
if args.validation != -1:
    val_data = DataWrapper(args.validation, "validation", device, loadMemory=args.loadMemory)
else:
    val_data = []

### Creating a new folder for the current run
outputPath = os.path.join(args.output, str(max([0] + [int(c) for c in os.listdir(args.output) if c.isdigit()])+1)) + "/"
os.makedirs(outputPath)

# Training Configuration
lr = 1e-4
epochs = 10
batch = 1
initW = he_normal
initB = torch.nn.init.zeros_
loss_fn = CrossEntropyDiceLoss
opt = torch.optim.Adam

# Architecture
filters = 32
modalities = 1

# POST-PROCESSING
#config["config.removeSmallIslands_thr"] = 20 # Remove independent connected components. Use 20. If not, -1

### Loading Weights
#config["config.model_state"] = "/home/miguelv/data/out/Lesion/Journal/3-ablation/level2_sameparams_mixed/2/model/model-699"
#config["config.model_state"] = ""


# Model
model = RatLesNetv2(modalities=modalities, filters=filters)
model.to(device)
opt = opt(model.parameters(), lr=lr)

# Weight initialization
def weight_init(m):
    if isinstance(m, torch.nn.Conv3d):
        initW(m.weight)
        initB(m.bias)
model.apply(weight_init)

e = 0 # Epoch counter
keep_training = True # Flag to stop training

print(now() + "Start training")
for e in range(epochs):

    # Train
    model.train()
    tr_loss = 0
    for tr_i in range(len(train_data)):
        X, Y, id_ = train_data[tr_i]

        output = model(X)
        pred = output[0]
        tr_loss_tmp = CrossEntropyDiceLoss(pred, Y)
        tr_loss += tr_loss_tmp

        opt.zero_grad()
        tr_loss_tmp.backward()
        opt.step()

    tr_loss /= len(train_data)

    with open(outputPath+"training_loss", "a") as f:
        tr_loss = str(tr_loss.cpu().detach().numpy())
        f.write(tr_loss+"\n")

    # Validation
    val_loss = ""
    if len(val_data) > 0:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_i in range(len(val_data)):
                X, Y, id_ = val_data[val_i]

                output = model(X)
                pred = output[0]
                val_loss += CrossEntropyDiceLoss(pred, Y)

        val_loss /= len(val_data)

        with open(outputPath+"validation_loss", "a") as f:
            val_loss = str(val_loss.cpu().numpy())
            f.write(val_loss+"\n")
            val_loss = "Val Loss: "+str(val_loss)

    print(now() + "Epoch: {}. Loss: {}. ".format(e, tr_loss) + val_loss)

    e += 1

# Save the model
torch.save(model.state_dict(), outputPath + "RatLesNetv2.model")
