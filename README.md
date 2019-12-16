RatLesNetv2
======================


### Table of Contents
* [1. Introduction](#1-instroduction)
* [2. Installation](#2-installation)
* [3. Training and Evaluation](#3-training-and-evaluation)
* [4. Architecture](#4-architecture)
* [5. License](#5-license)
* [6. Citation](#6-citation)

### 1. Introduction
We aim to provide an easy framework to segment brain lesions in rodent brain scans.
RatLesNetv2 is using python and pytoch (which are open source)
The code of RatLesNetv2 was simplified 

### 2. Installation

1. Install Python (preferably version 3)
2. Create virtual environment
3. install dependencies
4. Download RatLesNetv2

RatLesNetv2 is a ConvNet for **rodent brain lesion segmentation**.
*kkk* _mm_ a

```cshell
virtualenv -p python3 FOLDER_FOR_ENVS/ve_tf_dmtf     # or python2
source FOLDER_FOR_ENVS/ve_tf_dmtf/bin/activate       # If using csh, source ve_tf_dmtf/bin/activate.csh

train.py --input DIR --output DIR --gpu X --load\_memory 1
test.py --input DIR --output DIR --gpu X --labels 1
```

### 3. Training and Evaluation

* Specifiy the number of modalities.
* Format of the data (nifti, HWDH).
* Install dependencies of pip freeze.

Training RatLesNetv2 can be done by:


### 4. Architecture

### 5. License
TODO

### 6. Citation
TODO
