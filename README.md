# Galaxy morphology prediction using capsule networks
We implemented our capsule net based on the code provided in [here](https://github.com/gram-ai/capsule-networks). The paper is submitted to MNRAS and you can find it on arXiv at [arXiv:1809.08377](https://arxiv.org/abs/1809.08377).
## 1 Required Packages
1. python 3.6
2. numpy
3. scipy
4. [opencv](https://opencv.org/)
5. pandas
6. CUDA
6. [pytorch](https://pytorch.org/)
7. [torchnet](https://github.com/pytorch/tnt)
8. [tqdm](https://github.com/tqdm/tqdm)
9. sklearn

## 2 System Requirement
We used one NVIDIA Tesla P100 GPU unit along with 4 CPUs on Owen cluster at Ohio Supercomputer Center (OSC) with 16Gb of memory.

## 3 Data Preprocessing
Read the data preprocessing part in the paper. To crop and downsample the dataset from [Kaggle](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge) simply download the dataset to a folder and use the **data_preprocess.py** provided in **preporcess** folder. The preprocessed datasets are provided here.

## 4 Folder Structure
Each folder contains the following structure:
- folder/
    - code/
    - data/
    - results/
    - epochs/ (only for CapsNet)

## 5 Classification based on answers to question 1
### 5.1 Baseline CNN
The **CNN_Baseline_Morph_3_Drop_2fc** folder contains the code for this scenario. The datasets have been provided.

### 5.2 Capsule network
The Morph_2_new folder contains the code and data is the same as provided in baseline model. You can copy and paste it in the dataset folder here. The **epochs** folder is where the model after each epoch of training will be saved and if you want to restart your training from an specific epoch you can load the model and restart your training.

## 6 Regression
## 6.1 Baseline CNN
The **CNN_Baseline_Reger** folder contains the codes and the dataset.

## 6.2 Capsule Network
The **Morph_Reger** folder contains the code and the dataset is the same as provided in baseline model.
