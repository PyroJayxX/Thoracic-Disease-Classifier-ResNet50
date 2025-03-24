# Thoracic-Disease-Classifier
Thoracic Disease Classifier with ResNet-50 (via Transfer Learning)

> [!NOTE]
> - Model training/fine-tuning is not optimized YET; which makes this model very horribly, horrendously inaccurate — so use at your own risk.
> - Due to Github's file size limitations, I cannot upload the trained models to this repo.

## How to use the Notebook for Training
**Prepare the dataset**
1. Download the dataset from: https://www.kaggle.com/datasets/nih-chest-xrays/data/
2. Move it to where the directory of the notebook is (Needs to be at the same root)
3. Move all the images from **_images_001, images_002 ... images_012_** to a single folder **_images_** (You can do that using os library or something, or you can also just modify how the notebook handles finding the images)

**If ure using windows,**
1. Make a python venv with version 3.10
2. Download & Install Visual Studio 2019 w/ C++ Desktop Development
3. Download & Install CUDA 11.2
4. Download & Install cuDNN 8.1


**If ure using linux or mac**
1. Idk, check what you need by yourself here: https://www.tensorflow.org/install/source#tested_build_configurations
