# About Cats in Montenegro

![Alt text](https://www.whatwouldaithink.com/static//article1/IconArticle1.svg)

This repository contains jupyter notebooks and python files to train and compare different neural network with PyTorch.
The networks are trained to classify images with cats and dogs.
The result of the analysis is published on my website [whatwouldaithink.com](https://www.whatwouldaithink.com/articles/article1)

## Run the jupyter notebooks

### Clone the repository and install python libraries

Install Python 3.10.
I suggest to create a virtual environment.
Install the python libraries with pip.
The requirements file contain all libraries to run the jupyter
notebooks as well as pytest and pylint.

```bash
pip install -r requirements.txt
```

### Download the training and testing data from Kaggle
Log in and download the dataset from https://www.kaggle.com/datasets/charmz/cats-and-dogs.
Put the downloaded file archive.zip in the repository folder.

```bash
.
├── AnalyseColorImages
├── AnalyseColorImagesWithEfficientNetB0Model
├── AnalyseColorImagesWithFineTunedModels
├── AnalyseColorImagesWithoutBatchNorm
├── AnalyseColorImagesWithPreTrainedModels
├── AnalyseColorImagesWithResNet18Model
├── AnalyseColorImagesWithTwoClasses
├── AnalyseGrayScaleImages
├── AnalyseImagesWithEdges
├── plotting
├── printing
├── tests
├── vision
├── requirements_Python_3_10.txt
├── archive.zip
├── validate_sets
```


```bash
# unzip all datasets
# the validate_set is splitted in several parts but 7z automatically extract all parts
unzip archive.zip && 7z x validate_sets/validate_set.7z.001 && 7z x validate_sets/validate_set_cropped.7z

# move the validate sets to folder datasets
mv validate_sets/validate_set datasets && mv validate_sets/validate_set_cropped datasets

# delete the folder test, we dont need it
# rename the folder train to training_set
# rename the folder val to test_set. 
cd datasets && rm -r test && mv train training_set && mv val test_set

# rename the folder cat to cats and dog to dogs in the folders training_set and test_set
mv training_set/cat training_set/cats && mv training_set/dog training_set/dogs
mv test_set/cat test_set/cats && mv test_set/dog test_set/dogs
```

The final folder datasets looks like this.
```bash
.
├── datasets
│   ├── test_set
│   │   ├── cats
│   │   └── dogs
│   ├── training_set
│   │   ├── cats
│   │   └── dogs
│   ├── validate_set
│   │   └── cats
│   └── validate_set_cropped
│       └── cats
```

### Move the folder datasets to the folder with the notebook
For example if you want to run the notebook AnalyseColorImages.ipynb copy
the datasets to the folder AnalyseColorImages.

```bash
.
├── checkpoints
├── datasets
│   ├── test_set
│   │   ├── cats
│   │   └── dogs
│   ├── training_set
│   │   ├── cats
│   │   └── dogs
│   ├── validate_set
│   │   └── cats
│   └── validate_set_cropped
│       └── cats
├── AnalyseColorImages.ipynb
├── CompareTestAndTrainImages.ipynb
```

### Unzip the checkpoints
For example if you want to run the notebook AnalyseColorImages.ipynb go to the folder AnalyseColorImages.
Unzip all checkpoints and move them to the parent folder.

```bash
.
├── checkpoints
├── datasets
├── AnalyseColorImages.ipynb
├── checkpoint_ColorImagesAugmented_best.pt
├── checkpoint_ColorImagesAugmented.pt
├── checkpoint_ColorImages_best.pt
├── checkpoint_ColorImages.pt
├── CompareTestAndTrainImages.ipynb
```

Now you can run the jupyter notebooks.


## Train a model
For training the models there are two options:
+ resume/continue the training
+ start the training from scratch

In every notebook there is a cell like this:

```python
# Train the model
checkpoint="ColorImages"
tm = TrainingManager(lr=0.001, model=model, nameCheckpoint=checkpoint, 
                     batchSize=32, criterion=model.criterion, 
                     datasets={"train": dataSetTrain, 
                      "test": dataSetTest,
                      "vali": dataSetValidate,
                      "valiCrop": dataSetValidateCropped},
                      title="Classification Of\n Color Images")

n_epochs = 0

tm.resume(n_epochs)
```

To resume/continue the training enter the number of epochs to run e.g. for 10 epochs ``` n_epochs = 10 ```. 
Be sure the checkpoints are unziped.
The checkpoints will be read and extended with every epoch.

To train from scratch a new checkpoint must be created and ```train()``` method of the ```TrainingManager``` must be called.
With every epoch the checkpoints will be extended.

```python
# Train the model
checkpoint="newCheckpoint"
tm = TrainingManager(lr=0.001, model=model, nameCheckpoint=checkpoint, 
                     batchSize=32, criterion=model.criterion, 
                     datasets={"train": dataSetTrain, 
                      "test": dataSetTest,
                      "vali": dataSetValidate,
                      "valiCrop": dataSetValidateCropped},
                      title="Classification Of\n Color Images")

n_epochs = 10

tm.train(n_epochs)
```
# License 
The Jupyter Notebooks, the Validation Sets and the Source Code are licensed under the following license.

## License for the Jupyter Notebooks

The Jupyter Notebooks are licensed under a
[Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License][cc-by-nc-nd].

[![CC BY-NC-ND 4.0][cc-by-nc-nd-shield]][cc-by-nc-nd]

[![CC BY-NC-ND 4.0][cc-by-nc-nd-image]][cc-by-nc-nd]

[cc-by-nc-nd]: http://creativecommons.org/licenses/by-nc-nd/4.0/
[cc-by-nc-nd-image]: https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png
[cc-by-nc-nd-shield]: https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg

## License for the Validation Sets

The Validation Sets are licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

## License for the Source Code

The Source Code is licensed under a [BSD-3-Clause](https://opensource.org/license/BSD-3-Clause).

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
