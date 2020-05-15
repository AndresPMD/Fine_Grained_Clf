# Fine-grained classification with textual cues

Implementation based in our paper: https://arxiv.org/pdf/2001.04732.pdf

## Install

Create Conda environment

    $ conda env create -f environment.yml

Activate the environment

    $ conda activate finegrained

Train from scratch

python3 train.py 

## Datasets:

Con-Text dataset can be downloaded from:
https://staff.fnwi.uva.nl/s.karaoglu/datasetWeb/Dataset.html

Drink-Bottle dataset:
https://drive.google.com/open?id=1ss9Pxr7rsdCpYX7uKjd-_1R4qCpUYTWT

## Textual Features

The results depicted in the paper were obtained by using the Fisher Vector of a set of PHOCs obtained from an image.
To extract the PHOCs, the following to repos can be used:

 https://github.com/DreadPiratePsyopus/Pytorch-yolo-phoc (Pytorch)
 https://github.com/lluisgomez/single-shot-str (Tensorflow)
 
Finally, the Fisher Vector out of the obtained PHOCs are used during training/inference time.

The Fisher Vector implementation was take from:
https://gist.github.com/danoneata/9927923

In the folder 'preproc' there is a script which does the following:
1) Create a PHOC dictionary.
2) Perform Scaling, Normalization, PCA of the PHOC dictionary.
3) Train a GMM based on the PHOC data (Takes aprox. 6000 seconds)
4) Given a PHOC result path with .txt files as PHOC predictions, reads each file and constructs the Fisher Vector to be used to train the model.

phocs_to_FV.py
