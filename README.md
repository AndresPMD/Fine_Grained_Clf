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
