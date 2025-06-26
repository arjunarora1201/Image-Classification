# Image-Classification

Loaded dataset from tensofrlow --> Cifar10 dataset, has 60000 images of size 32*32 pixels, 6000 images per class
Preprocessing --> Extracted features like color, gabor, HOG etc before applying PCA
Training --> Chose Support Vector Machine as it gave the highest accuracy of 65% which is quite good in multi class image classification using ML only.

Model Saved, built a Python Flask app which takes static images as input and classifies them.
