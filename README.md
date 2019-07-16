# MNIST-Fashion-Classifier

This neural network can classify based on the MNIST Fashion data set. 
This program was created for the Secure and Private AI Scholarship Challenge from Udacity and Facebook
To train the model run the fashion classifier file.

# Test out your network!

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import helper

dataiter = iter(testloader)
images, labels = dataiter.next()
img = images[1]

# TODO: Calculate the class probabilities (softmax) for img
ps = torch.exp(model(img))

# Plot the image and probabilities
helper.view_classify(img, ps, version='Fashion')

