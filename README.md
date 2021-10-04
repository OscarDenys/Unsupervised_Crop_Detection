Engineered an Unsupervised Deep Learning algorithm (using Python, Keras & TensorFlow) to discern locations of crops and weeds based on high resolution aerial pictures of agricultural fields. This resulted in an F1-score of 91.5%, reduced use of pesticides, eliminated need for labeled data, and was done using following steps:
• Unsupervised generation of a crop and a weed dataset.
• Training of two separate Autoencoders, one on each dataset.
• Prior estimation of crop and weed locations based on difference between reconstruction errors.
• Windowed Fourier Transform is used to detect crop rows and decrease chance of misclassifications.
