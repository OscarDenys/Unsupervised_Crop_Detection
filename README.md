Engineered an Unsupervised Deep Learning algorithm (using Python, Keras & TensorFlow) to discern locations of crops and weeds based on high resolution aerial pictures of agricultural fields. This resulted in an F1-score of 91.5%, reduced use of pesticides, eliminated need for labeled data, and was done using following steps:
[1] Unsupervised generation of a crop and a weed dataset.
[2] Training of two separate Autoencoders, one on each dataset.
[3] Prior estimation of crop and weed locations based on difference between reconstruction errors.
[4] Windowed Fourier Transform (WFT) is used to detect crop rows and decrease chance of misclassifications for large interrow weeds.


The Python script before_matlab takes care of the prior estimation. This is loaded into Matlab and the final_algorithm Matlab script then performs a WFT transform. The output of this WFT is then loaded back into the after_matlab Python script and incorporated in the final crop detection. 


