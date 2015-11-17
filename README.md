# caffe_changed_accuracy_layer
Changed Accuracy Layer to Calculate Dice Metric 


The current accuracy layer of Caffe is suitable for classification tasks. For other tasks, you need to change 
the accuracy layer according to your needs or remove the accuracy layer from your.prottext file. If you remove the 
accuracy layer, the only indicator of training progress is the loss value. 

Here, for our segmentation task, we need to calculate the Dice coefficient on the test and training data while traing our network. 
This gives us a sense of the training progress in addition to the loss value.

The process is as follows:

1- Update file accuracy_layer.cpp located in ./caffe/src/caffe/layers as you need.
Here I updated the file to calculate the dice coefficient. 

2- Then in the terminal: 

./caffe$ make all 

This will update accuracy_layer.o file located in ./caffe/build/src/caffe/layers

3- you would probably need to change some of the parameter in the solver.prototext file depending on the size of 
your test and training data. For instance, if you have 500 images in the test data and batch_size=100
you can set:

#covering the full 500 testing images (500=100*5).
test_iter: 5

Also, you can tell Caffe how often perform test and show you accuracy value on test data.
#Carry out testing every 50 training iterations.
test_interval: 50

4- In case you want to see the accuracy values for both training and test data, you can add the layers:
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "output"
  bottom: "label"
  top: "accuracy_train"
  include {
    phase: TRAIN
  }
}

layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "output"
  bottom: "label"
  top: "accuracy_test"
  include {
    phase: TEST
  }
}


Now, if you train the network, you should be able to see the accuracy an loss value for both training and test data.











