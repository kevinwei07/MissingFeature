# Missing-Feature-Learning-baseline3

# Goal
Predict the missing feature and the label in 1 model

Treat given input features as auxiliary labels and learn the target task by multi-task training

## How to run

* preprocess and train
`python main.py --arch my_test --do_train --do_plot`

* predict
`python main.py --arch my_test --do_predict`