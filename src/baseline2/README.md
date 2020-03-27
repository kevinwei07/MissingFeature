# Missing-Feature-Learning-baseline2

# Goal:
Predict missing feature by the features we have,
and concat the missing with it as a new input to predict the label.

2-stage training
1. Feature predictor
2. Target task predictor

## How to run

* preprocess and train
`python main.py --arch my_test --do_train --do_plot`

* predict
`python main.py --arch my_test --do_predict`