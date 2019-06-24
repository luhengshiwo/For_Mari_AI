# Mulit Text Classification with Text-cnn

## Requirements
```
python3
tensorflow >= 1.13
```
## Task
Given a sentence, assign a label according to its' content.
```
i liked the Da Vinci Code a lot. --- 1
```

## data & preprocess
The data provided in the `data/` directory is a csv file

In `data_util.py` I provide some funtions to process the csv file.

## Usage
This contains several steps:
1. Before you can get started on training the model, you mast run
```
python data_util.py
```

2. After the dirty preprpcessing jobs, you can try running an training experiment with some configurations by:
```
python main.py
```

3. You can also run an evaluation by:
```
python server.py 
```

Then follow the instruction. Hope you enjoy it.

## Reference 
[A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1510.03820.pdf).

[Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759v2.pdf).

[Recurrent Convolutional Neural Networks for Text Classification](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745/9552).

[Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf).

## Folder Structure
```
├── data            - this fold contains all the data
│   ├── train
├── model           - this fold contains the pkl file to restore
├── main.py         - main structure of the project
├── methods.py      - all methods we use 
├── data_util.py    - preprocess the data
├── load_data.py    - data generator
├── server.py       - server
```

## To do
1. Still need parameters searching.
2. Need structure changing to satisfy parameters chosing.
3. Make codes nicer.
