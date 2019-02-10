# Chatbot in TensorFlow 1.0

The example on how to implement simple chatbot using seq2seq model in the python using tensorflow 1.0 version. In this chatbot I used attention mechanism. If you want to get more information about seq2seq model read this blog : [here](https://seq2seq.blogspot.com/)

## Dataset

I've used the Cornell Movie Dialogs corpuse for this example. You can download it: [here](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) 

## Install

### &nbsp;&nbsp;&nbsp; Supported Python version
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Python version used in this project: 3.5

### &nbsp;&nbsp;&nbsp; Libraries used

> *  [Pandas](http://pandas.pydata.org) 0.18.0
> *  [Numpy](http://www.numpy.org) 1.10.4
> *  [TensorFlow](https://www.tensorflow.org) 1.0.0

## Code

All the core function of this chatbot is applied inside **model.py**.

Data preprocessing is inside  **preprocessing.py**.

If you want to train this chatbot execute **training.py**

## Run

After training this chatbot try running  test.py and test how it is working . I want to provide chatbot with trained checkpoint file but,

it will take some time. So, right only this much is available
