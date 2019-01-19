#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 02:28:59 2019

@author: himanshu
"""

#Creating My first chatbot
 
#Importing the libraries
import numpy as np
import tensorflow as tf
import re 
import time
 
#Importing the dataset
lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore' ).read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore' ).read().split('\n')

#Creating a dictionary that maps each line and its id
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
         
#Cresting a list of all of the converesations
conversation_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'" , "").replace(" ","")
    conversation_ids.append(_conversation.split(","))
    
#Getting seperately question and answers 
questions = []
answers = []
for conversation in conversation_ids:
    for i in range(len(conversation)-1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
        
# Do a first cleaning of the texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm","i am",text)
    text = re.sub(r"he's","he is",text)
    text = re.sub(r"that's","he is",text)
    text = re.sub(r"she's","she is",text)
    text = re.sub(r"what's","what is",text)
    text = re.sub(r"it's","it is",text)
    text = re.sub(r"\'ll","will",text)
    text = re.sub(r"we'd","we had",text)
    text = re.sub(r"we're","we were",text)
    text = re.sub(r"can't","can not",text)
    text = re.sub(r"would'nt","would not",text)
    text = re.sub(r"should'nt","should not",text)
    text = re.sub(r"they're","they were",text)
    text = re.sub(r"they'd","they had",text)
    text = re.sub(r"\'re","are",text)
    text = re.sub(r"\'d","had",text)
    text = re.sub(r"\'ve","have",text)
    text = re.sub(r"[-()#/@;:<>{}+=~|.?,]","",text)
    return text
#Cleaning the questions 
clean_question = []
for question in questions:
    question = clean_text(question)
    clean_question.append(question)
#Cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))
#Creating a dictionary that maps each word to its number of occurences
words2count = {}
for questions in clean_question:
    for word in questions.split():
        if word not in words2count:
            words2count[word] = 1
        else:
            words2count[word]+=1
for answer in clean_answers:
    for word in questions.split():
        if word not in words2count:
            words2count[word] = 1
        else:
            words2count[word]+=1     
#Creating two dictionaries that maps question words and the answer words to a unique integer
threshold=20
questionwords2int = {}
word_number = 0
for word,count in words2count.items():
    if count>=threshold:
        questionwords2int[word]=word_number
        word_number+=1
answerwords2int = {}
word_number = 0
for word,count in words2count.items():
    if count>=threshold:
        answerwords2int[word]=word_number
        word_number+=1

#Adding the last tokens to these dictionaries
tokens =['<PAD>','<EOS>','<OUT>','<SOS>']
for token in tokens:
    questionwords2int[token]=len(questionwords2int)+1
for token in tokens:
    answerwords2int[token]=len(answerwords2int)+1            
#Creating the inverse dictionary of the answerwords2int
answerints2word={w_i:w for w,w_i in answerwords2int.items()}
#Adding the end of string token to the end of every answer    
for i in range(len(clean_answers)):
    clean_answers[i]+='<EOS>'
""" Translating all the questions and answers in to integers and replacing 
     all the words that were filtered out by<OUT>"""
question_to_int = []
for question in clean_question:
    ints = []
    for word in question.split():
        if word not in questionwords2int:
            ints.append(questionwords2int['<OUT>'])
        else:
            ints.append(questionwords2int[word])
    question_to_int.append(ints)

answer_to_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerwords2int:
            ints.append(answerwords2int['<OUT>'])
        else:
            ints.append(answerwords2int[word])
    answer_to_int.append(ints)
#Sorting questions and answers by the length of question
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1,25+1):
    for i in enumerate(question_to_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(question_to_int[i[0]])
            sorted_clean_answers.append(answer_to_int[i[0]])
            
 #Training The SEQ2SEQ model
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

#Defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()

#Loading the model inputs
inputs, targets, lr, keep_prob = model_input()

#Setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')

#Getting the shape of input tensor
input_shape = tf.shape(inputs)

#Getting the training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs[-1]),
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerwords2int),
                                                       len(questionwords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionwords2int)
#Setting up the loss error,the optimizer gradient clipping
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.Sequence_loss(training_predictions,targets,tf.ones([input_shape[0],sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [((tf.clip_by_value(grad_tensor),-5.,5.),grad_variable) for grad_tensor,grad_variable in gradients if grad_tensor is not None]
    optimize_gradient_clipping = optimizer.apply_gradients(clipped_gradients)
#Padding the sequence with the <PAD> token
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([(sequence) for sequence in batch_of_sequence]) 
    return [sequence+[word2int['<PAD>']]]