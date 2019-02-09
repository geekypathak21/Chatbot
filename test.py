import tensorflow as tf
import preprocessing 
import model
import numpy as np

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

# Getting Question words to integers and answer words to integers dictionary
questionwords2int,answerwords2int = preprocessing.creating_dictionaries()

#Getting sorted clean questions and answers
sorted_clean_questions,sorted_clean_answers = preprocessing.sorted_clean_ques_ans()

#Defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()

#Loading the model inputs
inputs, targets, lr, keep_prob = model.model_inputs()

#Setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')

#Getting the shape of input tensor
input_shape = tf.shape(inputs)

#Getting the training and test predictions
training_predictions, test_predictions = model.seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
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
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,targets,tf.ones([input_shape[0],sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor,-5.,5.),grad_variable) for grad_tensor,grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)
#Padding the sequence with the <PAD> token
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences]) 
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]
#Splitting the data into batches of questions and answers
def split_into_batches(questions, answers, batchsize):
    for batch_index in range (0, len(questions)//batch_size):
        start_index = batch_index*batch_size
        question_in_batch = questions[ start_index : start_index+batch_size]
        answer_in_batch = answers[start_index : start_index+batch_size]
        padded_question_in_batch = np.array(apply_padding(question_in_batch, questionwords2int))
        padded_answers_in_batch = np.array(apply_padding(answer_in_batch, answerwords2int))
        yield padded_question_in_batch, padded_answers_in_batch 
#Splitting the question and answers into training and validation sets
training_validation_split = int(len(sorted_clean_questions)*0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answer = sorted_clean_answers[:training_validation_split] 


checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session,checkpoint)
answerints2word = preprocessing.answers_inverse_dictionary()
# Converting the questions from strings to lists of encoding integers
def converts_string2int(question,words2int):
    question,_ = preprocessing.clean_questions_answers(question)
    return [words2int.get(word,words2int['<OUT>']) for word in question.split()]

# Setting up the chat
while(True):
    question = input("you: ")
    if question == "Goodbye":
        break
    question = converts_string2int(question,questionwords2int)
    question = question +[questionwords2int['<PAD>']*(20-len(question))]
    fake_batch = np.zeros((batch_size,20))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions,{inputs:fake_batch,keep_prob:0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer,1):
        if answerints2word[i] == 'i':
            token = 'I'
        elif answerints2word[i] == '<EOS>':
            token = '.'
        elif answerints2word[i] == '<OUT>':
            token = 'out'
        else:
            token = ''+answerints2word[i]
            answer += token
        if token == '.':
            break
    print("Chatbot:-"+answer)
     