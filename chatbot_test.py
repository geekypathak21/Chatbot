import tensorflow as tf
import preprocessing 
import model


checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.saver()
saver.restore(session,checkpoint)


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
    question = question +[questionswords2int['<PAD>']*(20-len(question))]
    fake_batch = np.zeros((batch_size,20))
    fake_batch[0] = question
    predicted_answer = session.run(test_prediction,{inputs:fake_batch,kepp_prob:0.5})[0]
    