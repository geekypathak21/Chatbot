
import re
# Importing the dataset
lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore' ).read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore' ).read().split('\n')
def get_conversation():
    # Creating a dictionary that maps each line and its id
    id2line = {}
    for line in lines:
        _line = line.split(' +++$+++ ')
        if len(_line) == 5:
           id2line[_line[0]] = _line[4]
         
    # Cresting a list of all of the converesations
    conversation_ids = []
    for conversation in conversations[:-1]:
        _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'" , "").replace(" ","")
        conversation_ids.append(_conversation.split(","))
    return conversation_ids,id2line    

    
def get_questions_and_answers():
    # Getting seperately question and answers 
    questions = []
    answers = []
    conversation_ids,id2line = get_conversation()
    for conversation in conversation_ids:
        for i in range(len(conversation)-1):
            questions.append(id2line[conversation[i]])
            answers.append(id2line[conversation[i+1]])
    return questions,answers    
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
def clean_questions_answers():
    questions,answers = get_questions_and_answers()
    # Cleaning the questions 
    clean_question = []
    for question in questions:
        question = clean_text(question)
        clean_question.append(question)
    # Cleaning the answers
    clean_answers = []
    for answer in answers:
        clean_answers.append(clean_text(answer))
    return clean_question,clean_answers    
def count_words():
    clean_question,clean_answers = clean_questions_answers()
    # Creating a dictionary that maps each word to its number of occurences
    words2count = {}
    for question in clean_question:
        for word in question.split():
            if word not in words2count:
               words2count[word] = 1
            else:
                words2count[word]+=1
    for answer in clean_answers:
        for word in answer.split():
            if word not in words2count:
               words2count[word] = 1
            else:
               words2count[word]+=1
    return words2count            
def creating_dictionaries():
    words2count = count_words()
    # Creating two dictionaries that maps question words and the answer words to a unique integer
    threshold_questions = 20
    questionwords2int = {}
    word_number = 0
    for word,count in words2count.items():
        if count>=threshold_questions:
           questionwords2int[word] = word_number
           word_number+=1
    threshold_answers = 20        
    answerwords2int = {}
    word_number = 0
    for word,count in words2count.items():
        if count>=threshold_answers:
           answerwords2int[word] = word_number
           word_number+=1
    # Adding the last tokens to these dictionaries
    tokens =['<PAD>','<EOS>','<OUT>','<SOS>']
    for token in tokens:
        questionwords2int[token]=len(questionwords2int)+1
    for token in tokens:
        answerwords2int[token]=len(answerwords2int)+1            
    return questionwords2int,answerwords2int   

def answers_inverse_dictionary():
    _,answerwords2int = creating_dictionaries()
    # Creating the inverse dictionary of the answerwords2int
    answerints2word={w_i:w for w,w_i in answerwords2int.items()}
    return answerints2word

def sorted_clean_ques_ans():
    clean_question,clean_answers = clean_questions_answers()
    questionwords2int,answerwords2int = creating_dictionaries()
    # Adding the end of string token to the end of every answer    
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
    # Sorting questions and answers by the length of question
    sorted_clean_questions = []
    sorted_clean_answers = []
    for length in range(1,25+1):
        for i in enumerate(question_to_int):
            if len(i[1]) == length:
               sorted_clean_questions.append(question_to_int[i[0]])
               sorted_clean_answers.append(answer_to_int[i[0]])
    return sorted_clean_questions,sorted_clean_answers           