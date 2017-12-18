# Build ChatBot with Deep NLP  

###### Data Preprocessing######## 


# Import libraries 
import numpy as np 
import tensorflow as tf 
import re # clean the text 
import time 

# Import dataset 
lines = open('movie_lines.txt', encoding = 'utf_8', errors='ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf_8', errors='ignore').read().split('\n')


#Create a dictionary  that maps each line and its id 
# create inputs and outputs for the training. 
# split the line to keep the id of the conversation 'L1045' and the line of the conversation 'they do not'

id2line = { }
for line in lines : 
    _line = line.split(' +++$+++ ')
    if len(_line) == 5: # check the lenght of the line is 5 elements 
             id2line[_line[0]]=_line[4]


# Creating a list for all the conversation 
conversations_ids = []
for conversation in conversations[:-1]: # exclude the last empty row of the conversation dataset 
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", " ").replace(" ", "")
    conversations_ids.append(_conversation.split(','))
    # [-1] keep the last column, [1:-1] remove the square bracket, replace("'"," ") remove ''
    
    
# Getting separately the questions and the answers 
questions = []
answers = []
for conversation in conversations_ids : 
    for i in range (len(conversation)-1): # index = conversation lenght-1
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
        
# Doing a first cleaning of the text 
def clean_text(text): 
    text= text.lower() #put into lowercase
    text = re.sub(r"i'm","i am",text)
    text = re.sub(r"he's","he is",text)
    text = re.sub(r"she's","she is",text)
    text = re.sub(r"that's","that is",text)
    text = re.sub(r"what's","what is",text)
    text = re.sub(r"where's","where is",text)
    text = re.sub(r"\'ll"," will",text)
    text = re.sub(r"\'re"," have",text)
    text = re.sub(r"\'d"," would",text)
    text = re.sub(r"won't","will not",text)
    text = re.sub(r"can't","cannot",text)
    text = re.sub(r"can't","cannot",text)
    text = re.sub(r"[-()\"#/@;:<>{}+\.?,|]","",text)
    return text    

#Cleaning the questions 
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

#Cleaning the answers 
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))            

#Creating a dictionnary that maps each word to its number of occurence 
word2count= {}    
for question in clean_questions : 
    for word in question.split()  : 
         if word not in word2count: 
             word2count[word]=1 # first time 
         else:
            word2count[word]+=1 # increment
            
for answer in clean_answers : 
    for word in answer.split()  : 
         if word not in word2count: 
             word2count[word]=1
         else:
            word2count[word]+=1
            
#Creating two dictionnaries that map questions words and the answers word to a unique interger
threshold = 20

questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold : 
        questionswords2int[word]= word_number
        word_number += 1
        
answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold : 
        answerswords2int[word]= word_number
        word_number += 1
        
# Adding the last tokens to these two dictionaries        
tokens = ['<PAD>','<EOS>','<OUT>','<SOS>']
# PAD - your GPU (or CPU at worst) processes your training data in batches 
# EOS - "end of sentence" - the same as <end>
# and all the sequences in your batch should have the same length. 
# OUT words filtered out by the two dictionnaries answerswords2int questionswords2int
# SOS -"start of string"
for token in tokens: 
         questionswords2int[token]= len(questionswords2int)+1
         
for token in tokens: 
         answerswords2int[token]= len(answerswords2int)+1         
         
# Creating an inverse dictionnary of the  answerswords2int dictionary
answerints2word ={w_i: w for w,w_i in answerswords2int.items()} # w_i: w ==> w_i integer value, word 


#  Adding the End of string of the string to the end <EOS> of every answer 
for i in range (len(clean_answers)): 
    clean_answers[i]+=' <EOS>'

# Translating all the question and the answers into integers 
# Replacing all the world filtered by <OUT>
questions_into_int=[]
for question in clean_questions: 
    ints= []
    for word in question.split():
        if word not in questionswords2int: 
            ints.append(questionswords2int['<OUT>'])
        else : 
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)

answers_into_int=[]

for answer in clean_answers: 
    ints= []
    for word in answer.split():
        if word not in answerswords2int: 
           ints.append(answerswords2int['<OUT>'])
        else : 
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)

    
#Sorting questions and answers by the lenghts of questions 
# to speed up the training 
sorted_clean_questions=[]
sorted_clean_answers=[]
# questions from 1 word to 25 words 
for length in range (1, 25 + 1):
    for i in enumerate(questions_into_int): # inedx and the question 
        if len(i[1]) == length: 
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])
            