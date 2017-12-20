#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 11:32:52 2017

@author: magalidrumare
"""
import tensorflow as tf


## PART 2 : BUILD THE SEQ2SEQ MODEL ##
 
#1_Creating placeholders for inputs and targets 
def model_inputs():
    # 2 dimensional inputs 
   inputs = tf.placeholder(tf.int32,[None,None],name='input') 
   targets = tf.placeholder(tf.int32,[None,None],name='target') 
   lr = tf.placeholder(tf.float32,name='learning_rate') 
   keep_prob = tf.placeholder(tf.float32,name='keep_prob')       
   return  inputs, targets, lr, keep_prob 



#2_'Preprocessing the targets' function : 
# Left_size SOS token at the beginning of each answer inside each batch #1
# Targets must be into batches (10 answers at the time) without last token #2
# Concatenation left_size and right_size 
   
def preprocess_targets(targets,word2int,batch_size):
    left_size = tf.fill([batch_size,1],word2int['<SOS>']) #1
    right_size = tf.strided_slice(targets,[0,0],[batch_size,-1],[1,1]) #2
    processed_targets = tf.concat([left_size, right_size],1)
    return processed_targets


#3_Encoder RNN Layer(LSTM) function : 
# Inputs of the function : 
    # rnn_inputs -> model_inputs
    # rnn_size -> number of neurons of the LSTM cell 
    # num_layers in the RNN
    # keep_prob ->dropout regularization to the LSTM to improve the accuracy
    # dropout = 20% desactiovation of 20% of the neurons of each layer.
    # lenght of the question inside each batch 
# Output of the function : encoder_state

def encoder_rnn_layer(rnn_inputs,rnn_size, num_layers,keep_prob, sequence_lenght): 
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout], num_layers)
    #_ : encoder_output 
    # only need the encoder_state as an input of decoder 
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                      cell_bw = encoder_cell,
                                                      sequence_lenght = sequence_lenght,
                                                      inputs= rnn_inputs, 
                                                      dtype = tf.float32)
    return encoder_state



# 4_'Decoding the training set' function : 
# Inputs:    
  # encoder_state : input of the decoder 
  # decoder_cell : cell in the NN decoder 
  # decoder_embedded_input: embedding of the input --> conversion of the words in real vector 
  # sequence_lenght 
  # tf.variable_scope : advance data structure that wraps the tensor variable tensorflow 1.4
  # output_function
  # keep_prob 
  # batch_size
# Output : output_function(decoder_output_dropout)
  
def decode_training_state(encoder_state,decoder_cell, decoder_embedded_input,sequence_lenght,decoding_scope,
                          output_function,keep_prob,batch_size):
    attention_states = tf.zeros([batch_size,1,decoder_cell.output_size])
# attention_keys : keys to be compared to the target state 
# attention_values : values use to build the context vector 
#---> the context vector is return by the encoder and use by the decoder as the first element of the decoding
# attention_score_function : use to build the similarity between the keys and target state
# attention_construct_function : function used to build the attention state 
    attention_keys,attention_values,attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(
            attention_states, attention_option= 'bahdanau', num_units = decoder_cell.output_size)

# function that decode the training set    
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0], 
                                                                            attention_keys,
                                                                            attention_values,
                                                                            attention_score_function,
                                                                            attention_construct_function,
                                                                            name ='attn_dec_train')
 # _ : decoder final state, _ : decoder_final_context_state  
 # only need the decoder_output                             
    decoder_output, _,_ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                 training_decoder_function, 
                                                                 decoder_embedded_input,
                                                                 sequence_lenght,
                                                                 scope=decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output,keep_prob)
    return output_function(decoder_output_dropout)


# 5_Decoding the test/ validation set function 
# Inputs 
    # Four new inputs regarding 'Decoding the training set' function : 
    #sos_id,eos_id,maximum_lenght,num_words 
# Output : test_predictions
    
def decode_test_state(encoder_state,decoder_cell, decoder_embeddedings_matrix,sos_id,eos_id,maximum_lenght,num_words,sequence_lenght,
        decoding_scope,output_function,keep_prob,batch_size):
                      
    attention_states = tf.zeros([batch_size,1,decoder_cell.output_size])


    attention_keys,attention_values,attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(
            attention_states, attention_option= 'bahdanau', num_units = decoder_cell.output_size)

#  new function to use : tf.contrib.seq2seq.attention_decoder_fn_inference
#  add the new inputs : decoder_embeddedings_matrix,sos_id,eos_id,maximum_lenght,num_words
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0], 
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddedings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_lenght,
                                                                              num_words,
                                                                              name ='attn_dec_test')
 # _ : decoder final state, _ : decoder_final_context_state  
 # only need the decoder_output                             
    test_predictions, _,_ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                 test_decoder_function, 
                                                                 scope = decoding_scope)

    return test_predictions


# 6_Create the Decoder RNN function: 
# Inputs : 
    # decoder_embedded_input, 
    #decoder_embeddings_matrix, 
    #encoder_state, 
    #num_words, 
    #sequence_lenght,
    #rnn_size,num_layers,
    #word2int,
    #keep_prob,
    #batch_size
# Output :  training_predictions, test_predictions
# a-obtain output_function
# b-use two functions : def decode_training_state and def decode_test_state 
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, 
                sequence_lenght,rnn_size,num_layers,word2int,keep_prob,batch_size):
        with tf.variable_scope("decoding") as decoding_scope: 
            lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
            lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob = keep_prob)
            decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout], num_layers)
            weights = tf.trucated_normal_initializer(stddev=0.1)
            biases = tf.zeros_initializer()
            output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                          num_words,
                                                                          None,
                                                                          scope= decoding_scope,
                                                                          weights_initializers = weights,
                                                                          bias_initializers = biases)

        training_predictions = decode_training_state(encoder_state,
                                                       decoder_cell, 
                                                       decoder_embedded_input,
                                                       sequence_lenght,
                                                       decoding_scope,
                                                       output_function,
                                                       keep_prob,
                                                       batch_size)
    
    
    
        decoding_scope.reuse_variable()
            
        test_predictions = decode_test_state(encoder_state,
                                                decoder_cell, 
                                                decoder_embeddings_matrix,
                                                word2int['SOS'],
                                                word2int['EOS'],
                                                sequence_lenght-1,
                                                num_words,
                                                sequence_lenght,
                                                decoding_scope,
                                                output_function,
                                                keep_prob,
                                                batch_size)
         
            
        return training_predictions, test_predictions 


#7_'Building the seq2seq model function': 
# Inputs : 
    #inputs,targets, 
    #keep_prob, 
    #batch_size, 
    #sequence_lenght,
    #answers_num_words,
    #questions_num_words,
    #encoder_embedding_size, 
    #decoder_embedding_size,
    #rnn_size,
    #num_layers,
    #questionsword2int
# Outputs : training_predictions, test_predictions
# a-encoder_embedded_input
# b-apply the function "def encoder_rnn_layer" -> Output : encoder_state
# c-apply the function "def preprocess_targets"-> Ouptput : preprocessed_targets
# d-decoder_embeddings_matrix
# e-decoder_embedded_input
# f-apply the function "def decoder_rnn"-> Outputs: training_predictions, test_predictions
    

def seq2seq_model(inputs,targets, keep_prob, batch_size, sequence_lenght,answers_num_words,questions_num_words,encoder_embedding_size, decoder_embedding_size,rnn_size,num_layers,questionsword2int): 
    
# encoder embbeded input 
    encoder_embedded_input = tf.contrib.layer.embed_sequence(inputs,
                                                         answers_num_words +1, 
                                                         encoder_embedding_size,
                                                         initializer = tf.random_uniform_initializer(0,1)) 

# apply the function "def encoder_rnn_layer(rnn_inputs,rnn_size, num_layers,keep_prob, sequence_lenght): "
# rnn_inputs = encoder_embedded_input
    encoder_state = encoder_rnn_layer(encoder_embedded_input,rnn_size, num_layers, keep_prob,sequence_lenght) 

# apply the function "def preprocess_targets(targets,word2int,batch_size):"
    preprocessed_targets = preprocess_targets(targets,questionswords2int,batch_size)

# decoder embedding_matrix 
# number of lines : question_num_words+1
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words+1,decoder_embedding_size],0,1))
     
# decoder embedded input 
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix,preprocessed_targets)
    
# applt the function : def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, 
                #sequence_lenght,rnn_size,num_layers,word2int,keep_prob,batch_size):
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_lenght, 
                                                         rnn_size,
                                                         num_layers,
                                                         questionsword2int,
                                                         keep_prob, 
                                                         batch_size)
    return training_predictions, test_predictions
        
    
