########## PART 2 - BUILDING THE SEQ2SEQ MODEL ##########
 
 
 
# Creating placeholders for the inputs and the targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob
 
# Preprocessing the targets
# Left_size SOS token at the beginning of each answer inside each batch #1
# Targets must be into batches (10 answers at the time) without last token #2
# Concatenation left_size and right_size 
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets
 
# Creating the Encoder RNN function 
# Inputs : 
    # rnn_inputs -> inputs
    # rnn_size -> number of neurons of the LSTM cell 
    # num_layers in the RNN
    # keep_prob -> dropout regularization to the LSTM to improve the accuracy. 
    # dropout = 20% desactiovation of 20% of the neurons of each layer.
    # sequence_length = lenght of the question inside each batch. 
# Output : encoder_state
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                                    cell_bw = encoder_cell,
                                                                    sequence_length = sequence_length,
                                                                    inputs = rnn_inputs,
                                                                    dtype = tf.float32)
    return encoder_state
 
# Decoding the training set
# Inputs : 
    # encoder_state : input of the decoder 
    # decoder_cell : cell in the decoder 
    # decoder_embedded_input: embedding of the input --> conversion of the words in real vector 
    # sequence_lenght 
    # tf.variable_scope : advance data structure that wraps the tensor variable 
    # output_function
    # keep_prob 
    # batch_size
# Output : output_function(decoder_output_dropout)

def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
  # Mechanism of attention 
    # attention_keys : keys to be compared to the target state 
    # attention_values : values use to build the context vector 
    #---> the context vector is return by the encoder and use by the decoder as the first element of the decoding
    # attention_score_function : use to build the similarity between the keys and target state
    # attention_construct_function : function used to build the attention state.  
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)
 
# Decoding the test/validation set
# Inputs 
    # Four new inputs regarding 'Decode the training set' function : sos_id,eos_id,maximum_lenght,num_words 
    # decoder_embeddings_matrix instead of decoder_embedded_input
# Output : test_predictions
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    # attention_decoder_fn_inference instead of attention_decoder_fn_train
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions
 
# Creating the Decoder RNN
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
# step 1 -obtain output_function
# step 2 -use two functions : def decode_training_set and def decode_test_
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions
 
# Building the seq2seq model
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
# step 1 -encoder_embedded_input
# step 2 -apply the function "def encoder_rnn" -> Output : encoder_state
# step 3-apply the function "def preprocess_targets"-> Ouptput : preprocessed_targets
# step 4-decoder_embeddings_matrix
# step 5-decoder_embedded_input
# step 6-apply the function "def decoder_rnn"-> Outputs: training_predictions, test_predictions
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions
 