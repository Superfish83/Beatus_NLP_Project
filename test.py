import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

TIME_STEPS = 20
INPUT_DIM  = 2

def attention_3d_block(inputs):
    
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    
    a = tf.keras.layers.Permute((2, 1))(inputs) # same transpose
    #a = tf.keras.layers.Reshape((input_dim, TIME_STEPS))(a) 
    # this line is not useful. It's just to know which dimension is what.
    a = tf.keras.layers.Dense(TIME_STEPS, activation='softmax')(a)
    
    a_probs = tf.keras.layers.Permute((2, 1), name='attention_vec')(a)
    
    output_attention_mul  = tf.keras.layers.multiply([inputs, a_probs])
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul

def model_attention_applied_after_lstm():
    
    inputs        = tf.keras.Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_units    = 32
    
    lstm_out      = tf.keras.layers.LSTM(lstm_units, return_sequences=True)(inputs)
    
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = tf.keras.layers.Flatten()(attention_mul)
    
    output        = tf.keras.layers.Dense(1, activation='sigmoid')(attention_mul)
    
    model         = tf.keras.Model(inputs=[inputs], outputs=output)
    
    return model

def model_attention_applied_before_lstm():
    
    inputs        = tf.keras.Input(shape=(TIME_STEPS, INPUT_DIM,))
    
    attention_mul = attention_3d_block(inputs)
    
    lstm_units    = 32
    
    attention_mul = tf.keras.layers.LSTM(lstm_units, return_sequences=False)(attention_mul)    
    output        = tf.keras.layers.Dense(1, activation='sigmoid')(attention_mul)
    
    model         = tf.keras.Model(inputs=[inputs], outputs=output)
    
    return model

def get_data_recurrent(n, time_steps, input_dim, attention_column=10):

    x = np.random.standard_normal(size=(n, time_steps, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))

    x[:, attention_column, :] = np.tile(y, input_dim)

    return x, y