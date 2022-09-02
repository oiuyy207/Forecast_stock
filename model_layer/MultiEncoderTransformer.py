import os
import numpy as np
import pandas as pd
import tensorflow as tf
from .MultiHeadAttention import *
from .encoder import * 
from .decoder import * 

def transformer(window_size, forecast_range, num_layers, dff,
                d_model, num_heads, dropout,
                name="transformer"):

    # 인코더의 입력
    inputs = tf.keras.Input(shape=(None,), name="inputs")

    # 디코더의 입력
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    # 인코더의 패딩 마스크X

    # 디코더의 룩어헤드 마스크(첫번째 서브층)
    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask, output_shape=(1, None, None),
        name='look_ahead_mask')(dec_inputs)

    # 디코더의 패딩 마스크X

    # 인코더의 출력은 enc_outputs. 디코더로 전달된다.
    enc_outputs = encoder(window_size=window_size, num_layers=num_layers, dff=dff,
                          d_model=d_model, num_heads=num_heads, dropout=dropout,)(inputs=[inputs]) # 인코더의 입력은 입력 문장과 패딩 마스크

    # 디코더의 출력은 dec_outputs. 출력층으로 전달된다.
    dec_outputs = decoder(forecast_range=forecast_range, num_layers=num_layers, dff=dff,
                          d_model=d_model, num_heads=num_heads, dropout=dropout,)(inputs=[dec_inputs, enc_outputs, look_ahead_mask])

    # 다음 단어 예측을 위한 출력층
    #outputs = tf.keras.layers.Dense(units=forecast_range, name="outputs")(dec_outputs)
    outputs = tf.keras.layers.Dense(units=1, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

def attention_enc_dec(window_size,d_model,num_heads,name='attention_enc2dec'):
    asset_outputs = tf.keras.Input(shape=(window_size,d_model), name="asset_outputs")
    pos_outputs = tf.keras.Input(shape=(window_size,d_model), name="pos_outputs")
    neg_outputs = tf.keras.Input(shape=(window_size,d_model), name="neg_outputs")
    index_outputs = tf.keras.Input(shape=(window_size,d_model), name="index_outputs")
    
    concat_outputs = tf.keras.layers.concatenate([asset_outputs,pos_outputs,neg_outputs,index_outputs],name = "concatenate")
    #concat_outputs = tf.keras.layers.concatenate([pos_outputs,neg_outputs,index_outputs],name = "concatenate")
    
    enc_outputs = MultiHeadAttention(d_model=d_model,num_heads=num_heads, name="multi_head_attention")(inputs={
        'query': concat_outputs, 'key': concat_outputs, 'value': concat_outputs, 
        'mask': None
    })
    #enc_outputs = MultiHeadAttention(d_model=d_model,num_heads=num_heads, name="multi_head_attention")(inputs={
    #    'query': asset_outputs, 'key': concat_outputs, 'value': concat_outputs, 
    #    'mask': None
    #})
    return tf.keras.Model(inputs=[asset_outputs,pos_outputs,neg_outputs,index_outputs], outputs=enc_outputs, name=name)

def siamese_enc_for_multi_input(window_size,num_layers,dff,d_model,num_heads,dropout,split_num,name=None):
    inputs = tf.keras.Input(shape=(split_num,window_size), name="inputs")
    
    squeeze_layer = reshape_layer((-1,window_size),name='squeeze')
    
    inputs_split = tf.split(inputs, num_or_size_splits=split_num, axis=1)
    for i in range(split_num):
        inputs_split[i] = squeeze_layer(inputs_split[i])
    outputs_concat = []
    
    siamese_enc = encoder(window_size=window_size, num_layers=num_layers, dff=dff,
                              d_model=d_model, num_heads=num_heads, dropout=dropout,name = "siamese_encoder")
    reshape = reshape_layer((-1,window_size,d_model,1), name='unsqueeze')
    for i in range(split_num):
        #siamese 구조
        outputs = siamese_enc(inputs=[inputs_split[i]])
        #outputs의 shape = (None,window_size,d_model)
        #reshape_layer안쓰려면 : outputs = tf.reshape(outputs,(-1,window_size,d_model,1))
        outputs = reshape(outputs)
        outputs_concat.append(outputs)
    outputs_concat = tf.keras.layers.concatenate(outputs_concat,axis=3,name = "concatenate")
    reduce_mean_outputs = tf.reduce_mean(outputs_concat,axis=3)
    return tf.keras.Model(inputs=[inputs], outputs=reduce_mean_outputs, name=name)

def multi_enc_transformer(window_size, forecast_range, P, N, I, num_layers, dff,
                         d_model, num_heads, dropout,
                         name="transformer"):

    # 인코더의 입력
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    
    # 상관주 인코더의 입력 <- 여러 벡터가 인풋으로 들어가는데 이걸 해결할 방법은? MI-LSTM은 어떻게 처리했더라??
    pos_inputs = tf.keras.Input(shape=(P,window_size), name="pos_inputs")
    neg_inputs = tf.keras.Input(shape=(N,window_size), name="neg_inputs")
    
    # 인덱스 인코더의 입력 <- 여러 벡터가 인풋으로 들어가는데 이걸 해결할 방법은? MI-LSTM은 어떻게 처리했더라??
    index_inputs = tf.keras.Input(shape=(I,window_size), name="index_inputs")
    
    # 디코더의 입력
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    # 디코더의 룩어헤드 마스크(첫번째 서브층).
    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask, output_shape=(1, None, None),
        name='look_ahead_mask')(dec_inputs)
    
    #에셋 인코더
    asset_outputs = encoder(window_size=window_size, num_layers=num_layers, dff=dff,
                          d_model=d_model, num_heads=num_heads, dropout=dropout,name = "asset_encoder")(inputs=[inputs])
    
    pos_outputs = siamese_enc_for_multi_input(window_size=window_size,num_layers=num_layers,dff=dff,
                                              d_model=d_model,num_heads=num_heads,dropout=dropout,split_num=P,name="pos_siamese_encoder")(inputs=[pos_inputs])
    neg_outputs = siamese_enc_for_multi_input(window_size=window_size,num_layers=num_layers,dff=dff,
                                              d_model=d_model,num_heads=num_heads,dropout=dropout,split_num=N,name="neg_siamese_encoder")(inputs=[neg_inputs])
    index_outputs = siamese_enc_for_multi_input(window_size=window_size,num_layers=num_layers,dff=dff,
                                              d_model=d_model,num_heads=num_heads,dropout=dropout,split_num=I,name="index_siamese_encoder")(inputs=[index_inputs])
    
    #concatenate-attention
    enc_outputs = attention_enc_dec(window_size,d_model,num_heads,name='attention_enc2dec')(inputs=[asset_outputs,pos_outputs,neg_outputs,index_outputs])
    
    # 디코더의 출력은 dec_outputs. 출력층으로 전달된다.
    dec_outputs = decoder(forecast_range=forecast_range, num_layers=num_layers, dff=dff,
                          d_model=d_model, num_heads=num_heads, dropout=dropout,)(inputs=[dec_inputs, enc_outputs, look_ahead_mask])

    # 다음 예측을 위한 출력층
    #outputs = tf.keras.layers.Dense(units=forecast_range, name="outputs")(dec_outputs)
    outputs = tf.keras.layers.Dense(units=1, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs,pos_inputs,neg_inputs,index_inputs, dec_inputs], outputs=outputs, name=name)

#그냥도 되긴 한데, 그림 그릴 때 좀더 보기 좋으라고 하는 reshape레이어
class reshape_layer(tf.keras.layers.Layer):
    def __init__(self, shape_, **kwargs):
        super(reshape_layer,self).__init__(**kwargs)
        self.shape_ = shape_
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'shape_':self.shape_,
        })
        return config
    
    def call(self,inputs):
        return tf.reshape(inputs,self.shape_)