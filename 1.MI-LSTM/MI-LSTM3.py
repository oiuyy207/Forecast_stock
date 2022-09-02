#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
#import data as d
#import model
import time
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

from util import *


# In[2]:


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

rnn=tf.compat.v1.nn.rnn_cell


# In[3]:


#config
path, path_asset, _ = make_dataset_for_M6(start_date='2012-01-01',end_date="2022-02-28",path = "./stock_220228")
#path = './stock_20h'
#path_asset = './stock_20h/asset'

INDEX_DATA = make_index_data(start_date='2012-01-01',end_date="2022-02-28")

timesize=[25]
#timesize=[35+5*i for i in range(6)]
timesize_for_calc_correlation=70
positive_correlation_stock_num=30
negative_correlation_stock_num=30
train_test_rate=0.8
batch_size=1024
H = 20

model_save_path = "./model_save/model_220228"

adam_lr = 0.001
epochs = 100
print(f"{' Config ':*^50}")

print(f"{'model_save_path' :>35} : {model_save_path}")

print(f"{'timesize' :>35} : {timesize}")
print(f"{'timesize_for_calc_correlation' :>35} : {timesize_for_calc_correlation}")
print(f"{'correlation_stock_num' :>35} : {positive_correlation_stock_num}, {negative_correlation_stock_num}")
print(f"{'H' :>35} : {H}")
print(f"{'train_test_rate' :>35} : {train_test_rate}")
print(f"{'batch_size' :>35} : {batch_size}")
print(f"{'adam_lr' :>35} : {adam_lr}")
print(f"{'epochs' :>35} : {epochs}")

print(f"{' Config ':*^50}")

#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# In[5]:



class base_LSTMCell(rnn.BasicLSTMCell):
    def __call__(self,inputs,state,scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
        # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(1, 2, state)
            concat = tf.layers.dense(tf.concat([inputs, h],axis=1), 4 * self._num_units)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(concat, 4, 1)

            new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) *
                    self._activation(j))
            new_h = self._activation(new_c) * tf.sigmoid(o)

            if self._state_is_tuple:
                new_state = rnn.LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat(1, [new_c, new_h])
        return new_h, new_state


# In[6]:


class MI_LSTMCell(rnn.BasicLSTMCell):
    """
    Multi-Input LSTM proposed in the paper, Stock Price Prediction Using Attention-based Multi-Input LSTM.
    """
    def __init__(self,
               num_units,
               num_inputs,
               forget_bias=1.0,
               state_is_tuple=True,
               activation=None,
               reuse=None,
               name=None,
               dtype=None,
               **kwargs):
        """
        Initialize the basic LSTM cell.
        args:
            num_inputs: MI-LSTM의 입력의 개수. 
                이 파라미터에 따라 입력 게이트의 어텐션 레이어를 설정.
                최소 1개이상.
                1개일 경우, 어텐션 레이어를 제외하고 기본 LSTM과 동일.
        """        
        super(MI_LSTMCell,self).__init__(num_units,
               forget_bias=1.0,
               state_is_tuple=True,
               activation=None,
               reuse=None,
               name=None,
               dtype=None,
               **kwargs)
        
        if(type(num_inputs) is not int):
            raise ValueError("num_inputs should be integer")
        if(num_inputs < 1):
            raise ValueError("num_inputs should not be less than 0")
        self.num_inputs = num_inputs
        self.alpha_weight=self.add_variable('alpha_weight',shape=[self._num_units,self._num_units])
        self.alpha_bias=[]
        for i in range(self.num_inputs):
            self.alpha_bias.append(self.add_variable('alpha_bias'+str(i),shape=[1],initializer=tf.zeros_initializer()))

    def __call__(self,inputs,state,scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.compat.v1.variable_scope(scope or type(self).__name__) as scope:  # "BasicLSTMCell"
        # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(1, 2, state)
            inputs_list = tf.split(inputs,self.num_inputs,1)
            #scope.reuse_variables()
            concat = tf.compat.v1.layers.dense(tf.concat([inputs_list[0], h],axis=1), (3+self.num_inputs) * self._num_units)
                                 
            # 0 = forget_gate, 1 = output_gate, 2= main_new_input, 3 = main_input_gate, 4~ = input_gate_for_auxiliary
            main_list = tf.split(concat, 3+self.num_inputs, 1)
                        
            #new_input_gate= list of all new_input.
            new_input_gate=[tf.tanh(main_list[2])]
            #linear layer for auxiliary inputs.
            for i in range(1,self.num_inputs):
                new_input_gate.append(tf.compat.v1.layers.dense(tf.concat([inputs_list[i], h],axis=1),self._num_units,activation=tf.tanh))

            #making list of l. l = sigmoid(input_gate) * tanh(new_input)
            new_l=[]
            for i,new_input in enumerate(new_input_gate,3):
                new_l.append(tf.sigmoid(main_list[i]) * new_input)


            #making list of u.            
            u=[]
            for i,l in enumerate(new_l):
                #temp = transpos(l) X W X Cell_State.
                temp1=tf.matmul(l,self.alpha_weight)
                temp1=tf.expand_dims(temp1,1)
                temp2=tf.matmul(temp1,tf.expand_dims(c,2))
                u.append(tf.tanh(tf.squeeze(temp2+self.alpha_bias[i],axis=2)))

            #making list of alpha.
            alpha=tf.nn.softmax(u,axis=0)

            #making L.
            L=[]
            for i,l in enumerate(new_l):
                L.append(alpha[i]*l)
            L=tf.reduce_sum(L,axis=0)


            #new state = c(t-1) * f + L. new h = tanh(c) + sigmoid(o)
            new_c = (c * tf.sigmoid(main_list[0] + self._forget_bias)+L)
            new_h = self._activation(new_c) * tf.sigmoid(main_list[1])

            if self._state_is_tuple:
                new_state = rnn.LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat(1, [new_c, new_h])
        return new_h, new_state


# In[7]:


class Attention_Layer():
    """
    어텐션 레이어.
    (None, TimeWindow, hidden_unit_size) shape의 LSTM 출력을 입력으로 받아 (None, 1, hidden_unit_size)의 텐서 출력.
    
    """
    def __init__(
        self,
        timewindow_size,
        input_hidden_unit_size,
        attention_size=None):
        """
        Setting parameter for attention layer.
        args:
            timewindow_size = time window size of previous lstm layer.
            input_hidden_unit_size = hidden unit number of previous lstm layer.
            attention_size = size of this attention. 
                default = input_hidden_unit_size.
        """
        if(attention_size is None):
            attention_size=input_hidden_unit_size
        self.o_size=attention_size
        self.h_size=input_hidden_unit_size
        self.t_size=timewindow_size

        self.beta_weight=tf.Variable(tf.random.normal([self.h_size,self.o_size]), name='beta_weight')
        self.beta_bias=tf.Variable(tf.zeros([self.o_size]),name='beta_bias')

        self.v=tf.Variable(tf.random.normal([self.o_size,1]),name='beta_v')

    def __call__(self,inputs):
        """
        producing output with actual inputs.
        shape of output will be (batch_size, 1, input_hidden_unit_size).
        """
        #temp = tanh(Y X W + b) ->shape of result = (-1, self.o_size)
        temp=tf.matmul(tf.reshape(inputs,[-1,self.h_size]),self.beta_weight)
         
        temp=tf.tanh(temp+self.beta_bias)
        
            
        #j=temp X v
        j=tf.reshape(tf.matmul(temp,self.v),[-1,self.t_size,1])

        beta=tf.nn.softmax(j)
        
        

        output=beta*inputs
        return output


# In[8]:


class LSTM_Model(keras.Model):
    """
    Basic LSTM list for test.
    """
    def __init__(self,windowsize,Pos,Neg):
        super(LSTM_Model, self).__init__()
        self.T=windowsize
        self.P=Pos
        self.N=Neg

        self.Y=tf.keras.layers.InputLayer(input_shape=(None,self.T,1),dtype = tf.float32)
        self.Xp=tf.keras.layers.InputLayer(input_shape=(None,self.P,self.T,1),dtype = tf.float32)
        self.Xn=tf.keras.layers.InputLayer(input_shape=(None,self.N,self.T,1),dtype = tf.float32)
        self.Xi=tf.keras.layers.InputLayer(input_shape=(None,9,self.T,1),dtype = tf.float32)

        self.LSTM1=tf.keras.layers.RNN(tf.keras.layers.LSTMCell(64,name='lstm1'),
                                       return_sequences=True,
                                       return_state=True)
        
        #MI-LSTM
        self.LSTM2=tf.keras.layers.RNN(MI_LSTMCell(64,4,name='lstm2'),
                                       return_sequences=True,
                                       return_state=True)
        
        #Attention_Layer
        self.attention_layer=Attention_Layer(self.T,64)

        self.flatten = tf.keras.layers.Flatten()

        self.dense_layers = tf.keras.models.Sequential()
        self.dense_layers.add(tf.keras.layers.Dense(64, activation='relu'))
        self.dense_layers.add(tf.keras.layers.Dense(64, activation='relu'))
        self.dense_layers.add(tf.keras.layers.Dense(64, activation='relu'))
        self.dense_layers.add(tf.keras.layers.Dense(64, activation='relu'))
        self.dense_layers.add(tf.keras.layers.Dense(1))

    def call(self,y,xp,xn,xi):
        Y_1,_,_=self.LSTM1(self.Y(y))
        
        Xis=tf.split(self.Xi(xi),9,1)
        Xi_list=[]
        for i in range(len(Xis)):
            o,_,_=self.LSTM1(tf.squeeze(Xis[i],axis=1))
            Xi_list.append(o)

        Xps=tf.split(self.Xp(xp),self.P,1)
        Xp_list=[]
        for i in range(len(Xps)):
            o,_,_=self.LSTM1(tf.squeeze(Xps[i],axis=1))
            Xp_list.append(o)
        
        Xns=tf.split(self.Xn(xn),self.N,1)
        Xn_list=[]
        for i in range(len(Xns)):
            o,_,_=self.LSTM1(tf.squeeze(Xns[i],axis=1))
            Xn_list.append(o)
        Xp_1=tf.reduce_mean(Xp_list,0)
        Xn_1=tf.reduce_mean(Xn_list,0)
        Xi_1=tf.reduce_mean(Xi_list,0)

        result=tf.concat([Y_1,Xp_1,Xn_1,Xi_1],axis=2)
       
        Y_2,_,_ =self.LSTM2(result)
        
        Y_3=self.attention_layer(Y_2)

        #Non-linear units for producing final prediction.
        R_1 = self.flatten(Y_3)
        R_6=self.dense_layers(R_1)

        return R_6


# In[9]:


def one_model_iter(Window_size,Window_size_CC):
    all_stock = StockData_3(path_asset,
                            INDEX_DATA,
                            Window_size,
                            Window_size_CC,
                            positive_correlation_stock_num,
                            negative_correlation_stock_num,
                            train_test_rate,
                            batch_size,
                            date_duration = 2000,
                            h = H)
    
    _,_ = scaler_info_csv(all_stock.scaler, all_stock.indexScaler, Window_size, all_stock.include_asset, all_stock.include_index)
    
    lstmModel=LSTM_Model(Window_size,
                         positive_correlation_stock_num,
                         negative_correlation_stock_num)
        
    result_dic={}
        
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer=tf.keras.optimizers.Adam(adam_lr)
    evalution_costplt=[]
    
    best_e_cost = 100
    
    os.makedirs(model_save_path+f"/Window_{Window_size}",exist_ok=True)
    os.makedirs(model_save_path+"/evalution_costplt",exist_ok=True)
    for i in range(epochs):
        #epoch start
        start_time = time.time()
        training_cost=0
        evalution_cost=0
    
        #training batch
        opt = 'training'
        for batch in tqdm(all_stock.getBatch('training'),unit=" Batches",desc=f"{opt:>15}"):
            with tf.GradientTape() as tape:
                logits = lstmModel(batch['y'],batch['xp'],batch['xn'],batch['xi'])
                loss_value = loss_fn(batch['target'], logits)
                loss_value += sum(lstmModel.losses)
            training_cost+=loss_value
            grads = tape.gradient(loss_value, lstmModel.trainable_weights,unconnected_gradients=tf.UnconnectedGradients.ZERO)
            optimizer.apply_gradients(zip(grads, lstmModel.trainable_weights))

        #evaluation batch
        opt = 'evaluation'
        for batch in tqdm(all_stock.getBatch('evaluation'),unit=" Batches",desc=f"{opt:>15}"):
            logits = lstmModel(batch['y'],batch['xp'],batch['xn'],batch['xi'])
            loss_value = loss_fn(batch['target'], logits)
            loss_value += sum(lstmModel.losses)
            evalution_cost+=loss_value
        
        #epoch end
        elapsed_time = time.time()-start_time
        training_cost=training_cost/all_stock.batchNum['training']
        evalution_cost=evalution_cost/all_stock.batchNum['evaluation']
        result_dic[i]=[training_cost,evalution_cost]
        evalution_costplt.append(evalution_cost)
        
        plt.plot(evalution_costplt,label=f"Window Size {Window_size}")
        plt.legend()
        plt.savefig(model_save_path + f'/evalution_costplt/windowsize_{Window_size}.png')
        plt.clf()
        
        #model_save
        if best_e_cost > evalution_cost:
            lstmModel.save_weights(model_save_path+f"/Window_{Window_size}/window_{Window_size}")
            best_e_cost = evalution_cost
            print(f" ** save weights! : epoch = {i} ** ")

        print(f'epoch : {i}, t_cost : {training_cost:0.6f}, e_cost : {evalution_cost:0.6f}, elapsed time : {elapsed_time:0.2f}sec')

    #Finish epoch
    
    sorted_result=sorted(result_dic,key=lambda k:result_dic[k][1])
    bestEpoch=sorted_result[0]
    print(f'\n#Best result at epoch {bestEpoch}')
    print(f't_cost : {result_dic[bestEpoch][0]:0.6f}, e_cost : {result_dic[bestEpoch][1]:0.6f}')

    plt.plot(evalution_costplt,label=f"Window Size {Window_size}")
    plt.legend()
    
    plt.savefig(f'./evalution_costplt/windowsize_{Window_size}.png')

    #reset memory
    del lstmModel
    del all_stock
    gc.collect()
    
    return None


# Main

# In[ ]:


for w in timesize:
    t = " WindowSize "+str(w)+" "
    print(f"{t:=^40}")
    one_model_iter(w,timesize_for_calc_correlation)
    print()


# In[ ]:




