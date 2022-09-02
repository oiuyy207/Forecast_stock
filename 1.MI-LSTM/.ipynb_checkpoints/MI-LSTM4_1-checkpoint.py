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
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

from util import *


# In[2]:


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

rnn=tf.compat.v1.nn.rnn_cell


# In[3]:


#config
path, path_asset, _ = make_dataset_for_M6(start_date='2012-01-01',end_date="2022-05-01",path = "./stock_220430")
#path = './stock_220228'
#path_asset = './stock_220228/asset'

INDEX_DATA = make_index_data(start_date='2012-01-01',end_date="2022-05-01")

timesize=[25,20,15]
#timesize=[35+5*i for i in range(6)]
timesize_for_calc_correlation=[30,25,20]
positive_correlation_stock_num=15
negative_correlation_stock_num=5
train_test_rate=0.8
batch_size=1024
H = 20

model_save_path = "./model_save/model_2200428"

adam_lr_start = 0.005
adam_lr_end = 0.00001
epochs = 300
print(f"{' Config ':*^50}")

print(f"{'model_save_path' :>35} : {model_save_path}")

print(f"{'timesize' :>35} : {timesize}")
print(f"{'timesize_for_calc_correlation' :>35} : {timesize_for_calc_correlation}")
print(f"{'correlation_stock_num' :>35} : {positive_correlation_stock_num}, {negative_correlation_stock_num}")
print(f"{'H' :>35} : {H}")
print(f"{'train_test_rate' :>35} : {train_test_rate}")
print(f"{'batch_size' :>35} : {batch_size}")
print(f"{'adam_lr' :>35} : {adam_lr_start}~{adam_lr_end}")
print(f"{'epochs' :>35} : {epochs}")

print(f"{' Config ':*^50}")

#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


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
    
    _,_ = scaler_info_csv(all_stock.scaler, all_stock.indexScaler, Window_size, all_stock.include_asset, all_stock.include_index,model_save_path,scaler_info = "minmax")
    
    lstmModel=LSTM_Model(Window_size,
                         positive_correlation_stock_num,
                         negative_correlation_stock_num)
        
    result_dic={}
        
    loss_fn = tf.keras.losses.MeanSquaredError()
    adam_lr = adam_lr_start
    optimizer=tf.keras.optimizers.Adam(adam_lr)
    train_costplt=[]
    evalution_costplt=[]
    
    linear_decrease_lr_sp = int(3/5*epochs)
    linear_decrease_lr_slope = (adam_lr_end - adam_lr_start)/(epochs - linear_decrease_lr_sp)
    
    print("-"*70)
    print(f'{"lr decrease start point":>40} : {linear_decrease_lr_sp}')
    print(f'{"lr decrease slope":>40} : {linear_decrease_lr_slope}')
    print("-"*70)
    
    best_e_cost = 100
    
    os.makedirs(model_save_path+f"/Window_{Window_size}",exist_ok=True)
    os.makedirs(model_save_path+"/evalution_costplt",exist_ok=True)
    for i in range(epochs):
        #epoch start
        start_time = time.time()
        training_cost=0
        evalution_cost=0
        
        #lr set
        if i > linear_decrease_lr_sp:
            adam_lr = adam_lr_start + linear_decrease_lr_slope*(i-linear_decrease_lr_sp)
            optimizer=tf.keras.optimizers.Adam(adam_lr)
        
        #training batch
        opt = 'training'
        for batch in tqdm(all_stock.getBatch('training'),unit=f" Batches(lr:{adam_lr:.6f})",desc=f"{opt:>15}"):
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
        train_costplt.append(training_cost)
        evalution_costplt.append(evalution_cost)
        
        #model_save
        if best_e_cost > evalution_cost:
            lstmModel.save_weights(model_save_path+f"/Window_{Window_size}/window_{Window_size}")
            best_e_cost = evalution_cost
            print(f" ** save weights! : epoch = {i} ** ")

        
        plt.plot(train_costplt,"C0--",label="train MSE")
        plt.plot(evalution_costplt,"C0-",label="validation MSE")
        plt.title(f"Window Size {Window_size}\nBest Validation MSE{best_e_cost}")
        plt.legend()
        plt.savefig(model_save_path + f'/evalution_costplt/windowsize_{Window_size}.png')
        plt.clf()

        print(f'epoch : {i}, t_cost : {training_cost:0.6f}, e_cost : {evalution_cost:0.6f}, elapsed time : {elapsed_time:0.2f}sec')

    #Finish epoch
    
    sorted_result=sorted(result_dic,key=lambda k:result_dic[k][1])
    bestEpoch=sorted_result[0]
    print(f'\n#Best result at epoch {bestEpoch}')
    print(f't_cost : {result_dic[bestEpoch][0]:0.6f}, e_cost : {result_dic[bestEpoch][1]:0.6f}')

    plt.plot(train_costplt,"C0--",label="train MSE")
    plt.plot(evalution_costplt,"C0-",label="validation MSE")
    plt.title(f"Window Size {Window_size}\nBest Validation MSE : {result_dic[bestEpoch][1]:0.6f}")
    plt.legend()
    
    plt.savefig(model_save_path + f'/evalution_costplt/windowsize_{Window_size}.png')

    #reset memory
    del lstmModel
    del all_stock
    gc.collect()
    
    return None


# Main

# In[ ]:
if type(timesize_for_calc_correlation) == list:
    for w,w_c in zip(timesize,timesize_for_calc_correlation):
        t = " WindowSize "+str(w)+" "
        print(f"{t:=^40}")
        one_model_iter(w,w_c)
        print()
    
elif type(timesize_for_calc_correlation) == int:
    for w in timesize:
        t = " WindowSize "+str(w)+" "
        print(f"{t:=^40}")
        one_model_iter(w,timesize_for_calc_correlation)
        print()
else:
    raise TypeError(f'type of timesize_for_calc_correlation({type(timesize_for_calc_correlation)}) must be int or list')

# In[ ]:




