from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D,Concatenate,concatenate, LSTM,LayerNormalization, BatchNormalization, Flatten, Dropout, Dense,SpatialDropout1D,SeparableConv1D
from tensorflow.keras.layers.experimental import RandomFourierFeatures
from tensorflow.keras.losses import poisson

from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import keras
from keras.regularizers import l2
import numpy as np







def prediction(filename,number_of_sequences,model_weights):

    def encode_seq(s):
        Encode = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'U':[0,0,0,1],'T':[0,0,0,1],'N':[0,0,0,0]}
        return np.array([Encode[x] for x in s])

    def EIIP(s):
        Encode = {'A':[0.1260],'C':[0.1340],'G':[0.0806],'T':[0.1335],'U':[0.1335],'N':[0]}
        return np.array([Encode[x] for x in s])
    def cal(c,cb,i):
        bases ={'A':[1,1,1], 'C':[0,1,0], 'G':[1,0,0,], 'T':[0,0,1],'U':[0,0,1],'N':[0,0,0]}
        p=[]
        p=bases[c]
        p.append(np.round(cb/float(i+1),2))
        return(p)
    def calculate(s):
        p=f=list()
        cba=cbc=cbt=cbg=0
        for i,c in enumerate(s):
            if c=='A':
                cba+=1
                p=cal(c,cba,i)
            elif c=='T':
                cbt+=1
                p=cal(c,cbt,i)
            elif c=='U':
                cbt+=1
                p=cal(c,cbt,i)
            elif c=='C':
                cbc+=1
                p=cal(c,cbc,i)
            elif c=='G':
                cbg+=1
                p=cal(c,cbg,i)
            else:
                p=[0,0,0,0]
            f.append(p)
        return(f)
    
    def listToString(s):  
    
        str1 = ""  
        
        # traverse in the string   
        for ele in s:  
            str1 += ele   
        
        # return string   
        return str1   
    
    file = open(filename,"r")
    count=0
    Training=[0]*number_of_sequences
    for line in file:
      
      #Let's split the line into an array called "fields" using the ";" as a separator:
      Data = line.split(':')
      Training[count] = Data
      count=count+1
      
  
    
    elements1 = {}
    
    accumulator=0
    for row in Training:
      #print(row)
      row=listToString(row)
      row=row.strip('\n')
      if len(row)<101:
          row=row+'N'*(101-len(row))
      my_hottie = encode_seq((row))
      out_final=my_hottie
      out_final = np.array(out_final)
      elements1[accumulator]=out_final
      accumulator += 1 
   
    X = list(elements1.items())
    an_array = np.array(X)
    an_array=an_array[:,1]
    transpose = an_array.T
    transpose_list = transpose.tolist()
    X=np.transpose(transpose_list)
    X=np.transpose(X)
    #y = np.array(data['label'], dtype = np.int32);
    

    elements1 = {}
    accumulator=0
    for row in Training:
      row=listToString(row)
      row=row.strip('\n')
      if len(row)<101:
          row=row+'N'*(101-len(row))
      my_hottie = calculate((row))
      out_final=my_hottie
      out_final = np.array(out_final)
      elements1[accumulator]=out_final
      accumulator += 1 
   
    Y = list(elements1.items())
    an_array = np.array(Y)
    an_array=an_array[:,1]
    transpose = an_array.T
    transpose_list = transpose.tolist()
    Y=np.transpose(transpose_list)
    Y=np.transpose(Y)
   
    
   
    elements1 = {}
    accumulator=0
    for row in Training:
      row=listToString(row)
      row=row.strip('\n')
      if len(row)<101:
          row=row+'N'*(101-len(row))
      my_hottie = EIIP((row))
      out_final=my_hottie
      out_final = np.array(out_final)
      elements1[accumulator]=out_final
      accumulator += 1 
    Z = list(elements1.items())
    an_array = np.array(Z)
    an_array=an_array[:,1]
    transpose = an_array.T
    transpose_list = transpose.tolist()
    Z=np.transpose(transpose_list)
    Z=np.transpose(Z)
      
   
    input_shape = (101,4) # One Hot
    inputs = Input(shape = input_shape)
    inputs2 = Input(shape = (101,4))  #NCPD
    inputs3 = Input(shape = (101,1))  #EIIP

    initializer = tf.keras.initializers.RandomUniform()
    c1 = Conv1D(64,3, strides=1, activation='relu', input_shape=(101, 4))(inputs)
    c1 = Conv1D(32,3, strides=1, activation='relu', input_shape=(101, 4))(c1)

    c1 = LayerNormalization()(c1)
    c1 = MaxPooling1D(2,strides=3)(c1)

    c2 = Conv1D(64,3, strides=1, activation='relu', input_shape=(101, 4))(inputs2)
    c2 = Conv1D(32,3, strides=1, activation='relu', input_shape=(101, 4))(c2)

    c2 = LayerNormalization()(c2)
    c2 = MaxPooling1D(2,strides=3)(c2)
    
    c3 = Conv1D(64,3, strides=1, activation='relu', input_shape=(101, 1))(inputs3)
    c3 = Conv1D(16,3, strides=1, activation='relu', input_shape=(101, 1))(c3)

    c3 = LayerNormalization()(c3)
    c3 = MaxPooling1D(2,strides=3)(c3)
    
    con0 = concatenate([c1,c2,c3])
    

    cd = Dropout(0.50)(con0)
    c3 = Conv1D(32,3, strides=1, activation='relu')(cd)
    c3 = LayerNormalization()(c3)
    c3 = MaxPooling1D(2,strides=3)(c3)
    c3 = Dropout(0.10)(c3)
    

    c3 = Conv1D(16,3, strides=1, activation='relu')(c3)
    c3 = LayerNormalization()(c3)
    #c3 = MaxPooling1D(2,strides=2)(c3)         
    #fc0 = Dense(16, activation='elu')(fc)
    #R1=RandomFourierFeatures(300, kernel_initializer="gaussian")(fc)
    fc = Flatten()(c3)



    fc1 = Dense(64, activation='relu')(fc)
    cd = Dropout(0.50)(fc1)

    fc1 = Dense(32, activation='relu')(cd)

    fc2 = Dense(1, activation='sigmoid')(fc1)
    
    #model1 = Model(inputs =[ip], outputs = [fc2])
    
    #model1.compile(loss='kl_divergence', optimizer= 'adam', metrics=['accuracy'])
    model1 = Model(inputs =[inputs,inputs2,inputs3], outputs = [fc2])    
    opt=SGD(learning_rate=0.003, momentum = 0.80)

    model1.compile(loss='binary_crossentropy', optimizer= opt, metrics=['accuracy'])
    
      
    
    Predict=[]
    
    model1.load_weights(model_weights)
    
    Predict = model1.predict([X,Y,Z])
    
    Predict=Predict.round();
    return Predict


model_weights='Human.h5'
number_of_sequences=24532


Prediction=prediction("Positive.txt",number_of_sequences,model_weights)