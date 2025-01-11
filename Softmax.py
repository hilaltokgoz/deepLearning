#Softmax
#anemi tipi bulma 
#sınıflandırma örneği

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


df= pd.read_csv('diagnosed_anemi.csv')
print(df.head(3))

y = df['Diagnosis']
x = df.drop('Diagnosis',axis = 1) #sütun çıkar
y #Normocytic hypochromic anemia -> ilk eleman

#preproccessing
#Label Encoder
le = LabelEncoder()
y = le.fit_transform(y) #text no numeric
y #y=5 -> ilk eleman

#y=le.inverse_transform([5]) 
#y #Normocytic hypochromic anemia -> ilk eleman numeric to text

#kategori için dönüşüm yapılması gerekli.
#01000 -> 1. değer
#10000 -> 0. değer
#00001 -> 4. değer
y_categorical = to_categorical(y)
y_categorical #[0., 0., 0., 0., 0., 1., 0., 0., 0.] ->ilk eleman bu şekle dönüştü.


#train-test
x_train,x_test,y_train,y_test= train_test_split(x,y_categorical,train_size = 0.75,random_state=43)

#model oluşturma
model = Sequential()
col =len(x.columns)
model.add(Dense(64,activation='relu',input_dim=col)) #input
model.add(Dense(32,activation='relu')) #hidden layer
model.add(Dense(9,activation='softmax'))#output, kaç çeşit anemi varsa

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=128, batch_size=32, validation_split=0.24)
#accuracy: 0.8078,doğruluk en son değeri %80 olmuş

tahmin=np.array([[9.4,34,58,2.8,4.3,3.3,8.8,32,78,24,26,155,15,0.17]])
sonuc = model.predict(tahmin)

test=np.argmax(sonuc)
print(test) #5, numeric

num_to_text=le.inverse_transform([test])
print(num_to_text) #['Normocytic hypochromic anemia']

