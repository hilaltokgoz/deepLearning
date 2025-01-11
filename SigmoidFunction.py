#Sigmoid Function
#mantarın zehirli olup olmadığını bulacağız
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

df = pd.read_csv('mushroom_cleaned.csv')
print(df.head(3))

#bağımlı-bağımsız değişkenleri bulma
y = df['class']
x= df.drop('class', axis=1) #sütun çıkar

y = y.astype('float32')
x = x.astype('float32')
#model oluşturma
model = Sequential()
#input layer
model.add(Dense(32,input_dim=8,activation='relu'))#32 nöron var, giriş boyutu:8
#hidden layer
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
#output layer
#mantar ya zehirli ya zehirsiz olur.
#sonuç o ya da 1. bu yüzden sigmoid function kullanılmalıdır.
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#entropi dağınıklığı ölçer. Yoğunluk nerede? 0 da mı 1 de mi?
#sigmoid function da binary.crossentropy (0-1)entropi kullanılır.

model.fit(x=x,y=y,epochs=8,batch_size=128,validation_split=0.2)
#epoch: kaç kere ağdan dönecek
#batch_size: toplam veri kaçarlı gönderilecek
#validation: öğrenme sırasındaki testtir, eğirimin kalitesi güçlenir.veri train ve test olarak ayrıldı.(test kenarda dursun) 
#train sırasında da veriyi train ve test olarak ayırıyor,

model.summary() #model özetini yazdırma