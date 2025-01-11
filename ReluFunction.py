#Yapay Sinir Ağları 
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


df = pd.read_csv('Student_Grades.csv')
print(df.head(3))

y = df['Scores']
x = df.drop(columns= ['Scores','Grade'])

#keras kullanıldığı için veriler float a çevrildi.
y = y.values.astype ('float32')
x = x.values.astype ('float32')

#yapay sinir ağı modeli oluşturma
model = Sequential() 
model.add(Dense(16)) #giriş katmanı
model.add(Dense(32,activation='relu')) #gizli katman1 , relu aktivasyon kodudur.
model.add(Dense(32,activation='relu')) #gizli katman2
model.add(Dense(1)) #çıkış katmanı

model.compile(optimizer='adam',loss="mse") #modeli derleme  
model.fit(x, y, epochs=128,batch_size=16) #eğitme

#ilk hata  loss: 3303.4163 iken 32. çalışmada oss: 2175.7312 olmuş
#düşme devam epochs artır.

predict =model.predict(np.array([[3,1,1,2,3.3]])) 
print(predict) #score: [[27.021938]]