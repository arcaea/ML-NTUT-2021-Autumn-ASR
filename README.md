#註:H5為32114KB 無法放入github(最大25MB)

環境
=======
google colaboratory \
Hardware accelerator：GPU

建立模型
=======
Keras(conV1D)

流程
=======
![image](https://github.com/arcaea/ML-2021-Autumn-ASR/blob/main/PIC/%E6%B5%81%E7%A8%8B%E5%9C%96.jpg)

匯入函式庫
=======
```python
#import
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import librosa
import pandas as pd
import numpy as np
import random
import pickle
import glob
import csv
import os

from keras.layers import Input, Activation, Conv1D, Lambda, Add, Multiply, BatchNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow.keras.optimizers import Adam, SGD
from python_speech_features import mfcc
from keras.models import load_model
from IPython.display import Audio
from keras.models import Model
from keras import backend as K
from tqdm import tqdm
```

資料預處理(csv 轉 txt)
=======
```python
if(not os.path.exists(tran_path)):
  os.mkdir('/content/data/ML@NTUT-2021-Autumn-ASR/train/txt/')

with open(trainCSV,newline='',errors='ignore') as csvfile:
    rows=csv.reader(csvfile)
    for row in rows:
        a=row[0]
        b=row[1]
        f=open('/content/data/ML@NTUT-2021-Autumn-ASR/train/txt/'+a+'.txt','w')
        f.write(b)
        f.close()
```

資料預處理(讀取檔案)
=======
```python
def get_wav_files(wav_path):
    wav_files=[]
    for (dirpath,dirnames,filenames) in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                #print(filename)
                filename_path=os.path.join(dirpath,filename)
                #print(filename_path)
                wav_files.append(filename_path)
    return wav_files

def get_tran_texts(wav_files,tran_path):
    tran_texts=[]
    for wav_file in wav_files:
        basename=os.path.basename(wav_file)
        x=os.path.splitext(basename)[0]
        tran_file=os.path.join(tran_path,x+'.txt')
        #print(wav_filename)
        if os.path.exists(tran_file) is False:
            return None
        
        fd=open(tran_file,'r')
        text=fd.readline()
        tran_texts.append(text.split('\n')[0])
        fd.close()
    return tran_texts
```

資料預處理(MFCC)
=======
```python
fig,ax=plt.subplot(2,1)

#波形圖
librosa.display.waveshow(y=y,sr=sr,ax=ax[0])
ax[0].set_title('nutcracker waveform')

#梅爾頻祖譜
S=librosa.feature.melspectrogram(y=y,sr=sr,n_mels=128,fmax=8000)
S_dB=librosa.power_to_dB(S,ref=np.max)
librosa.display.specshow(S_dB,x_axis='time',y_axis='mel',sr=sr,fmax=8000,ax=ax[1])
ax[1].set_title('Mel-frequency spectrogram')

plt.tight_layout()
plt.show()

#MFCC取音檔特徵值
features=[]


for i in tqdm(range(len(wav_files))):
    path=wav_files[i]
    audio,sr=load_and_trim(path)
    features.append(mfcc(audio,sr,numcep=mfcc_dim,nfft=551,highfreq=8000))

samples=random.sample(features,100)
samples=np.vstack(samples)

mfcc_mean=np.mean(samples,axis=0)
mfcc_std=np.std(samples,axis=0)
print(mfcc_mean)
print(mfcc_std)

features=[(feature-mfcc_mean)/(mfcc_std+1e-14) for feature in features]    

print(len(features),features[0].shape)
```

資料預處理(文檔)
=======
```python
#檢查有使用到的文字
chars={}
for text in tran_texts:
  for e in text:
    chars[e]=chars.get(e,0)+1

chars=sorted(chars.items(),key=lambda x:x[1],reverse=True)
chars=[char[0] for char in chars]
print(len(chars),chars[:100])

char2id={c: i for i,c in enumerate(chars)}
id2char={i: c for i,c in enumerate(chars)}
print("char2id",char2id)
print("id2char",id2char)
```

存取音檔及文檔特徵值
=======
```python
data_index = np.arange(len(wav_path))
np.random.shuffle(data_index)
train_size = int(0.9 * len(wav_path))
test_size = len(wav_path) - train_size
train_index = data_index[:train_size]
test_index = data_index[train_size:]

X_train = [features[i] for i in train_index]
Y_train = [texts[i] for i in train_index]
X_test = [features[i] for i in test_index]
Y_test = [texts[i] for i in test_index]
```

模型建構
=======
```python
Y_pred=activation(batchnorm(conv1d(h1,len(char2id)+1,1,1)),'softmax')
sub_model=Model(inputs=X,outputs=Y_pred)

def calc_ctc_loss(args):
    y,yp,ypl,yl=args
    return K.ctc_batch_cost(y,yp,ypl,yl)

ctc_loss=Lambda(calc_ctc_loss,output_shape=(1,),name='ctc')([Y,Y_pred,X_length,Y_length])
```

資料視覺化
=======
![image](https://github.com/arcaea/ML-2021-Autumn-ASR/blob/main/PIC/loss.png)
 
優化器
=======
```python
history = model.fit_generator(
  generator=batch_generator(X_train, Y_train), 
  steps_per_epoch=len(X_train),
  epochs=epochs, 
  validation_data=batch_generator(X_test, Y_test), 
  validation_steps=len(X_test), 
  callbacks=[checkpointer, lr_decay])

optimizer=SGD(lr=0.02,momentum=0.9,nesterov=True,clipnorm=5)
model.compile(loss={'ctc':lambda ctc_true,ctc_pred: ctc_pred},optimizer=optimizer)
```
