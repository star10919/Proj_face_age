### 랜덤 ->  [4차원 -> flow] -> concatenate

import numpy as np
from icecream import ic
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAveragePooling2D, Conv1D, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
import time 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt



### ImageDataGenerator로 데이터 증폭시키기
# imageGen = ImageDataGenerator(
#     rescale=1./255,
#     horizontal_flip=True,
#     vertical_flip=True,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     rotation_range=5,
#     zoom_range=1.2,
#     shear_range=0.7,
#     fill_mode='nearest',
#     validation_split=0.2
#     )

# predictGen = ImageDataGenerator(
#     rescale=1./255,
#     horizontal_flip=True,
#     vertical_flip=True,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     rotation_range=10,
#     zoom_range=1.0,
#     shear_range=0.7,
#     fill_mode='nearest',
#     )




# xy_train = imageGen.flow_from_directory(
#     '../data/real_age',
#     target_size=(32, 32),
#     batch_size=2000,
#     class_mode='categorical',
#     subset='training'
# )
# # Found 880 images belonging to 11 classes.
# ic(xy_train[0][0].shape)     #  (880, 32, 32, 3)
# ic(xy_train[0][1].shape)     #  (880, 11)


# xy_test = imageGen.flow_from_directory(
#     '../data/real_age',
#     target_size=(32, 32),
#     batch_size=2000,
#     class_mode='categorical',
#     subset='validation'
# )
# # Found 220 images belonging to 11 classes.
# ic(xy_test[0][0].shape)     # (220, 32, 32, 3)
# ic(xy_test[0][1].shape)     # (220, 11)


# xy_pred = predictGen.flow_from_directory(
#     '../data/real_age_predict',
#     target_size=(32, 32),
#     batch_size=2000,
#     class_mode='categorical',
#     shuffle=False
# )
# # Found 11 images belonging to 11 classes.
# ic(xy_pred[0][0].shape)     # (11, 32, 32, 3)
# ic(xy_pred[0][1].shape)     # (11, 11)

# x_train = xy_train[0][0]
# y_train = xy_train[0][1]
# x_test = xy_test[0][0]
# y_test = xy_test[0][1]
# x_pred = xy_pred[0][0]
# y_pred = xy_pred[0][1]

# ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape, x_pred.shape, y_pred.shape)



# ### 넘파이로 세이브하기
# np.save('./_save/_npy/proj_faceage_x_train.npy', arr=x_train)
# np.save('./_save/_npy/proj_faceage_x_test.npy', arr=x_test)
# np.save('./_save/_npy/proj_faceage_y_train.npy', arr=y_train)
# np.save('./_save/_npy/proj_faceage_y_test.npy', arr=y_test)
# np.save('./_save/_npy/proj_faceage_x_pred.npy', arr=x_pred)
# np.save('./_save/_npy/proj_faceage_y_pred.npy', arr=y_pred)


# ============================================================================================================

### 데이터 로드하기
x_train = np.load('./_save/_npy/proj_faceage_x_train.npy')
x_test = np.load('./_save/_npy/proj_faceage_x_test.npy')
y_train = np.load('./_save/_npy/proj_faceage_y_train.npy')
y_test = np.load('./_save/_npy/proj_faceage_y_test.npy')
x_pred = np.load('./_save/_npy/proj_faceage_x_pred.npy')
y_pred = np.load('./_save/_npy/proj_faceage_y_pred.npy')

ic(x_train)
ic(x_test)
ic(y_train)
ic(y_test)
ic(x_pred)
ic(y_pred)
ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape, x_pred.shape, y_pred.shape)

'''
    x_train.shape: (880, 32, 32, 3)
    x_test.shape: (220, 32, 32, 3)
    y_train.shape: (880, 11)
    y_test.shape: (220, 11)
    x_pred.shape: (11, 32, 32, 3)
    y_pred.shape: (11, 11)
'''



ic(np.unique(y_train))  # 0, 1 : 2개
# 전처리 하기 -> scailing
# 단, 2차원 데이터만 가능하므로 4차원 -> 2차원
x_train = x_train.reshape(x_train.shape[0], 32*32*3)
x_test = x_test.reshape(x_test.shape[0], 32*32*3)
x_pred = x_pred.reshape(x_pred.shape[0], 32*32*3)

# 1-2. x 데이터 전처리
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
x_pred = x_pred.reshape(x_pred.shape[0], 32, 32, 3)

# 2. 모델 구성(GlobalAveragePooling2D 사용)
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), padding='same',                        
                        activation='relu' ,input_shape=(32, 32, 3)))
model.add(MaxPool2D(2,2))
model.add(Conv2D(100, (2,2), padding='same', activation='relu'))
model.add(MaxPool2D(2,2))        
model.add(Dropout(0.6))                                             
# model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
# model.add(MaxPool2D(2,2))                  
# model.add(Conv2D(64, (2,2), padding='same', activation='relu'))   
# model.add(Dropout(0.4)) 
# model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
# model.add(MaxPool2D(2,2))      
model.add(GlobalAveragePooling2D())                                              
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(11, activation='softmax'))



# 3. 컴파일(ES), 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', patience=500, verbose=1, mode='min')
cp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True,
                        filepath='./_save/ModelCheckPoint/face_age_MCP6.hdf5')

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10000, verbose=2, callbacks=[es, cp], validation_split=0.4, shuffle=True, batch_size=512)
end_time = time.time() - start_time

model.save('./_save/ModelCheckPoint/face_age_model_save6.h5')

# model = load_model('./_save/ModelCheckPoint/face_age_model_save.h5')           # save model
# model = load_model('./_save/ModelCheckPoint/face_age_MCP.hdf5')                # checkpoint

# 4. 평가, 예측
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

loss = model.evaluate(x_test, y_test)
print("걸린시간 :", end_time)
print('acc :',acc[-1])
print('val_acc :',val_acc[-1])
print('loss :',loss[-1])
print('val_loss :',val_loss[-1])


y_predict = model.predict(x_pred)
ic(y_predict)
y_predict = np.argmax(y_predict,0)
ic(y_predict)


real = np.array(y_predict)
age = []
def real_age():
    for i in range(11):
        if real[i] == 0:
            print("11-15세")
        elif real[i] == 1:
            print("16-20세")
        elif real[i] == 2:
            print("21-25세")
        elif real[i] == 3:
            print("26-30세")
        elif real[i] == 4:
            print("31-35세")
        elif real[i] == 5:
            print("36-40세")
        elif real[i] == 6:
            print("41-45세")
        elif real[i] == 7:
            print("46-50세")
        elif real[i] == 8:
            print("51-55세")
        elif real[i] == 9:
            print("56-60세")
        elif real[i] == 10:
            print("61세 이상")
        else:
            print("ERR")
        age.append(real[i])
    return print(age)

real_age()


# 시각화 
plt.figure(figsize=(9,5))

# 1
plt.subplot(2, 1, 1) # 2개의 플롯을 할건데, 1행 1열을 사용하겠다는 의미 
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

# 2
plt.subplot(2, 1, 2) # 2개의 플롯을 할건데, 1행 2열을 사용하겠다는 의미 
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])

plt.show()


'''
*augment X
걸린시간 : 6.861128091812134
acc : 0.40430623292922974
val_acc : 0.20454545319080353

'./_save/ModelCheckPoint/face_age_model_save2.h5'
걸린시간 : 6.263158321380615
acc : 0.5179426074028015
val_acc : 0.27272728085517883
loss : 0.16818182170391083
val_loss : 2.3232505321502686
[10, 2, 2, 4, 3, 8, 9, 0, 9, 10, 6]  1개

모델수정(deep하게)
'./_save/ModelCheckPoint/face_age_model_save3.h5'
걸린시간 : 6.7808356285095215
acc : 0.6842105388641357
val_acc : 0.20454545319080353
loss : 0.13181817531585693
val_loss : 2.907097101211548
[4, 4, 1, 4, 4, 2, 9, 10, 8, 3, 7]  2개

'./_save/ModelCheckPoint/face_age_model_save4.h5'
걸린시간 : 16.13235878944397
acc : 0.31937798857688904
val_acc : 0.11363636702299118
loss : 0.15454545617103577
val_loss : 2.37274169921875
[7, 5, 2, 3, 3, 4, 7, 0, 9, 1, 1]  2개

'./_save/ModelCheckPoint/face_age_model_save5.h5'
걸린시간 : 4.002293586730957
acc : 0.09280303120613098
val_acc : 0.08806817978620529
loss : 0.09090909361839294
val_loss : 2.400324821472168
[9, 3, 4, 3, 10, 3, 4, 7, 9, 6, 9] 1개

'./_save/ModelCheckPoint/face_age_model_save6.h5'


'''