### 랜덤 ->  [4차원 -> flow] -> concatenate

import numpy as np
from icecream import ic
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer, QuantileTransformer
import time 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.python.ops.math_ops import scalar_mul
import tensorflow as tf



# ### ImageDataGenerator로 데이터 증폭시키기
# imageGen = ImageDataGenerator(
#     rescale=1./255,
#     horizontal_flip=True,
#     vertical_flip=True,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     rotation_range=10,
#     zoom_range=1.0,
#     shear_range=0.7,
#     fill_mode='constant',#'nearest',
#     validation_split=0.20
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
#     fill_mode='constant',#'nearest',
#     )


# xy_train = imageGen.flow_from_directory(
#     '../data/real_age',
#     target_size=(32, 32),
#     batch_size=2000,
#     class_mode='categorical',
#     subset='training',
#     shuffle=True
# )
# # Found 880 images belonging to 11 classes.
# ic(xy_train[0][0].shape)     #  (880, 32, 32, 3)
# ic(xy_train[0][1].shape)     #  (880, 11)


# xy_test = imageGen.flow_from_directory(
#     '../data/real_age',
#     target_size=(32, 32),
#     batch_size=2000,
#     class_mode='categorical',
#     subset='validation',
#     shuffle=True
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

# ic(x_pred, y_pred)


# #####랜덤
# # 데이터 증폭
# augment_size=50000

# randidx = np.random.randint(x_train.shape[0], size=augment_size)
# x_augment = x_train[randidx].copy()
# y_augment = y_train[randidx].copy()
# print('%%%%%%%%%%%%%%% 1 %%%%%%%%%%%%%%%%')
# ic(x_augment.shape, y_augment.shape)        # (50000, 32, 32, 3), (50000, 11)


# #####flow
# x_augment = imageGen.flow(# x와 y를 각각 불러옴
#             x_augment,  # x
#             np.zeros(augment_size),  # y
#             batch_size=augment_size,
#             save_to_dir='d:/temp/',
#             shuffle=False).next()[0]
# ic(type(x_augment), x_augment.shape)       # <class 'numpy.ndarray'>, (50000, 32, 32, 3)
# print('%%%%%%%%%%%%%%% 2 %%%%%%%%%%%%%%%%')
# ic(x_train.shape, x_augment.shape)  #(880, 32, 32, 3), (50000, 32, 32, 3)
# ic(y_train.shape, y_augment.shape)  #(880, 11), (50000, 11)


# #####concatenate
# x_train = np.concatenate((x_train, x_augment))
# y_train = np.concatenate((y_train, y_augment))
# print('%%%%%%%%%%%%%%% 3 %%%%%%%%%%%%%%%%')
# ic(x_train.shape, y_train.shape)        #  (50880, 32, 32, 3), (50880, 11)

# ic(x_train)
# ic(x_test)
# ic(y_train)
# ic(y_test)
# ic(x_pred)
# ic(y_pred)
# ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape, x_pred.shape, y_pred.shape)

# '''
#     x_train.shape: (50880, 32, 32, 3)
#     x_test.shape: (220, 32, 32, 3)
#     y_train.shape: (50880, 11)
#     y_test.shape: (220, 11)
#     x_pred.shape: (11, 32, 32, 3)
#     y_pred.shape: (11, 11)
# '''


# ### 넘파이로 세이브하기
# np.save('./_save/_npy/proj_faceage_aug_x_train.npy', arr=x_train)
# np.save('./_save/_npy/proj_faceage_aug_x_test.npy', arr=x_test)
# np.save('./_save/_npy/proj_faceage_aug_y_train.npy', arr=y_train)
# np.save('./_save/_npy/proj_faceage_aug_y_test.npy', arr=y_test)
# np.save('./_save/_npy/proj_faceage_aug_x_pred.npy', arr=x_pred)
# np.save('./_save/_npy/proj_faceage_aug_y_pred.npy', arr=y_pred)


# ============================================================================================================

### 데이터 로드하기
x_train = np.load('./_save/_npy/proj_faceage_aug_x_train.npy')
x_test = np.load('./_save/_npy/proj_faceage_aug_x_test.npy')
y_train = np.load('./_save/_npy/proj_faceage_aug_y_train.npy')
y_test = np.load('./_save/_npy/proj_faceage_aug_y_test.npy')
x_pred = np.load('./_save/_npy/proj_faceage_aug_x_pred.npy')
y_pred = np.load('./_save/_npy/proj_faceage_aug_y_pred.npy')

ic(x_train)
ic(x_test)
ic(y_train)
ic(y_test)
ic(x_pred)
ic(y_pred)
ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape, x_pred.shape, y_pred.shape)

'''
    x_train.shape: (50880, 32, 32, 3)
    x_test.shape: (220, 32, 32, 3)
    y_train.shape: (50880, 11)
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
ic(x_train.shape, x_test.shape, y_train.shape, y_test.shape, x_pred.shape, y_pred.shape)



# 2. 모델 구성(GlobalAveragePooling2D 사용)
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, GlobalAveragePooling2D, Conv1D, LSTM
# model = Sequential()
# model.add(Conv2D(filters=64, kernel_size=(2,2), padding='same',                        
#                         activation='relu' ,input_shape=(32, 32, 3)))
# model.add(MaxPool2D(2,2))
# model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(MaxPool2D(2,2))                                                     
# model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
# # model.add(Dropout(0.2))
# model.add(MaxPool2D(2,2))                  
# model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
# model.add(GlobalAveragePooling2D())                                              
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# # model.add(Dense(64, activation='relu'))
# # model.add(Dense(32, activation='relu'))
# model.add(Dense(11, activation='softmax'))



# 3. 컴파일(ES), 훈련
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# es = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
# cp = ModelCheckpoint(monitor='val_loss', mode='auto', save_best_only=True,
#                      filepath='./_save/ModelCheckPoint/face_age_MCP3_aug5_3.hdf5')

# start_time = time.time()
# hist = model.fit(x_train, y_train, epochs=10000, verbose=2, callbacks=[es, cp], validation_split=0.05, shuffle=True, batch_size=200)
# end_time = time.time() - start_time

# model.save('./_save/ModelCheckPoint/face_age_model_save_aug5_3.h5')

model = load_model('./_save/ModelCheckPoint/face_age_model_save_aug5_6.h5')           # save model
# model = load_model('./_save/ModelCheckPoint/face_age_MCP.hdf5')                # checkpoint

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
ic(results)

y_predict = model.predict(x_pred)
ic(y_predict)
y_predict = tf.argmax(y_predict)
ic(y_predict)


'''
*augment 전
걸린시간 : 6.861128091812134
acc : 0.40430623292922974
val_acc : 0.20454545319080353

*augment 50000
'./_save/ModelCheckPoint/face_age_model_save_aug5_2.h5'
걸린시간 : 197.54050087928772
acc : 0.8777929544448853
val_acc : 0.7338836193084717
val_loss : 0.9268954992294312

'./_save/ModelCheckPoint/face_age_model_save_aug5_3.h5'
걸린시간 : 216.13762092590332
acc : 0.7695713043212891
val_acc : 0.6855345964431763
loss : 0.10000000149011612
val_loss : 0.9884343147277832
'''

real = np.array(y_predict,0)
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