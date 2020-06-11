import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import time
import tensorflow as tf
from tensorflow import keras
import sklearn
from sklearn.preprocessing import StandardScaler
import pathlib
import random
from PIL import Image
from sklearn.model_selection import train_test_split

data_path = pathlib.Path('./data/trash')
all_image_paths = list(data_path.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
label_names = sorted(item.name for item in data_path.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))

labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
images = []

def read_images(img_name):
    im = Image.open(img_name)
    data = np.array(im)
    data = tf.image.convert_image_dtype(data, dtype=tf.float32) #下面的色彩调整，必须是实数类型
    data = tf.image.random_flip_up_down(data)  #随机上下翻转
    data = tf.image.random_flip_left_right(data)  #随机左右翻转
    data = tf.image.random_brightness(data,max_delta=0.02) #色彩亮度调整，在(-max_delta, max_delta)的范围随机调整图像的亮度,0.1太大了，调整为0.02尝试
    data = tf.image.random_saturation(data, lower=0.9, upper=1.1) #饱和度调整
    data = tf.image.random_hue(data, max_delta=0.1) #调整色相
    data = tf.image.random_contrast(data, lower=0.9, upper=1.1)   #调整对比度
    data = tf.clip_by_value(data, 0.0, 1.0)  #最后对色彩进行调整，值限定再0-1之间,float32的格式，
    return  np.array(data)


def read_imagesnormal(img_name):
    im = Image.open(img_name)
    data = np.array(im)
    data = tf.image.convert_image_dtype(data, dtype=tf.float32) #下面的色彩调整，必须是实数类型
    data = tf.clip_by_value(data, 0.0, 1.0)  #最后对色彩进行调整，值限定再0-1之间,float32的格式，
    return  np.array(data)

for img_name in all_image_paths[:1769]:  #只需要对训练集进行数据预处理
    images.append(read_images(img_name))
for img_name in all_image_paths[1769:]:
    images.append(read_imagesnormal(img_name))
x=np.array(images)
y = np.array(labels)

x_train=x[:1769]
y_train=y[:1769]
x_test=x[1769:]
y_test=y[1769:]

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train.astype(np.float32).reshape(-1,1)).reshape(-1,384,512,3)  #最后一个参数是通道数目
x_test_sacled = scaler.transform(x_test.astype(np.float32).reshape(-1,1)).reshape(-1,384,512,3)



def create_moedel():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu',input_shape=(384,512,3)))   #添加卷积层操作，最后一个参数是通道数目
    model.add(keras.layers.MaxPool2D(pool_size=2,strides=2))

    model.add(keras.layers.Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=2,strides=2))

    model.add(keras.layers.Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
    model.add(keras.layers.Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=2,strides=2))

    model.add(keras.layers.Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
    model.add(keras.layers.Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=2,strides=2))

    model.add(keras.layers.Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
    model.add(keras.layers.Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=2,strides=2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(1024,kernel_regularizer=keras.regularizers.l2(0.05),activation='relu'))   #L2正则化参数调整到0.05尝试
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(1024,kernel_regularizer=keras.regularizers.l2(0.05),activation='relu'))
    model.add(keras.layers.Dense(6,activation='softmax'))
    model.compile(loss="sparse_categorical_crossentropy",optimizer=tf.keras.optimizers.SGD(learning_rate=0.003),metrics =['accuracy'])   #学习率也可以再调低一些0.004
    return model
tb = tf.keras.callbacks.TensorBoard(log_dir='logs',  # log 目录
                 histogram_freq=5,  # 按照何等频率（epoch）来计算直方图，0为不计算
                 batch_size=40,     # 用多大量的数据计算直方图
                 write_graph=True,  # 是否存储网络结构图
                 write_grads=False, # 是否可视化梯度直方图
                 write_images=True,# 是否可视化参数
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                  embeddings_metadata=None)
model = create_moedel()
print(model.summary())
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',patience=4) #连续10个epoch 验证集loss没有降低将停止训练
history = model.fit(x_train_scaled,y_train,batch_size=20 ,epochs=100,validation_data=(x_test_sacled,y_test),callbacks=[early_stop,tb]) #一次给神经网络10个样本，批处理，
print(history.history)
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True) #显示网格
    plt.gca().set_ylim(0,1)
    plt.show()
plot_learning_curves(history)



'''保存模型
model.save('my_model.h5')
new_model=keras.models.load_model('my_model.h5')
new_model.summary()
loss,acc=new_model.evaluate(x_test_sacled,y_test)
print("Model accuracy:"+acc)
'''

'''保存权重

model.save_weights('./checkpoints/my_checkpoint')
new2_model=create_model()
new2_model.load_weights('./checkpoints/my_checkpoint')
loss,acc=new2_model.evaluate(x_test_sacled,y_test)
print("Model accuracy:"+acc)
'''


'''
checkpoint_path="training_1/cp-1.ckpt"
checkpoint_dir=os.path.dirname(checkpoint_path)
cp_callback=tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_weight_only=True,verbose=1,period=10) #10次记录一次
new3_model.fit(callbacks=[cp_callback]) #训练时放进入

latest=tf.train.latest_checkpoint(checkpoint_dir)
new4_model.load_weights(latest) #放进新模型

'''
