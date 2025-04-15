import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

# إعداد المسارات الخاصة بالتدريب والاختبار
train_dir = 'train'
test_dir = 'test'

row, col = 48, 48
classes = 7

# تعريف دالة لحساب عدد الصور في كل فئة
def count_exp(path, set_):
    dict_ = {}
    for expression in os.listdir(path):
        dir_ = os.path.join(path, expression)
        if os.path.isdir(dir_):
            dict_[expression] = len(os.listdir(dir_))
    df = pd.DataFrame(dict_, index=[set_])
    return df


train_count = count_exp(train_dir, 'train')
test_count = count_exp(test_dir, 'test')

print(train_count)
print(test_count)

train_count.transpose().plot(kind='bar')


plt.title('Train Set Image Counts')
plt.xlabel('Expression')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(14,22))
i = 1
for expression in os.listdir(train_dir):
    img_path = os.path.join(train_dir, expression, os.listdir(os.path.join(train_dir, expression))[5])
    img = load_img(img_path)
    plt.subplot(1, 7, i)  
    plt.imshow(img)
    plt.title(expression)
    plt.axis('off')  
    i += 1
plt.show()

# تحضير البيانات
train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True,
                                   validation_split=0.2)

# تحميل مجموعة البيانات
training_set = train_datagen.flow_from_directory(train_dir,
                                                batch_size=64,
                                                target_size=(48,48),
                                                shuffle=True,
                                                color_mode='grayscale',
                                                class_mode='categorical',
                                                subset='training')

validation_set = train_datagen.flow_from_directory(train_dir,
                                                batch_size=64,
                                                target_size=(48,48),
                                                shuffle=True,
                                                color_mode='grayscale',
                                                class_mode='categorical',
                                                subset='validation')

test_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True)

test_set = test_datagen.flow_from_directory(test_dir,
                                                batch_size=64,
                                                target_size=(48,48),
                                                shuffle=True,
                                                color_mode='grayscale',
                                                class_mode='categorical')

# بناء النموذج
model = tf.keras.Sequential([
    Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)),  # تم تحديد الشكل هنا بشكل صحيح
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(classes, activation='softmax')
])

# إعداد المُحسن
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# إعداد المحفزات
checkpointer = [EarlyStopping(monitor='val_accuracy', verbose=1, restore_best_weights=True, mode="max", patience=10),
                ModelCheckpoint(filepath='model.weights.best.keras', monitor="val_accuracy", verbose=1, save_best_only=True, mode="max")]

# حساب عدد الخطوات
steps_per_epoch = training_set.n // training_set.batch_size
validation_steps = validation_set.n // validation_set.batch_size

# # تدريب النموذج
# history = model.fit(
#     x=training_set,
#     validation_data=validation_set,
#     epochs=200,
#     callbacks=[checkpointer],
#     steps_per_epoch=steps_per_epoch,
#     validation_steps=validation_steps
# )
# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, val_loss, 'b-')
plt.legend(['Training Loss', 'Val Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()