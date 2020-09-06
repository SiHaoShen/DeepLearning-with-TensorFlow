import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#print out 0, 1, 2...9
dataset = tf.data.Dataset.range(10)
for val in dataset:
    print(val.numpy())

#Print out [0 1 2 3 4], [1 2 3 4 5]... [8 9] [9]
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1)
for window_dataset in dataset:
    for val in window_dataset:
        print(val.numpy(), end=" ")
    print()

#Print out [0 1 2 3 4], ... [5 6 7 8 9]
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
for window in dataset:
    print(window.numpy())

#Print out random shuffled x=[[2 3 4 5] [4 5 6 7]] y= [[6] [8]]...
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:])) #Separate the last value
dataset = dataset.shuffle(buffer_size = 10)
dataset = dataset.batch(2).prefetch(1)
for x,y in dataset:
    print("x = ", x.numpy())
    print("y = ", y.numpy())

