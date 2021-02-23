import tensorflow as tf
import numpy as np

class MyLayer(tf.keras.layers.Layer):
    def __init__(self, chunk_window, max_length, sample_rate):
        super(MyLayer, self).__init__()
        self.C = int(np.floor(max_length/chunk_window))
        self.chunk_window = chunk_window
        self.max_length = max_length
        self.sample_rate = sample_rate

    def call(self, input):
        print('input:', input)
        self.delta_C = (input.shape[1]-self.chunk_window)/(self.C - 1.)
        print('delta_C:', self.delta_C)
        return cut_tensor_on_slices(input, self.C, int(self.chunk_window*self.sample_rate), int(self.delta_C*self.sample_rate))


@tf.function
def cut_tensor_on_slices(tensor, C, window, step):
    print(tensor.shape)
    cut_tensors=[]
    for i in range(C):
        start=i*step
        end=i*step+window
        print('start:', start)
        print('end:', end)
        slice=tensor[:,start:end]
        slice=tf.expand_dims(slice, axis=1)
        print('slice %i:'%(i),slice)
        cut_tensors.append(slice)
    cut_tensors = tf.concat(cut_tensors, axis=1)
    print('concated cut_tensors:',cut_tensors.shape)
    print(cut_tensors)
    return cut_tensors



if __name__=="__main__":
    path_to_data=""
    path_to_labels=r"E:\Databases\SEMAINE\SEM_labels_arousal_100Hz_gold_shifted.csv"
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(12,5)))
    model.add(MyLayer(6, 18, 1))
    model.compile(optimizer='SGD', loss='mse')
    model.summary()
    tens_1=np.ones((9,6,5))
    #tens=tens*np.array([i for i in range(18)])
    #tens=tens[np.newaxis,...])
    tens_2 = np.ones((9, 10, 5))
    tens_3 = np.ones((9, 10, 5))
    tens_4 = np.ones((9, 10, 5))



    print('final output tens_1:',model.predict(tens_1).shape)
    print('final output tens_2:', model.predict(tens_1).shape)
    print('final output tens_3:', model.predict(tens_1).shape)
    print('final output tens_4:', model.predict(tens_1).shape)

