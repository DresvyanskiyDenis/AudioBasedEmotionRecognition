import tensorflow as tf
import numpy as np

class CutLayer(tf.keras.layers.Layer):
    def __init__(self, *,chunk_window, max_length, sample_rate):
        super(CutLayer, self).__init__()
        self.C = int(np.floor(max_length/chunk_window))
        self.chunk_window = chunk_window
        self.max_length = max_length
        self.sample_rate = sample_rate

    def call(self, input):
        tf.print('call_function input:', input.shape)
        self.delta_C = (input.shape[1]-self.chunk_window)/(self.C - 1.)
        #print('delta_C:', self.delta_C)
        print("input tensor.shape:", input.shape)
        print("evaluated C:", self.C)
        print("evaluated delta_C:", self.delta_C)
        print("input:", input)
        return self.cut_tensor_on_slices(input, self.C, int(self.chunk_window*self.sample_rate), int(self.delta_C*self.sample_rate))

    def compute_output_shape(self, input_shape):
        return tf.Tensorshape([input_shape[0], self.C, self.chunk_window, input_shape[-1]])


    @tf.function
    def cut_tensor_on_slices(self,tensor, C, window, step):
        tf.print("we are in cut_tensor_on_slices")
        tf.print(tensor.shape)
        #tf.print(tensor)
        cut_tensors=[]
        for i in range(C):
            start=i*step
            end=i*step+window
            tf.print('start:', start)
            tf.print('end:', end)
            slice=tensor[:,start:end]
            slice=tf.expand_dims(slice, axis=1)
            tf.print('slice:',slice.shape)
            cut_tensors.append(slice)
        tf.print("before concat")
        cut_tensors = tf.concat(cut_tensors, axis=1)
        tf.print("after concat")
        tf.print("cut_tensors.shape:",cut_tensors.shape)
        #print('concated cut_tensors:',cut_tensors.shape)
        #print(cut_tensors)
        return cut_tensors



if __name__=="__main__":
    path_to_data=""
    path_to_labels=r"E:\Databases\SEMAINE\SEM_labels_arousal_100Hz_gold_shifted.csv"
    model=tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(18,5)))
    model.add(CutLayer(chunk_window=6, max_length=18, sample_rate=1))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(100, 3,activation="relu")))
    model.compile(optimizer='SGD', loss='mse')
    model.summary()
    tens_1=np.ones((9,6,5))
    #tens=tens*np.array([i for i in range(18)])
    #tens=tens[np.newaxis,...])
    tens_2 = np.ones((8, 10, 5))
    tens_3 = np.ones((7, 18, 5))
    tens_4 = np.ones((9, 14, 5))

    print("----------------------------------------------------")
    print('final output tens_1:',model.predict(tens_1).shape)
    print("----------------------------------------------------")
    print('final output tens_2:', model.predict(tens_2).shape)
    print("----------------------------------------------------")
    print('final output tens_3:', model.predict(tens_3).shape)
    print("----------------------------------------------------")
    print('final output tens_4:', model.predict(tens_4).shape)

