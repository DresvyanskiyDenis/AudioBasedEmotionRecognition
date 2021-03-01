from src.SEWA_audio_emotion_recognition import load_and_split_labels
from src.utils.generators import AudioFixedChunksGenerator

if __name__ == '__main__':
    path_to_data = r"D:\Databases\SEWA\Original\audio"
    path_to_labels = r"D:\Databases\SEWA\SEW_labels_arousal_100Hz_gold_shifted.csv"
    labels = load_and_split_labels(path_to_labels)

    a=1+2
    '''for key, value in labels:
        labels[key]=
    
    
    generator = AudioFixedChunksGenerator(sequence_max_length=, window_length=, 
                                          load_mode='path',
                                          data=None, load_path=None,
                                          data_preprocessing_mode='raw', num_mfcc=128,
                                          labels=labels, labels_type='sequence_to_one', batch_size=4)'''