import librosa

def preprocessing( audio ):
    
    filter_audio = librosa.effects.preemphasis( audio )

    return filter_audio

## Calcular los MFCC
def vectorizer( audio, sample_rate ):

    filter_audio = preprocessing( audio )
    mfccs = librosa.feature.mfcc(y = filter_audio, sr = sample_rate, n_mfcc=13)
    
    return mfccs