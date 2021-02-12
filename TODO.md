# TODOs

### Misc

* other ltts is not generated if it is missing the raw folder
* Better result aggregator (num LTnum ...)
* Fix augment

* Top one error as metric (described in the Google dataset paper)
* spoken\_digit dataset as final test, ljspeech, vctk.
* Analyze fsdd and increase volume?
* Add silence and unknown labels to learn
* Nel paper di Xception ti dicono che parametri usare per gli optimizer
* https://www.pyimagesearch.com/faqs/single-faq/how-do-i-reference-or-cite-one-of-your-blog-posts-books-or-courses/
* Mode data augmentation, mask, only roll, only stratch...
* All the documentation
* More AreaNet
* Multiple training of the same hypas to check consistency, build model name takes a param and finds the first `_00x` free

* When padding use -80 or whatever the min of the spectrogram is

### More datasets

For background noise

* https://www.tensorflow.org/datasets/catalog/fuss

##### Mozilla common voice

* https://commonvoice.mozilla.org/en/datasets

##### LibriSpeech

LibriSpeech is a corpus of approximately 1000 hours of read English speech with sampling rate of 16 kHz

* http://www.openslr.org/12

##### LibriTTS

LibriTTS is a multi-speaker English corpus of approximately 585 hours of read English speech at 24kHz sampling rate
The audio files are at 24kHz sampling rate. The speech is split at sentence breaks.

* http://www.openslr.org/60

##### Ljspeech

This is a public domain speech dataset consisting of 13,100 short audio clips of a single speaker reading passages from 7 non-fiction books.

* https://keithito.com/LJ-Speech-Dataset/

##### Done

* Check fscore in CNN: when doing evaluate precision/recall are at 95 then fscore is 87
* DataGenerator https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

### Transfer learning

##### Done

* Use different base models in TRA

### Data augmentation

##### Done

* https://github.com/pyyush/SpecAugment/blob/master/augment.py
* https://www.kaggle.com/CVxTz/audio-data-augmentation/notebook#Data-augmentation-definition-:

### Learning rate schedule

##### Done

Function callback:

* https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler
* The func might take more inputs, use partial to do hypa tune on it

ReduceLROnPlateau:

* https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau

Cyclic learning rate:

* https://github.com/JonnoFTW/keras_find_lr_on_plateau
* https://github.com/bckenstler/CLR/blob/master/clr_callback.py
* https://www.pyimagesearch.com/2019/07/29/cyclical-learning-rates-with-keras-and-deep-learning/

* https://www.pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/
