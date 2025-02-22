# TODOs

### AreaNet

* DO THE LR FINDER! It's easy, and you know it.
* Pool more in the freq axis to get a single value for each time step
* MAYBE just get the average then upsample in the vertical axis
* More AreaNet

### Misc

##### FixMe

* Pass label translations to plotter and use them, for outer parameter add on what you averaged
* other ltts is not generated if it is missing the raw folder
* All the documentation

##### Demo

* Constantly compute predictions in the demo
* Show inference time for each one
* Show spectrograms

* https://python-sounddevice.readthedocs.io/en/0.3.15/examples.html#plot-microphone-signal-s-in-real-time
* https://python-sounddevice.readthedocs.io/en/0.3.15/api/streams.html

##### Improvements

* Train of FSDD to check training performances on small datasets
* Add audiogenerator already with files save in batches of 16
* Multiple training of the same hypas to check consistency, build model name takes a param and finds the first `_00x` free
* Top one error as metric (described in the Google dataset paper)
* Analyze fsdd and increase volume?
* Add silence and unknown labels to learn
* Nel paper di Xception ti dicono che parametri usare per gli optimizer
* Mode data augmentation, mask, only roll, only stratch...
* When padding use -80 or whatever the min of the spectrogram is
* MAYBE Normalize the spectrograms

##### Done

* Fix augment
* Better result aggregator (num LTnum ...)
* spoken\_digit dataset as final test, ljspeech, vctk.
* https://www.pyimagesearch.com/faqs/single-faq/how-do-i-reference-or-cite-one-of-your-blog-posts-books-or-courses/
* Check fscore in CNN: when doing evaluate precision/recall are at 95 then fscore is 87
* DataGenerator https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

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

General LR

* https://www.pyimagesearch.com/2019/07/22/keras-learning-rate-schedules-and-decay/
