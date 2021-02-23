# Keyword Spotting Project

### How to run the demo

```
pip install -r requirements.txt
python demo_stream.py
```

Please note that some libraries need external dependencies (e.g. audioread/sounddevice).
I do not know precisely the list of external libraries needed.

For `sounddevice` to work `libportaudio` must be installed.

```
sudo apt install libportaudio2
```

### How to run the training

Please install the python dependencies.

```
pip install -r requirements.txt
```

Please place the Google Speech command dataset in the `data_raw` folder,
inside the `hda-speech-recog` folder:

```
hda-speech-recog/data_raw
```

Be careful to include the validation and testing list provided with the dataset.

To have access to the
[FSDD](https://github.com/Jakobovski/free-spoken-digit-dataset)
/
[LibriTTS](https://www.tensorflow.org/datasets/catalog/libritts)
datasets, please download them in 

```
~/audiodatasets/LibriTTS/dev-clean/
~/audiodatasets/free_spoken_digit_dataset/
```

or change the source folders in the corresponding preprocess script.

Please prepare the loudest section only dataset, if needed

```
python gs_dataset.py
```

The rest of the preprocessing *should* automatically work from the training
scripts, simply run

```
python train_{arch}.py
```

Where `arch` is the selected type of training.

The script will execute the `hyper_train_{arch}` function, no particular values
of hyper-parameters are set.

The augmentation preprocess does not work on one of the machines where I tested
the code, unless the following option is set:

```
export CUDA_VISIBLE_DEVICES="-1"
```

It is suggested to re-enable CUDA after preprocessing.

Generally speaking, running `script_name.py -h` gives a decent idea of what
that script can do.

Utils script and dataset generators (listed below) are not meant to be run alone.

### Files

##### Training scripts

* `train_area.py`
* `train_attention.py`
* `train_cnn.py`
* `train_imagenet.py`
* `train_transfer.py`

##### Evaluation scripts

* `evaluate_all.py`
* `evaluate_area.py`
* `evaluate_attention.py`
* `evaluate_cnn.py`
* `evaluate_stream.py`
* `evaluate_transfer.py`
* `visualize.py`

##### Preprocessing scripts

* `augment_data.py`
* `background_noise.py`
* `fsdd_preprocess.py`
* `gs_dataset.py`
* `ljspeech_preprocess.py`
* `ltts_preprocess.py`
* `preprocess_data.py`

##### Model files

* `area_model.py`
* `models.py`

##### Dataset generators

* `audio_generator.py`
* `imagenet_generator.py`

##### Demo

* `demo_stream.py`

##### Utils

* `clr_callback.py`
* `keras_diagram.py`
* `lr_finder.py`
* `plot_utils.py`
* `renamer.py`
* `schedules.py`
* `utils.py`
