# Keyword Spotting Project

### How to run the demo

```
pip install -r requirements.txt
python demo_stream.py
```

Note that some libraries need external dependencies (e.g. audioread/sounddevice).
I do not know precisely the list of external libraries needed.

### Files

##### Model files

* `area_model.py`
* `models.py`

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
