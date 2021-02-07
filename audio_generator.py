from pathlib import Path
from sklearn.preprocessing import LabelEncoder  # type: ignore
from tensorflow.keras.utils import Sequence  # type: ignore
from tqdm import tqdm  # type: ignore
import argparse
import logging
import multiprocessing as mp
import numpy as np  # type: ignore
import typing as ty

from preprocess_data import get_spec_dict
from preprocess_data import wav2mel
from preprocess_data import wav2mfcc
from utils import setup_logger
from utils import words_types


class AudioGenerator(Sequence):
    def __init__(
        self,
        list_IDs: ty.List[str],
        ids2labels: ty.Dict[str, ty.Tuple[str, str]],
        data_split_path: Path,
        batch_size=32,
        dim=(64, 64),
        n_channels=1,
        n_classes=10,
        shuffle=True,
    ) -> None:
        """TODO: what is __init__ doing?

        https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

        partition = {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
        ids2labels = {'id-1': ('happy', '12543_nohash_1'), 'id-2': ('wow',
        '92543_nohash_1'), 'id-3': ('learn', '12843_nohash_1'), 'id-4': ('happy',
        '18240_nohash_1')}

        will be loaded by np.load('data/' + ID + '.npy')
        will be called like train_gen = AudioGenerator(partition['train'], ids2labels, **params)


        >>> lab = ['a', 'b', 'c', 'a']
        >>> y = LabelEncoder().fit_transform(lab)
        >>> y
        array([0, 1, 2, 0])
        >>> cat = to_categorical(y)
        >>> cat
        array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 0., 0.]], dtype=float32)

        >>> le = LabelEncoder()
        >>> le.fit(lab)
        >>> le.classes_
        array(['a', 'b', 'c'], dtype='<U1')
        >>> ind = np.argmax(cat[0])
        >>> ind
        0
        >>> le.inverse_transform([ind])
        array(['a'], dtype='<U1')


        """
        logg = logging.getLogger(f"c.{__name__}.__init__")
        # logg.setLevel("INFO")
        logg.debug("Start __init__")

        self.dim = dim
        self.batch_size = batch_size
        self.ids2labels = ids2labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

        # y_labels: ['happy', 'learn', 'wow', 'visual']
        self.y_labels = sorted(list(set(ids2labels[k] for k in ids2labels)))
        logg.debug(f"self.y_labels: {self.y_labels}")
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.y_labels)

        # the root folder of the processed dataset
        self.data_split_path = data_split_path

    def __len__(self) -> int:
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index) -> ty.Tuple[np.ndarray, np.ndarray]:
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self) -> None:
        "Updates indexes after each epoch"
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Generates data containing batch_size samples

        X : (n_samples, *dim, n_channels)
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y: ty.List[str] = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            word, wav_stem = self.ids2labels[ID]
            file_path = self.data_split_path / word / f"{wav_stem}.npy"
            X[i] = np.load(file_path)
            y.append(word)

        y_hot = self.label_encoder.fit(y)
        return X, y_hot


def parse_arguments() -> argparse.Namespace:
    """Setup CLI interface"""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-pt",
        "--preprocess_type",
        type=str,
        default="preprocess_split",
        choices=["preprocess_split", "test_audio_generator"],
        help="Which preprocess to perform",
    )

    parser.add_argument(
        "-dn",
        "--dataset_name",
        type=str,
        default="mel01",
        help="Name of the output dataset folder",
    )

    parser.add_argument(
        "-wt",
        "--words_type",
        type=str,
        default="f1",
        choices=words_types.keys(),
        help="Words to preprocess",
    )

    parser.add_argument(
        "-fp",
        "--force_preprocess",
        dest="force_preprocess",
        action="store_true",
        help="Force the preprocess and overwrite the previous results",
    )

    # last line to parse the args
    args = parser.parse_args()
    return args


def setup_env() -> argparse.Namespace:
    setup_logger("DEBUG")
    args = parse_arguments()

    # build command string to repeat this run
    # FIXME if an option is a flag this does not work, sorry
    recap = "python3 audio_generator.py"
    for a, v in args._get_kwargs():
        recap += f" --{a} {v}"
    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)

    return args


def compute_folder_spec(
    word_in_folder,
    word_out_folder,
    dataset_name,
    force_preprocess,
    spec_kwargs,
    p2d_kwargs,
) -> None:
    """TODO: what is compute_folder_spec doing?"""
    # extract all the wavs
    all_wavs = list(word_in_folder.iterdir())

    for wav_path in all_wavs:
        wav_stem = wav_path.stem

        spec_path = word_out_folder / f"{wav_stem}.npy"
        if spec_path.exists():
            if force_preprocess:
                pass
            else:
                continue

        if dataset_name.startswith("mfcc"):
            log_spec = wav2mfcc(wav_path, spec_kwargs, p2d_kwargs)
        elif dataset_name.startswith("mel"):
            log_spec = wav2mel(wav_path, spec_kwargs, p2d_kwargs)
        img_spec = log_spec.reshape((*log_spec.shape, 1))

        np.save(spec_path, img_spec)


def preprocess_split(
    dataset_name: str, words_type: str, force_preprocess: bool = False
) -> None:
    """TODO: what is preprocess_split doing?"""
    logg = logging.getLogger(f"c.{__name__}.preprocess_split")
    # logg.setLevel("INFO")
    # logg.debug("Start preprocess_split")

    # original / processed dataset base locations
    data_raw_path = Path("data_raw")
    processed_folder = Path("data_split") / f"{dataset_name}"
    if not processed_folder.exists():
        processed_folder.mkdir(parents=True, exist_ok=True)

    # args for the power_to_db function
    p2d_kwargs = {"ref": np.max}

    # args for the mfcc spec
    spec_dict = get_spec_dict()
    spec_kwargs = spec_dict[dataset_name]

    arguments = []

    words = words_types[words_type]
    for word in words:
        # logg.debug(f"Processing {word}")

        # the output folder path
        word_out_folder = processed_folder / word
        if not word_out_folder.exists():
            word_out_folder.mkdir(parents=True, exist_ok=True)

        # the input folder path
        word_in_folder = data_raw_path / word

        # build the arguments of the functions you will run
        arguments.append(
            (
                word_in_folder,
                word_out_folder,
                dataset_name,
                force_preprocess,
                spec_kwargs,
                p2d_kwargs,
            )
        )

    logg.info(f"dataset_name: {dataset_name}")
    pool = mp.Pool(processes=mp.cpu_count() * 2)
    results = [pool.apply_async(compute_folder_spec, args=a) for a in arguments]
    for p in tqdm(results):
        p.get()


def prepare_partitions(words_type: str) -> None:
    """TODO: what is prepare_partitions doing?"""
    logg = logging.getLogger(f"c.{__name__}.prepare_partitions")
    # logg.setLevel("INFO")
    logg.debug("Start prepare_partitions")

    data_raw_path = Path("data_raw")

    # list of file names for validation
    validation_path = data_raw_path / "validation_list.txt"
    validation_names = []
    with validation_path.open() as fvp:
        for line in fvp:
            validation_names.append(line.strip())

    # list of file names for testing
    testing_path = data_raw_path / "testing_list.txt"
    testing_names = []
    with testing_path.open() as fvp:
        for line in fvp:
            testing_names.append(line.strip())

    partition: ty.Dict[str, ty.List[str]] = {
        "training": [],
        "validation": [],
        "testing": [],
    }
    ids2labels: ty.Dict[str, ty.Tuple[str, str]] = {}

    words = words_types[words_type]
    for word in words:

        # the input folder path
        word_in_folder = data_raw_path / word

        # go through all the words in this folder
        all_wavs = list(word_in_folder.iterdir())
        for wav_path in tqdm(all_wavs):
            wav_stem = wav_path.stem
            wav_name = f"{word}/{wav_path.name}"

            if wav_name in validation_names:
                which = "validation"
            elif wav_name in testing_names:
                which = "testing"
            else:
                which = "training"

            ID = wav_name
            partition[which].append(ID)
            ids2labels[ID] = (word, wav_stem)


def test_audio_generator(which_dataset: str, words_type: str) -> None:
    """TODO: what is test_audio_generator doing?"""
    logg = logging.getLogger(f"c.{__name__}.test_audio_generator")
    # logg.setLevel("INFO")
    logg.debug("Start test_audio_generator")

    prepare_partitions(words_type)


def run_audio_generator(args: argparse.Namespace) -> None:
    """TODO: What is audio_generator doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_audio_generator")
    logg.debug("Starting run_audio_generator")

    which_dataset = args.dataset_name
    words_type = args.words_type
    preprocess_type = args.preprocess_type
    force_preprocess = args.force_preprocess

    if preprocess_type == "preprocess_split":
        preprocess_split(which_dataset, words_type, force_preprocess)
    elif preprocess_type == "test_audio_generator":
        test_audio_generator(which_dataset, words_type)


if __name__ == "__main__":
    args = setup_env()
    run_audio_generator(args)
