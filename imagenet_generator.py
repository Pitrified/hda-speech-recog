from pathlib import Path
from timeit import default_timer as timer
import argparse
import logging
import typing as ty

from PIL import Image  # type: ignore
from sklearn.preprocessing import LabelEncoder  # type: ignore
from tensorflow.keras.utils import Sequence  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
import numpy as np  # type: ignore

# from random import seed as rseed


def prepare_partitions(
    label_list: ty.List[str],
    dataset_raw_folder: Path,
    test_fold_size: float = 0.1,
    val_fold_size: float = 0.1,
    rng: np.random.Generator = None,
) -> ty.Tuple[ty.Dict[str, ty.List[str]], ty.Dict[str, ty.Tuple[str, str]]]:
    r"""MAKEDOC: what is prepare_partitions doing?"""
    # logg = logging.getLogger(f"c.{__name__}.prepare_partitions")
    # logg.setLevel("INFO")
    # logg.debug("Start prepare_partitions")

    partition: ty.Dict[str, ty.List[str]] = {
        "training": [],
        "validation": [],
        "testing": [],
    }
    ids2labels: ty.Dict[str, ty.Tuple[str, str]] = {}

    # get a new rng, if you want repeatable folds pass your own
    if rng is None:
        rng = np.random.default_rng(int(timer()))

    # analyse all interesting classes
    for label in label_list:

        label_folder = dataset_raw_folder / label

        for img_raw_path in label_folder.iterdir():
            img_stem = img_raw_path.stem
            img_name = f"{label}/{img_raw_path.name}"

            x = rng.random()
            if x < test_fold_size:
                which = "testing"
            elif x < test_fold_size + val_fold_size:
                which = "validation"
            else:
                which = "training"

            ID = img_name
            partition[which].append(ID)
            ids2labels[ID] = (label, img_stem)

    return partition, ids2labels


class ImageNetGenerator(Sequence):
    def __init__(
        self,
        list_IDs: ty.List[str],
        ids2labels: ty.Dict[str, ty.Tuple[str, str]],
        label_names: ty.List[str],
        dataset_proc_folder: ty.Optional[Path],
        dataset_raw_folder: Path,
        preprocess_type: str,
        save_processed: bool,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> None:
        r"""MAKEDOC: what is __init__ doing?

        https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

        partition = {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
        ids2labels = {'id-1': ('happy', '12543_nohash_1'), ...}

        file names are either
            data/image_raw/cicada/cicada_0001.jpg
            data/aug05/cicada/cicada_0001.npy
        """

        logg = logging.getLogger(f"c.{__name__}.__init__")
        # logg.setLevel("INFO")
        logg.debug("Start __init__")

        # info on the data and labels
        self.list_IDs = list_IDs
        self.ids2labels = ids2labels
        self.label_names = label_names

        # a LabelEncoder to go from label name to index realiably
        # then use to_categorical to get one hot encoded
        # to reverse, get the indexes with argmax from the predictions
        # and use inverse_transform
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.label_names)

        # the root folder of the original dataset (data/image_raw)
        self.dataset_raw_folder = dataset_raw_folder
        # the root folder of the processed dataset (data/aug05)
        self.dataset_proc_folder = dataset_proc_folder
        # the type of preprocessing to do (aug05)
        self.preprocess_type = preprocess_type
        # whether to save the processed images
        self.save_processed = save_processed

        if self.save_processed:
            if self.dataset_proc_folder is None:
                logg.warn("No processed folder passed, will not save output.")
                self.save_processed = False

        # define the preprocess_params to use
        self.define_preprocess_types()

        # info on the shape of the batch to generate
        self.batch_size = batch_size
        self.n_classes = len(self.label_names)
        self.img_shape = self.get_img_shape()

        # whether to shuffle or not the data
        self.shuffle = shuffle

        # call on_epoch_end to generate the indexes to use
        self.on_epoch_end()

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

    def define_preprocess_types(self) -> None:
        r"""MAKEDOC: what is define_preprocess_types doing?"""
        logg = logging.getLogger(f"c.{__name__}.define_preprocess_types")
        # logg.setLevel("INFO")
        logg.debug("Start define_preprocess_types")

        # all the known preprocess params
        self.all_preprocess_params = {}

        # aug01 params
        preprocess_params = {
            "img_shape": (256, 256, 3),
        }
        self.all_preprocess_params["aug01"] = preprocess_params

        # set the current ones
        self.preprocess_params = self.all_preprocess_params[self.preprocess_type]

    def get_img_shape(self) -> ty.Tuple[int, int, int]:
        r"""MAKEDOC: what is get_img_shape doing?"""
        logg = logging.getLogger(f"c.{__name__}.get_img_shape")
        # logg.setLevel("INFO")
        logg.debug("Start get_img_shape")

        return self.preprocess_params["img_shape"]

    def __preprocess_img(self, label: str, img_stem: str) -> np.ndarray:
        r"""MAKEDOC: what is __preprocess_img doing?

        Image.resize(size, resample=3, box=None, reducing_gap=None)
        box – An optional 4-tuple of floats providing the source image region to be
        scaled. The values must be within (0, 0, width, height) rectangle

        """
        logg = logging.getLogger(f"c.{__name__}.__preprocess_img")
        # logg.setLevel("INFO")
        logg.debug("Start __preprocess_img")

        # the original image
        img_raw_path = self.dataset_raw_folder / label / f"{img_stem}.jpg"

        # load the PIL Image
        img_pil = Image.open(img_raw_path)
        # img_pil.show()
        # logg.debug(img_pil.format)
        # logg.debug(img_pil.mode)

        width, height = img_pil.size
        logg.debug(f"width: {width} height: {height}")

        resize_size = self.img_shape[0 : 1 + 1]
        res_width, res_height = resize_size

        # resize the image according to the specified preprocess_type
        if self.preprocess_type == "aug01":

            # FIXME if an image is smaller than the resize_size it will fail

            # we need to resize it doooown
            if width == height:
                box = (0, 0, width, height)

            # tall image
            elif width < height:
                pad = height - res_height
                top = pad // 2
                bottom = height - pad // 2
                box = (0, top, width, bottom)
                logg.debug(f"TALL bottom-top: {bottom-top}")

            elif width > height:
                pad = width - res_width
                left = pad // 2
                right = width - pad // 2
                box = (left, 0, right, height)

            logg.debug(f"box: {box}")
            img_resized = img_pil.resize(resize_size, box=box)
            logg.debug(f"img_resized.size: {img_resized.size}")
            # img_resized.show()

        # convert to numpy
        img_np = np.array(img_resized)
        logg.debug(f"img_np.shape: {img_np.shape}")

        # save the processed image
        if self.save_processed and self.dataset_proc_folder is not None:
            img_proc_path = self.dataset_proc_folder / f"{img_stem}.npy"
            np.save(img_proc_path, None)

        return img_np

    def __data_generation(self, list_IDs_temp) -> ty.Tuple[np.ndarray, np.ndarray]:
        """Generates data containing batch_size samples

        X : (n_samples, *img_shape, n_channels)
        """
        # Initialization
        X = np.empty((self.batch_size, *self.img_shape))
        y: ty.List[str] = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # extract path info for this word
            label, img_stem = self.ids2labels[ID]

            load_available = False

            if self.dataset_proc_folder is not None:
                # generate the processed path
                img_proc_path = self.dataset_proc_folder / f"{img_stem}.npy"
                # if the image is already processed with this tag, load it
                if img_proc_path.exists():
                    load_available = True

            if load_available:
                img = np.load(img_proc_path)
            else:
                img = self.__preprocess_img(label, img_stem)

            # save the image, the shape should fit in X
            X[i] = img
            # save the label for this image
            y.append(label)

        y_le = self.label_encoder.transform(y)
        y_hot = to_categorical(y_le, num_classes=self.n_classes)
        return X, y_hot


def parse_arguments() -> argparse.Namespace:
    r"""Setup CLI interface"""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-lld",
        "--log_level_debug",
        type=str,
        default="DEBUG",
        help="Level for the debugging logger",
        choices=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"],
    )

    parser.add_argument(
        "-llt",
        "--log_level_type",
        type=str,
        default="m",
        help="Message format for the debugging logger",
        choices=["anlm", "nlm", "lm", "nm", "m"],
    )

    # last line to parse the args
    args = parser.parse_args()
    return args


def setup_logger(logLevel: str = "DEBUG", msg_type: str = "m") -> None:
    r"""Setup logger that outputs to console for the module"""
    logroot = logging.getLogger("c")
    logroot.propagate = False
    logroot.setLevel(logLevel)

    module_console_handler = logging.StreamHandler()

    if msg_type == "anlm":
        log_format_module = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    elif msg_type == "nlm":
        log_format_module = "%(name)s - %(levelname)s: %(message)s"
    elif msg_type == "lm":
        log_format_module = "%(levelname)s: %(message)s"
    elif msg_type == "nm":
        log_format_module = "%(name)s: %(message)s"
    else:
        log_format_module = "%(message)s"

    formatter = logging.Formatter(log_format_module)
    module_console_handler.setFormatter(formatter)

    logroot.addHandler(module_console_handler)


def setup_env() -> argparse.Namespace:
    r"""Setup the logger and parse the args"""
    args = parse_arguments()
    setup_logger(args.log_level_debug, args.log_level_type)

    # build command string to repeat this run
    # FIXME if an option is a flag this does not work, sorry
    recap = "python3 imagenet_preprocess.py"
    for a, v in args._get_kwargs():
        recap += f" --{a} {v}"

    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)

    return args


def test_imagenetgenerator() -> None:
    r"""MAKEDOC: what is test_imagenetgenerator doing?"""
    logg = logging.getLogger(f"c.{__name__}.test_imagenetgenerator")
    # logg.setLevel("INFO")
    logg.debug("Start test_imagenetgenerator")

    dataset_raw_folder = Path.home() / "datasets" / "imagenet" / "imagenet_images"
    dataset_proc_base_folder = Path.home() / "datasets" / "imagenet"

    label_list = ["heron", "cicada"]

    rng = np.random.default_rng(12345)

    # get partitions
    partition, ids2labels = prepare_partitions(label_list, dataset_raw_folder, rng=rng)

    # check partitions
    for fold in partition:
        recap = f"len(partition[{fold}]) {len(partition[fold])}"
        recap += f"\tpartition[{fold}][:2]: {partition[fold][:2]}"
        logg.debug(recap)
    logg.debug(f"len(ids2labels): {len(ids2labels)}")
    for ID in ids2labels:
        logg.debug(f"ids2labels[{ID}]: {ids2labels[ID]}")
        break

    preprocess_type = "aug01"
    dataset_proc_folder = dataset_proc_base_folder / preprocess_type
    save_processed = False
    batch_size = 2
    shuffle = True

    training_generator = ImageNetGenerator(
        partition["training"],
        ids2labels,
        label_list,
        dataset_proc_folder=dataset_proc_folder,
        dataset_raw_folder=dataset_raw_folder,
        preprocess_type=preprocess_type,
        save_processed=save_processed,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    logg.debug(f"len(training_generator): {len(training_generator)}")

    X, y = training_generator[0]
    logg.debug(f"X.shape: {X.shape} y.shape: {y.shape}")


def run_imagenet_generator(args: argparse.Namespace) -> None:
    r"""TODO: What is imagenet_generator doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_imagenet_generator")
    logg.debug("Starting run_imagenet_generator")

    test_imagenetgenerator()


if __name__ == "__main__":
    args = setup_env()
    run_imagenet_generator(args)
