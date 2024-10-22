from pathlib import Path
from tensorflow.keras import models  # type: ignore
import argparse
import json
import librosa  # type: ignore
import logging
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import re
import typing as ty

from augment_data import augment_signals
from augment_data import compute_spectrograms
from preprocess_data import get_spec_dict
from preprocess_data import get_spec_shape_dict
from train_area import build_area_name
from train_attention import build_attention_name
from train_cnn import build_cnn_name
from utils import setup_gpus
from utils import setup_logger
from utils import words_types


def parse_arguments() -> argparse.Namespace:
    """Setup CLI interface"""
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-et",
        "--evaluation_type",
        type=str,
        default="ltts",
        choices=["ltts"],
        help="Which evaluation to perform",
    )

    parser.add_argument(
        "-at",
        "--architecture_type",
        type=str,
        default="cnn",
        choices=["cnn", "attention", "area"],
        help="Which architecture to use",
    )

    tra_types = [w for w in words_types.keys() if not w.startswith("_")]
    parser.add_argument(
        "-tw",
        "--train_words_type",
        type=str,
        default="f2",
        choices=tra_types,
        help="Words the dataset was trained on",
    )

    parser.add_argument(
        "-dn",
        "--dataset_name",
        type=str,
        default="mel01",
        help="Name of the dataset folder",
    )

    parser.add_argument(
        "-sid",
        "--sentence_index",
        type=int,
        default=0,
        help="Which sentence to show",
    )

    # last line to parse the args
    args = parser.parse_args()
    return args


def setup_env() -> argparse.Namespace:
    setup_logger("DEBUG")
    args = parse_arguments()
    # build command string to repeat this run
    # FIXME if an option is a flag this does not work, sorry
    recap = "python3 evaluate_stream.py"
    for a, v in args._get_kwargs():
        recap += f" --{a} {v}"
    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)
    return args


def build_ltts_sentence_list(
    train_words_type: str, do_filter_match: bool, do_filter_len: bool
) -> ty.Tuple[ty.Dict[str, Path], ty.Dict[str, str]]:
    """MAKEDOC: what is build_ltts_sentence_list doing?


    do_filter_match set to True only keeps sentences that contain a train word
    do_filter_len set to True only keeps sentences s.t. 4 < len_sen < 15
    """
    logg = logging.getLogger(f"c.{__name__}.build_ltts_sentence_list")
    # logg.setLevel("INFO")
    logg.debug("Start build_ltts_sentence_list")

    # the location of this file
    this_file_folder = Path(__file__).parent.absolute()
    logg.debug(f"this_file_folder: {this_file_folder}")

    # the location of the original dataset
    ltts_base_folder = Path.home() / "audiodatasets" / "LibriTTS" / "dev-clean"
    logg.debug(f"ltts_base_folder: {ltts_base_folder}")

    # get the words
    # train_words = words_types["all"]
    if "num" in train_words_type:
        train_words = words_types["num"]

    train_words_bound = [fr"\b{w}\b" for w in train_words]
    # logg.debug(f"train_words_bound: {train_words_bound}")
    train_words_re = re.compile("|".join(train_words_bound))
    # logg.debug(f"train_words_re: {train_words_re}")

    # build good_wav_paths dict
    # from wav_ID to orig_wav_path
    good_wav_paths: ty.Dict[str, Path] = {}

    # build good_sentences dict
    # from wav_ID to norm_tra
    good_sentences: ty.Dict[str, str] = {}

    no_match_count = 0

    for reader_ID_path in ltts_base_folder.iterdir():
        for chapter_ID_path in reader_ID_path.iterdir():
            for file_path in chapter_ID_path.iterdir():

                # extract the name of the file
                file_name = file_path.name

                # only process normalized files
                if "normalized" not in file_name:
                    continue
                # logg.debug(f"file_path: {file_path}")

                # read the normalized transcription
                norm_tra = file_path.read_text()

                # select this row if one of the training words is in the sentence
                match = train_words_re.search(norm_tra)
                if match is None:
                    no_match_count += 1
                    # and if we *want* to filter out the ones without
                    if do_filter_match:
                        continue

                # filter sentences that are too long or short
                if do_filter_len:
                    if not 4 < len(norm_tra.split()) < 15:
                        continue

                # build the wav path
                #    file_path /path/to/file/wav_ID.normalized.txt
                #    with stem extract wav_ID.normalized
                #    then remove .normalized
                wav_ID = file_path.stem[:-11]
                # logg.debug(f"wav_ID: {wav_ID}")
                orig_wav_path = chapter_ID_path / f"{wav_ID}.wav"

                # save the file path
                good_wav_paths[wav_ID] = orig_wav_path
                good_sentences[wav_ID] = norm_tra

    logg.debug(f"no_match_count: {no_match_count}")
    len_good = len(good_wav_paths)
    logg.debug(f"len_good: {len_good}")

    return good_wav_paths, good_sentences


def split_sentence(
    sentence_sig: np.ndarray,
    spec_type: str,
    sentence_hop_length: int,
    split_length: int,
    new_sr: int = 16000,
) -> ty.List[np.ndarray]:
    """MAKEDOC: what is split_sentence doing?"""
    logg = logging.getLogger(f"c.{__name__}.split_sentence")
    logg.setLevel("INFO")
    logg.debug("Start split_sentence")

    # split the sentence here
    splits: ty.List[np.ndarray] = []

    # count the splits
    num_split = (len(sentence_sig) - new_sr) // sentence_hop_length
    logg.debug(f"num_split: {num_split}")

    for split_index in range(num_split):
        start_index = split_index * sentence_hop_length
        end_index = start_index + split_length
        split = sentence_sig[start_index:end_index]
        logg.debug(f"split.shape: {split.shape}")
        splits.append(split)

    return splits


def load_trained_model_att(
    override_hypa, do_load: bool = True
) -> ty.Tuple[models.Model, str]:
    """MAKEDOC: what is load_trained_model_att doing?"""
    logg = logging.getLogger(f"c.{__name__}.load_trained_model_att")
    # logg.setLevel("INFO")
    logg.debug("Start load_trained_model_att")
    logg.debug(f"override_hypa: {override_hypa}")

    # default values for the hypas
    hypa = {
        "batch_size_type": "02",
        "conv_size_type": "02",
        "dense_width_type": "01",
        "dropout_type": "01",
        "epoch_num_type": "02",
        "kernel_size_type": "01",
        "learning_rate_type": "03",
        "lstm_units_type": "01",
        "optimizer_type": "a1",
        "query_style_type": "01",
    }
    use_validation = True

    # # for num
    # hypa = {
    #     "batch_size_type": "02",
    #     "conv_size_type": "02",
    #     # "dataset_name": "mel04",
    #     "dense_width_type": "01",
    #     "dropout_type": "01",
    #     "epoch_num_type": "01",
    #     "kernel_size_type": "01",
    #     "learning_rate_type": "04",
    #     "lstm_units_type": "01",
    #     "optimizer_type": "a1",
    #     "query_style_type": "05",
    #     # "words_type": "num"
    # }
    # use_validation = False

    # override the values
    for hypa_name in override_hypa:
        hypa[hypa_name] = override_hypa[hypa_name]

    model_name = build_attention_name(hypa, use_validation)
    logg.debug(f"model_name: {model_name}")

    if do_load:
        # model_folder = Path("trained_models") / "attention"
        model_folder = Path("good_models") / "attention"
        model_path = model_folder / f"{model_name}.h5"
        if not model_path.exists():
            logg.error(f"Model not found at: {model_path}")
            logg.error(f"Train it with hypa_grid = {hypa}")
            raise FileNotFoundError
        model = models.load_model(model_path)

    else:
        model = None

    return model, model_name


def load_trained_model_cnn(
    override_hypa, do_load: bool = True
) -> ty.Tuple[models.Model, str]:
    """MAKEDOC: what is load_trained_model_cnn doing?"""
    logg = logging.getLogger(f"c.{__name__}.load_trained_model_cnn")
    # logg.setLevel("INFO")
    logg.debug("Start load_trained_model_cnn")

    # default values for the hypas
    hypa: ty.Dict[str, ty.Union[str, int]] = {
        "base_dense_width": 32,
        "base_filters": 32,
        "batch_size": 32,
        "dataset": "aug07",
        "dropout_type": "01",
        "epoch_num": 15,
        "kernel_size_type": "02",
        "learning_rate_type": "04",
        "optimizer_type": "a1",
        "pool_size_type": "01",
        "words": "all",
    }

    # override the values
    for hypa_name in override_hypa:
        hypa[hypa_name] = override_hypa[hypa_name]

    model_name = build_cnn_name(hypa)
    logg.debug(f"model_name: {model_name}")

    if do_load:
        model_folder = Path("trained_models") / "cnn"
        model_path = model_folder / f"{model_name}.h5"
        if not model_path.exists():
            logg.error(f"Model not found at: {model_path}")
            logg.error(f"Train it with hypa_grid = {hypa}")
            raise FileNotFoundError

        model = models.load_model(model_path)

    else:
        model = None

    return model, model_name


def load_trained_model_area(
    override_hypa, do_load: bool = True
) -> ty.Tuple[models.Model, str]:
    """MAKEDOC: what is load_trained_model_area doing?"""
    logg = logging.getLogger(f"c.{__name__}.load_trained_model_area")
    # logg.setLevel("INFO")
    logg.debug("Start load_trained_model_area")

    hypa = {
        "batch_size_type": "32",
        "epoch_num_type": "15",
        "learning_rate_type": "03",
        # "net_type": "VAN",
        "net_type": "AAN",
        "optimizer_type": "a1",
    }
    # use_validation = False
    use_validation = True

    # override the values
    for hypa_name in override_hypa:
        hypa[hypa_name] = override_hypa[hypa_name]

    model_name = build_area_name(hypa, use_validation)
    logg.debug(f"model_name: {model_name}")

    if do_load:
        model_folder = Path("trained_models") / "area" / model_name
        model_path = model_folder / f"{model_name}.h5"
        if not model_path.exists():
            logg.error(f"Model not found at: {model_path}")
            logg.error(f"Train it with hypa_grid = {hypa}")
            raise FileNotFoundError
        model = models.load_model(model_path)

    else:
        model = None

    return model, model_name


def load_trained_model(
    model_type: str, datasets_type: str, train_words_type: str, do_load: bool = True
) -> ty.Tuple[models.Model, str]:
    """MAKEDOC: what is load_trained_model doing?"""
    logg = logging.getLogger(f"c.{__name__}.load_trained_model")
    # logg.setLevel("INFO")
    logg.debug(f"Start load_trained_model {model_type}")

    if model_type == "cnn":
        override_hypa = {"dataset": datasets_type, "words": train_words_type}
        model, model_name = load_trained_model_cnn(override_hypa, do_load)

    elif model_type == "attention":
        override_hypa = {"dataset_name": datasets_type, "words_type": train_words_type}
        model, model_name = load_trained_model_att(override_hypa, do_load)

    elif model_type == "area":
        override_hypa = {"dataset_name": datasets_type, "words_type": train_words_type}
        model, model_name = load_trained_model_area(override_hypa, do_load)

    return model, model_name


def plot_sentence_pred(
    sentence_sig: np.ndarray,
    y_pred: np.ndarray,
    norm_tra: str,
    train_words: ty.List[str],
    sentence_hop_length: int,
    split_length: int,
    fig_name: str,
) -> None:
    """MAKEDOC: what is plot_sentence_pred doing?"""
    logg = logging.getLogger(f"c.{__name__}.plot_sentence_pred")
    # logg.setLevel("INFO")
    logg.debug("Start plot_sentence_pred")

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))

    end_pad = split_length // sentence_hop_length
    pad_width = ((0, end_pad + 1), (0, 0))
    # pad_width = ((0, end_pad - 1), (0, 0))

    # end_pad = split_length // sentence_hop_length // 2
    # end_pad = 0
    # pad_width = ((0, end_pad), (0, 0))

    y_pred_padded = np.pad(y_pred, pad_width=pad_width, mode="edge")

    ax[0].plot(sentence_sig)
    ax[1].imshow(y_pred_padded.T, cmap=plt.cm.viridis, aspect="auto")
    # ax[2].imshow(y_pred.T, cmap=plt.cm.viridis, aspect="auto")

    ax[0].set_title(norm_tra, fontsize=20)
    # norm_tra_list = norm_tra.split()
    # len_sentence_words = len(norm_tra_list)
    # x_tickpos = np.arange(len_sentence_words) * len(sentence_sig) / len_sentence_words
    # ax[0].set_xticks(x_tickpos)
    # ax[0].set_xticklabels(norm_tra_list, rotation=40)
    ax[0].set_xlim(0, len(sentence_sig))

    y_tickpos = np.arange(len(train_words))
    ax[1].set_yticks(y_tickpos)
    ax[1].set_yticklabels(train_words)
    ax[1].set_title("Stream predictions", fontsize=20)
    # ax[2].set_yticks(y_tickpos)
    # ax[2].set_yticklabels(train_words)

    fig.tight_layout()

    plot_folder = Path("plot_stream") / "all5"
    if not plot_folder.exists():
        plot_folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_folder / fig_name.format("pdf"))
    fig.savefig(plot_folder / fig_name.format("png"))


def evaluate_stream(
    model: models.Model,
    datasets_type: str,
    train_words_type: str,
    architecture_type: str,
    model_name: str,
    orig_wav_path: Path,
    norm_tra: str,
    wav_ID: str,
) -> np.ndarray:
    """MAKEDOC: what is evaluate_stream doing?

    CNN_nf32_ks02_ps01_dw32_dr01_lr04_opa1_dsmeL04_bs32_en15_wLTnumLS
    if not 4 < len(norm_tra.split()) < 15:
    sentence_index 10
    sentence_wav_paths[6241_61943_000011_000003]: 6241/61943/6241_61943_000011_000003.wav
    sentence_norm_tra[6241_61943_000011_000003]: As usual, the crew was small, five Danes doing the whole of the work.

    sentence_index 16
    sentence_wav_paths[2412_153947_000023_000000]: 2412/153947/2412_153947_000023_000000.wav
    sentence_norm_tra[2412_153947_000023_000000]: june ninth eighteen seventy two

    sentence_index 19
    sentence_wav_paths[174_168635_000014_000000]: 174/168635/174_168635_000014_000000.wav
    sentence_norm_tra[174_168635_000014_000000]: CHAPTER three-TWO MISFORTUNES MAKE ONE PIECE OF GOOD FORTUNE

    sentence_index 22
    sentence_wav_paths[3000_15664_000017_000003]: 3000/15664/3000_15664_000017_000003.wav
    sentence_norm_tra[3000_15664_000017_000003]: The full grown bucks weigh nearly three hundred and fifty pounds.

    sentence_index 26
    sentence_wav_paths[2277_149897_000035_000002]: 2277/149897/2277_149897_000035_000002.wav
    sentence_norm_tra[2277_149897_000035_000002]: Three o'clock came, four, five, six, and no letter.

    sentence_index 33
    sentence_wav_paths[8297_275154_000019_000000]: 8297/275154/8297_275154_000019_000000.wav
    sentence_norm_tra[8297_275154_000019_000000]: "I will do neither the one nor the other.

    sentence_index 36
    sentence_wav_paths[8297_275156_000017_000007]: 8297/275156/8297_275156_000017_000007.wav
    sentence_norm_tra[8297_275156_000017_000007]: In any event, her second marriage would lead to one disastrous result.

    sentence_index 42
    sentence_wav_paths[7976_110124_000012_000000]: 7976/110124/7976_110124_000012_000000.wav
    sentence_norm_tra[7976_110124_000012_000000]: "Be sure that you admit no one," commanded the merchant.

    sentence_index 46
    sentence_wav_paths[7976_105575_000008_000004]: 7976/105575/7976_105575_000008_000004.wav
    sentence_norm_tra[7976_105575_000008_000004]: Five of my eight messmates of the day before were shot.

    sentence_index 66
    sentence_wav_paths[251_118436_000005_000000]: 251/118436/251_118436_000005_000000.wav
    sentence_norm_tra[251_118436_000005_000000]: one Death Strikes a King

    interesting because the two after the silence is identified, the four said quickly is missed
    sentence_index 67
    sentence_wav_paths[251_137823_000033_000000]: 251/137823/251_137823_000033_000000.wav
    sentence_norm_tra[251_137823_000033_000000]: Of the four other company engineers, two were now stirring and partly conscious.

    sentence_index 100
    sentence_wav_paths[1993_147966_000015_000000]: 1993/147966/1993_147966_000015_000000.wav
    sentence_norm_tra[1993_147966_000015_000000]: We had three weeks of this mild, open weather.

    ATT_ct02_dr01_ks01_lu01_qt04_dw01_opa1_lr04_bs02_en01_dsmeLa3_wLTnumLS
    sentence_index 10
    sentence_wav_paths[6241_61943_000011_000003]: /home/pmn/audiodatasets/LibriTTS/dev-clean/6241/61943/6241_61943_000011_000003.wav
    sentence_norm_tra[6241_61943_000011_000003]: As usual, the crew was small, five Danes doing the whole of the work.

    sentence_index 40
    sentence_wav_paths[7976_110124_000015_000000]: /home/pmn/audiodatasets/LibriTTS/dev-clean/7976/110124/7976_110124_000015_000000.wav
    sentence_norm_tra[7976_110124_000015_000000]: "Have pity upon a poor unfortunate one!" he called out.
    """
    logg = logging.getLogger(f"c.{__name__}.evaluate_stream")
    # logg.setLevel("INFO")
    logg.debug("Start evaluate_stream")

    # a random number generator to use
    rng = np.random.default_rng(12345)

    # the sample rate to use
    new_sr = 16000

    # load the sentence and resample it
    sentence_sig, sentence_sr = librosa.load(orig_wav_path, sr=None)
    sentence_sig = librosa.resample(sentence_sig, sentence_sr, new_sr)

    # split the sentence in chunks every sentence_hop_length
    sentence_hop_length = new_sr // 16

    # the words the model was trained on
    words = sorted(words_types[train_words_type])
    logg.debug(f"words: {words}")

    # the length of the split is chosen to match the training type
    if train_words_type.endswith("LS"):
        split_length = new_sr // 2
    else:
        split_length = new_sr

    splits = split_sentence(
        sentence_sig, datasets_type, sentence_hop_length, split_length
    )
    logg.debug(f"len(splits): {len(splits)}")

    # if there are no splits available we pretend to have predicted only background
    # or only random if there are only numbers
    if len(splits) == 0:
        num_words = len(words)
        if num_words == 10:
            fake_pred = np.ones((1, num_words), dtype=np.float32)
            fake_pred /= num_words
        else:
            fake_pred = np.zeros((1, num_words), dtype=np.float32)
            fake_pred[0][0] = 1
        return fake_pred

    # compute spectrograms / augment / compose
    if datasets_type.startswith("au"):
        specs = augment_signals(splits, datasets_type, rng, which_fold="testing")
        logg.debug(f"specs.shape: {specs.shape}")
        specs_img = np.expand_dims(specs, axis=-1)
        logg.debug(f"specs_img.shape: {specs_img.shape}")

    elif datasets_type.startswith("me"):

        spec_dict = get_spec_dict()
        mel_kwargs = spec_dict[datasets_type]
        logg.debug(f"mel_kwargs: {mel_kwargs}")

        spec_shape_dict = get_spec_shape_dict()
        spec_shape = spec_shape_dict[datasets_type]
        requested_length = spec_shape[1]
        logg.debug(f"requested_length: {requested_length}")

        p2d_kwargs = {"ref": np.max}
        specs_img = compute_spectrograms(
            splits, mel_kwargs, p2d_kwargs, requested_length
        )
        logg.debug(f"specs_img.shape: {specs_img.shape}")

    y_pred = model.predict(specs_img)
    # logg.debug(f"y_pred: {y_pred}")
    y_index = np.argmax(y_pred, axis=1)
    # logg.debug(f"y_index: {y_index}")
    y_pred_labels = [words[i] for i in y_index]
    # logg.debug(f"y_pred_labels: {y_pred_labels}")

    save_plots = False
    # save_plots = True
    if save_plots:
        clean_labels = []
        for yl in y_pred_labels:
            if yl.startswith("_other"):
                clean_labels.append(".")
            else:
                clean_labels.append(yl)
        logg.debug(f"Predictions {clean_labels}")

        # fig_name = f"{architecture_type}_{evaluation_type}_{datasets_type}_{train_words_type}_{norm_tra}.{{}}"
        # fig_name = f"{architecture_type}"
        fig_name = f"{model_name}"
        # fig_name += f"_{evaluation_type}"
        fig_name += f"_{datasets_type}"
        fig_name += f"_{train_words_type}"
        fig_name += f"_{wav_ID}"
        # fig_name += f"_{norm_tra}.{{}}"
        fig_name += ".{}"

        plot_sentence_pred(
            sentence_sig,
            y_pred,
            norm_tra,
            words,
            sentence_hop_length,
            split_length,
            fig_name,
        )
        # plt.show()

    return y_pred


def do_stream_evaluation(
    architecture_type,
    which_dataset,
    train_words_type,
    sentence_wav_paths,
    sentence_norm_tra,
    good_sentences,
) -> None:
    r"""MAKEDOC: what is do_stream_evaluation doing?"""
    logg = logging.getLogger(f"c.{__name__}.do_stream_evaluation")
    # logg.setLevel("INFO")
    logg.debug("Start do_stream_evaluation")

    wav_IDs = list(sentence_wav_paths.keys())
    logg.debug(f"len(wav_IDs): {len(wav_IDs)}")

    good_count = 0
    bad_count = 0

    # magic to fix the GPUs
    setup_gpus()

    # load the model
    model, model_name = load_trained_model(
        architecture_type, which_dataset, train_words_type
    )

    # all_y_pred: ty.Dict[str, ty.List[float]] = {}

    ypred_folder = Path("plot_stream") / "y_pred" / model_name
    if not ypred_folder.exists():
        ypred_folder.mkdir(parents=True, exist_ok=True)

    for sentence_index in good_sentences:

        # get info for one sentence
        wav_ID = wav_IDs[sentence_index]
        logg.debug(f"\nsentence_index {sentence_index} / {len(good_sentences)-1}")
        orig_wav_path = sentence_wav_paths[wav_ID]
        logg.debug(f"sentence_wav_paths[{wav_ID}]: {orig_wav_path}")
        norm_tra = sentence_norm_tra[wav_ID]
        logg.debug(f"sentence_norm_tra[{wav_ID}]: {norm_tra}")

        # build the output path
        pred_name = f"{model_name}"
        pred_name += f"_{wav_ID}"
        pred_name += ".npy"
        logg.debug(f"pred_name SINGLE: {pred_name}")
        pred_path = ypred_folder / pred_name

        if pred_path.exists():
            logg.info(f"Already predicted {pred_path}")
            continue

        y_pred = evaluate_stream(
            model,
            which_dataset,
            train_words_type,
            architecture_type,
            model_name,
            orig_wav_path,
            norm_tra,
            wav_ID,
        )
        logg.debug(f"y_pred.shape: {y_pred.shape}")

        np.save(pred_path, y_pred)

        # all_y_pred[wav_ID] = y_pred.tolist()

        # wasgood = input()
        # if wasgood == "y":
        #     good_count += 1
        # else:
        #     bad_count += 1

    logg.debug(f"good_count: {good_count}")
    logg.debug(f"bad_count: {bad_count}")
    logg.debug(f"total: {bad_count+good_count}")

    # logg.debug(f"all_y_pred: {all_y_pred}")

    # pred_name = f"{model_name}"
    # pred_name += f"_{sum(good_sentences)}_{good_sentences[0]}_{good_sentences[-1]}"
    # pred_name += ".json"
    # logg.debug(f"pred_name: {pred_name}")

    # ypred_folder = Path("plot_stream")
    # pred_path = ypred_folder / pred_name
    # pred_path.write_text(json.dumps(all_y_pred, indent=4))

    # plt.show()


def clean_words(words) -> None:
    r"""MAKEDOC: what is clean_words doing?"""
    logg = logging.getLogger(f"c.{__name__}.clean_words")
    # logg.setLevel("INFO")
    logg.debug("Start clean_words")

    for i, word in enumerate(words):

        # remove LS tags
        if word.startswith("loudest_"):
            clean_word = word[8:]
            words[i] = clean_word
        elif word.endswith("_loud"):
            clean_word = word[:-5]
            words[i] = clean_word

        # translate weird labels to readable names
        if words[i] == "_background":
            words[i] = "_silence"
        if words[i] == "_other_ltts":
            words[i] = "_conversation"


def compute_sentence_far(
    train_words, y_pred, norm_tra, th_list
) -> ty.Dict[int, ty.Dict[str, float]]:
    r"""MAKEDOC: what is compute_sentence_far doing?

    train_words: list of words the model was trained on
    y_pred:      (num_split, num_words) numpy array of predictions
    norm_tra:    the normalized transcript of the sentence

    false alarm
    false reject

    4 cases:
        - a train word is in the sentence
            - is predicted: CORRECT found
            - is missed:    false reject
        - no train word is in the sentence
            - is predicted: false alarm
            - is missed:    CORRECT miss
    """
    logg = logging.getLogger(f"c.{__name__}.compute_sentence_far")
    # logg.setLevel("INFO")
    logg.debug("Start compute_sentence_far")

    # # a regex to find the trained words
    # train_words_bound = [fr"\b{w}\b" for w in train_words]
    # train_words_re = re.compile("|".join(train_words_bound))

    # # find out if a train word is in the sentence
    # match = train_words_re.search(norm_tra)
    # if match is None:
    #     is_good_sentence = False
    # else:
    #     is_good_sentence = True
    # logg.debug(f"is_good_sentence: {is_good_sentence}")

    found_words = set()
    is_good_sentence = False
    for w in train_words:
        if w[0] == "_":
            continue
        match = re.search(fr"\b{w}\b", norm_tra)
        if match is None:
            pass
        else:
            found_words.add(w)
            is_good_sentence = True
    logg.debug(f"is_good_sentence: {is_good_sentence}")
    logg.debug(f"found_words: {found_words}")

    num_words = train_words[-10:]

    results: ty.Dict[int, ty.Dict[str, float]] = {}

    # set pred threshold ranges
    for ti, th in enumerate(th_list):
        # logg.debug(f"\nth: {th}")

        correct_miss = 0
        correct_found = 0
        false_alarm = 0
        false_reject = 0

        # get the predictions relative to the numbers
        num_pred = y_pred[:, -10:]
        # logg.debug(f"num_pred.shape: {num_pred.shape}")

        # for every word get all predictions
        for wi, word_pred in enumerate(num_pred.T):
            # logg.debug(f"word_pred.shape: {word_pred.shape}")

            # if it is active
            if np.any(word_pred > th):

                # if it is actually there set it correct
                if num_words[wi] in found_words:
                    # logg.debug(f"Found correct: {num_words[wi]} at {wi}")
                    correct_found += 1

                # else is a false alarm
                else:
                    # logg.debug(f"False alarm: {num_words[wi]} at {wi}")
                    false_alarm += 1

            # if it is not active
            else:
                # but it should have been
                if num_words[wi] in found_words:
                    # logg.debug(f"False reject: {num_words[wi]} at {wi}")
                    false_reject += 1
                else:
                    # logg.debug(f"Correct miss: {num_words[wi]} at {wi}")
                    correct_miss += 1

        results[ti] = {
            "correct_miss": correct_miss,
            "correct_found": correct_found,
            "false_alarm": false_alarm,
            "false_reject": false_reject,
        }

    return results


def compute_all_false_alarm_reject(
    architecture_type,
    which_dataset,
    train_words_type,
    sentence_wav_paths,
    sentence_norm_tra,
    good_sentences,
) -> None:
    r"""MAKEDOC: what is compute_all_false_alarm_reject doing?"""
    logg = logging.getLogger(f"c.{__name__}.compute_all_false_alarm_reject")
    # logg.setLevel("INFO")
    logg.debug("\n\nStart compute_all_false_alarm_reject")

    # extract the wav_IDs for clarity
    wav_IDs = list(sentence_wav_paths.keys())
    logg.debug(f"len(wav_IDs): {len(wav_IDs)}")

    # get the model name lol so inefficient
    _, model_name = load_trained_model(
        architecture_type, which_dataset, train_words_type, do_load=False
    )

    # where the predictions should be
    ypred_folder = Path("plot_stream") / "y_pred" / model_name
    roc_folder = Path("plot_stream") / "roc_results" / model_name
    if not roc_folder.exists():
        roc_folder.mkdir(parents=True, exist_ok=True)

    # get the words the model was trained on
    words = sorted(words_types[train_words_type])
    logg.debug(f"words: {words}")
    clean_words(words)
    logg.debug(f"words: {words}")

    # set the thresholds
    num_th = 20
    th_list = np.linspace(0, 1, num_th, dtype=np.float32)

    # analyze all sentences
    for sentence_index in good_sentences:
        logg.debug(f"\nsentence_index {sentence_index} / {len(good_sentences)-1}")

        # get info for one sentence
        wav_ID = wav_IDs[sentence_index]

        # get the result path
        roc_name = f"{model_name}"
        roc_name += f"__{wav_ID}"
        roc_name += f"__{num_th}"
        roc_name += ".json"
        roc_path = roc_folder / roc_name
        if roc_path.exists():
            logg.info(f"Already computed {roc_path}")
            continue

        # get the prediction path
        pred_name = f"{model_name}"
        pred_name += f"_{wav_ID}"
        pred_name += ".npy"
        pred_path = ypred_folder / pred_name
        logg.debug(f"\npred_name: {pred_name}")

        # check if there is the prediction
        if not pred_path.exists():
            logg.warn(f"\nMissing {pred_path}")
            continue

        # load it
        y_pred = np.load(pred_path)
        logg.debug(f"y_pred.shape: {y_pred.shape}")

        # get the normalized transcript of the sentence
        norm_tra = sentence_norm_tra[wav_ID]
        logg.debug(f"sentence_norm_tra[{wav_ID}]: {norm_tra}")

        roc_results = compute_sentence_far(words, y_pred, norm_tra, th_list)

        recap: ty.Dict[str, ty.Any] = {}
        recap["norm_tra"] = norm_tra
        recap["th_list"] = th_list.tolist()
        recap["roc_results"] = roc_results

        # logg.debug(f"{json.dumps(recap, indent=4)}")

        roc_path.write_text(json.dumps(recap, indent=4))


def analyze_false_alarm_rejects(
    architecture_type,
    which_dataset,
    train_words_type,
    sentence_wav_paths,
    sentence_norm_tra,
    good_sentences,
) -> None:
    r"""MAKEDOC: what is analyze_false_alarm_rejects doing?"""
    logg = logging.getLogger(f"c.{__name__}.analyze_false_alarm_rejects")
    # logg.setLevel("INFO")
    logg.debug("\n\nStart analyze_false_alarm_rejects")

    wav_IDs = list(sentence_wav_paths.keys())
    logg.debug(f"len(wav_IDs): {len(wav_IDs)}")

    # get the model name lol so inefficient
    _, model_name = load_trained_model(
        architecture_type, which_dataset, train_words_type, do_load=False
    )

    # where the FAR should be
    roc_folder = Path("plot_stream") / "roc_results" / model_name
    roc_analysis = Path("plot_stream") / "roc_analysis" / model_name
    if not roc_analysis.exists():
        roc_analysis.mkdir(parents=True, exist_ok=True)

    # get the words the model was trained on
    words = sorted(words_types[train_words_type])
    logg.debug(f"words: {words}")
    clean_words(words)
    logg.debug(f"words: {words}")

    # set the thresholds used when computing the data
    num_th = 20

    # accumulate data for each th value
    all_correct_miss = {ti: 0 for ti in range(num_th)}
    all_correct_found = {ti: 0 for ti in range(num_th)}
    all_false_alarm = {ti: 0 for ti in range(num_th)}
    all_false_reject = {ti: 0 for ti in range(num_th)}

    # analyze all sentences
    for sentence_index in good_sentences:
        # logg.debug(f"\nsentence_index {sentence_index} / {len(good_sentences)-1}")

        # get info for one sentence
        wav_ID = wav_IDs[sentence_index]

        # get the result path
        roc_name = f"{model_name}"
        roc_name += f"__{wav_ID}"
        roc_name += f"__{num_th}"
        roc_name += ".json"
        roc_path = roc_folder / roc_name

        # check if there is the roc data
        if not roc_path.exists():
            logg.warn(f"\nMissing {roc_path}")
            continue

        # load it
        roc_recap = json.loads(roc_path.read_text())
        # logg.debug(f"norm_tra: {roc_recap['norm_tra']}")
        th_list = roc_recap["th_list"]
        th_res = roc_recap["roc_results"]

        for ti, th in enumerate(th_list):

            # get the results for this th
            th_res_ti = th_res[str(ti)]
            correct_miss = th_res_ti["correct_miss"]
            correct_found = th_res_ti["correct_found"]
            false_alarm = th_res_ti["false_alarm"]
            false_reject = th_res_ti["false_reject"]

            # if there are correct words we take correct
            # even if there are alarms or rejects
            if correct_found > 0:
                # all_correct_found[ti] += correct_found
                all_correct_found[ti] += 1

            # if there are alarms those are worse than rejects
            elif false_alarm > 0:
                # all_false_alarm[ti] += false_alarm
                all_false_alarm[ti] += 1

            # if we arrived here and there are rejects,
            # the sample is a false reject
            elif false_reject > 0:
                # all_false_reject[ti] += false_reject
                all_false_reject[ti] += 1

            # if we reached here, the sample does not contain useful words
            # and it was not falsely selected
            elif correct_miss > 0:
                all_correct_miss[ti] += 1

            else:
                logg.warn(f"Weird th_res: {th_res}")
                pass

    # logg.debug(f"all_correct_miss: {all_correct_miss}")
    # logg.debug(f"all_correct_found: {all_correct_found}")
    # logg.debug(f"all_false_alarm: {all_false_alarm}")
    # logg.debug(f"all_false_reject: {all_false_reject}")

    header = " " * 10
    header += "    CM"
    header += "    CF"
    header += "    FR"
    header += "    FA"
    logg.debug(f"{header}")

    for ti, th in enumerate(th_list):
        recap = f"th: {th:.4f}"
        recap += f" {all_correct_miss[ti]: 5d}"
        recap += f" {all_correct_found[ti]: 5d}"
        recap += f" {all_false_reject[ti]: 5d}"
        recap += f" {all_false_alarm[ti]: 5d}"
        logg.debug(recap)


def run_evaluate_stream(args: argparse.Namespace) -> None:
    """MAKEDOC: What is evaluate_stream doing?"""
    logg = logging.getLogger(f"c.{__name__}.run_evaluate_stream")
    logg.debug("Starting run_evaluate_stream")

    # evaluation_type = args.evaluation_type
    train_words_type = args.train_words_type
    which_dataset = args.dataset_name
    architecture_type = args.architecture_type
    # sentence_index = args.sentence_index

    # get the sentence wav paths and transcripts
    do_filter_match = False
    # do_filter_match = True
    do_filter_len = False
    # do_filter_len = True
    sentence_wav_paths, sentence_norm_tra = build_ltts_sentence_list(
        train_words_type, do_filter_match=do_filter_match, do_filter_len=do_filter_len
    )

    wav_IDs = list(sentence_wav_paths.keys())
    logg.debug(f"len(wav_IDs): {len(wav_IDs)}")

    # good_sentences = [10, 16, 19, 22, 26, 33, 36, 42, 46, 66, 67, 100]
    # good_sentences = [19, 26, 40, 46, 67]
    # good_sentences = [19, 26, 67]
    # good_sentences = list(range(116))
    # good_sentences = list(range(33, 38))
    # good_sentences = list(range(3))
    good_sentences = list(range(len(wav_IDs)))

    do_the_stream = True
    # do_the_stream = False
    if do_the_stream:
        do_stream_evaluation(
            architecture_type,
            which_dataset,
            train_words_type,
            sentence_wav_paths,
            sentence_norm_tra,
            good_sentences,
        )

    do_the_compute_far = True
    # do_the_compute_far = False
    if do_the_compute_far:
        compute_all_false_alarm_reject(
            architecture_type,
            which_dataset,
            train_words_type,
            sentence_wav_paths,
            sentence_norm_tra,
            good_sentences,
        )

    do_the_analyze_far = True
    # do_the_analyze_far = False
    if do_the_analyze_far:
        analyze_false_alarm_rejects(
            architecture_type,
            which_dataset,
            train_words_type,
            sentence_wav_paths,
            sentence_norm_tra,
            good_sentences,
        )


if __name__ == "__main__":
    args = setup_env()
    run_evaluate_stream(args)
