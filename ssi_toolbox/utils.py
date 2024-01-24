import yaml
import os
import pandas as pd
import re

from gensim.models import Word2Vec, load_facebook_model
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def read_csv_auto(file_path, min_columns=2, **kwargs):
    """
    Read a CSV file with automatic separator detection using pandas.

    Parameters:
    - file_path (str): The path to the CSV file.
    - min_columns (int): Minimum number of columns expected. If the loaded DataFrame has fewer columns, an exception is raised.
    - **kwargs: Additional keyword arguments to pass to pandas.read_csv.

    Returns:
    - pd.DataFrame: The loaded DataFrame.
    """
    # List of common separators to check
    separators = [",", ";", "\t", "|"]

    for sep in separators:
        try:
            # Try reading the CSV with the current separator
            df = pd.read_csv(file_path, sep=sep, **kwargs)

            # Check if the loaded DataFrame has a reasonable number of columns
            if df.shape[1] >= min_columns:
                return df
        except pd.errors.ParserError:
            # If reading fails, try the next separator
            pass

    # If none of the separators work or the number of columns is too small, raise an error
    raise ValueError(
        "Unable to automatically detect the separator or the loaded DataFrame has too few columns. Please specify the 'sep' parameter."
    )


def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    alphabets = "([A-Za-z])"

    text = " " + text + "  "
    text = text.replace("\n", " ")

    # English and French prefixes
    prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"

    # Common starters for English
    english_starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt)"

    # Common starters for French
    french_starters = "(M|Mme|Mlle|Dr|Prof|Capt|Cpt|Lt)"

    starters = french_starters + "|" + english_starters

    # Acronyms for both languages
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"

    # Websites for both languages
    websites = "[.](com|net|org|io|gov|edu|me)"

    # Digits for both languages
    digits = "([0-9])"

    # Multiple dots
    multiple_dots = r"\.{2,}"

    text = " " + text + "  "
    text = text.replace("\n", " ")
    text = re.sub(prefixes, "\\1<prd>", text)
    text = re.sub(websites, "<prd>\\1", text)
    text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
    text = re.sub(
        multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text
    )
    if "Ph.D" in text:
        text = text.replace("Ph.D.", "Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
    text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
    text = re.sub(
        alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]",
        "\\1<prd>\\2<prd>\\3<prd>",
        text,
    )
    text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
    text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
    text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
    text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
    if "”" in text:
        text = text.replace(".”", "”.")
    if '"' in text:
        text = text.replace('."', '".')
    if "!" in text:
        text = text.replace('!"', '"!')
    if "?" in text:
        text = text.replace('?"', '"?')
    text = text.replace(".", ".<stop>")
    text = text.replace("?", "?<stop>")
    text = text.replace("!", "!<stop>")
    text = text.replace("<prd>", ".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]:
        sentences = sentences[:-1]
    return sentences

    return sentences


def load_model(config, model_name, lang="fr"):
    model_folder = config["model_folder"]

    if model_name == "w2v":
        if lang == "fr":
            return Word2Vec.load(os.path.join(model_folder, "w2v-fr.model"))
        elif lang == "en":
            return KeyedVectors.load(os.path.join(model_folder, "wv-kv-300.kv"))
        else:
            raise ValueError("lang not supported")

    elif model_name == "fasttext":
        if lang == "fr":
            return load_facebook_model.load(
                os.path.join(model_folder, "fr_fast_text.kv")
            )
        elif lang == "en":
            return KeyedVectors.load(os.path.join(model_folder, "fast_text.kv"))
        else:
            raise ValueError("lang not supported")
    elif model_name == "sentence_sim":
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    else:
        raise ValueError("model not supported")


def clean_text(text):
    text = text.replace("\n", " ")  # remove new lines
    text = text.replace("\t", " ")  # remove tabs
    text = text.replace("  ", " ")  # remove double spaces
    return text


def flatten_list(l):
    return [item for sublist in l for item in sublist]
