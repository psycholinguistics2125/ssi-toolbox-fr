import pandas as pd
import numpy as np
import logging
import pickle
import spacy

from ssi_toolbox.data_loader import (
    load_interviewer_data,
    load_respondant_data,
    load_interviews_data,
)
from ssi_toolbox.utils import split_into_sentences, clean_text

from concurrent.futures import ThreadPoolExecutor


class Utterance:
    def __init__(
        self,
        text,
        start_time,
        end_time,
        speaker,
        word_list=None,
        nlp=None,
        tokenized=False,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.text = text
        self.s_time = start_time
        self.e_time = end_time
        self.speaker = speaker
        self.word_list = word_list
        self.nlp = nlp

        self.trf_sentence_vect = None
        self.w2v_sentence_vect = None
        self.fasttext_sentence_vect = None

    def compute_sentence_vectors(self, trf_model, w2v_model, fasttext_model):
        self.trf_sentence_vect = [
            trf_model.encode(sentence) for sentence in self.sentences
        ]
        self.w2v_sentence_vect = [
            average_sentence_encoding(w2v_model, token_list)
            for token_list in self.token_sentence_vect
        ]
        self.fasttext_sentence_vect = [
            average_sentence_encoding(fasttext_model, token_list)
            for token_list in self.token_sentence_vect
        ]
        return self

    def process_text(self):
        if self.nlp:
            self.doc = self.nlp(self.text)
            self.sentences = [
                sent.text for sent in self.doc.sents
            ]  # split_into_sentences(self.text)
            self.token_sentence_vect = [
                [token.text for token in sent] for sent in self.doc.sents
            ]  # [self._tokenize_text(nlp, text= sentence) for sentence in self.sentences]
            self.lemma_sentence_vect = [
                [token.lemma_ for token in sent] for sent in self.doc.sents
            ]  # [self._lemmatize_text(nlp, text = sentence ) for sentence in self.sentences]

        return self

    def _tokenize_text(self, text=None):
        if text is None:
            text = self.text
        return [token.text for token in self.doc]

    def _lemmatize_text(self, nlp, text=None):
        if text is None:
            text = self.text
        return [token.lemma_ for token in self.doc]

    def _to_dict(self):
        utt_dict = {
            "text": self.text,
            "start_time": self.s_time,
            "end_time": self.e_time,
            "speaker": self.speaker,
            "word_list": self.word_list,
            "sentences": self.sentences,
            "token_sentence_vect": self.token_sentence_vect,
            "lemma_sentence_vect": self.lemma_sentence_vect,
            "trf_sentence_vect": self.trf_sentence_vect,
            "w2v_sentence_vect": self.w2v_sentence_vect,
            "fasttext_sentence_vect": self.fasttext_sentence_vect,
        }

        return utt_dict

    def _from_dict(self, utt_dict):
        """
        Load the Utterance object from a dictionary.

        Raises:
            e: _description_

        Returns:
            _type_: _description_
        """
        try:
            self.text = utt_dict["text"]
            self.s_time = utt_dict["start_time"]
            self.e_time = utt_dict["end_time"]
            self.speaker = utt_dict["speaker"]
            self.word_list = utt_dict["word_list"]
            self.sentences = utt_dict["sentences"]
            self.token_sentence_vect = utt_dict["token_sentence_vect"]
            self.lemma_sentence_vect = utt_dict["lemma_sentence_vect"]
            self.trf_sentence_vect = utt_dict["trf_sentence_vect"]
            self.w2v_sentence_vect = utt_dict["w2v_sentence_vect"]
            self.fasttext_sentence_vect = utt_dict["fasttext_sentence_vect"]
        except Exception as e:
            self.logger.error(f"Error loading Utterance object from dict: {e}")
            raise e

        return self


class Corpus:
    def __init__(self, config_path=None, corpus_path=None, lang="fr"):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.lang = lang
        if self.lang == "fr":
            self.nlp = spacy.load(
                "fr_core_news_sm", disable=["ner", "parser", "tagger"])
            self.nlp.add_pipe("sentencizer")
        else:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "tagger"])
            self.nlp.add_pipe("sentencizer")

        if corpus_path:
            loaded_corpus = self.load(corpus_path)
            self.__dict__ = loaded_corpus.__dict__
            self.logger.info(f"Corpus object '{self.name}' loaded from {corpus_path}")
        else:
            try:
                self.df_interviewer_metadata = load_interviewer_data(
                    config_path=config_path
                )
                self.logger.info(f"Loaded interviewer metadata ! ")

                self.df_respondant_metadata = load_respondant_data(
                    config_path=config_path
                )
                self.logger.info(f"Loaded respondant metadata ! ")

                loaded_interview_df = load_interviews_data(config_path=config_path)
                self.df_interview_list = self.convert_interview_list_to_df(
                    loaded_interview_df
                )
                self.logger.info(f"Loaded interview list ! ")

                self.corpus = self.df_interviewer_metadata.merge(
                    self.df_respondant_metadata, on="code_enq"
                ).merge(self.df_interview_list, on="code")
            except Exception as e:
                self.logger.error(f"Error creating Corpus object: {e}")

    def convert_interview_list_to_df(self, loaded_interview_df, target_col="part_1"):
        df = pd.DataFrame()
        df["code"] = loaded_interview_df["code"]

        df["interviews"] = loaded_interview_df[target_col].apply(
            lambda x: [
                Utterance(
                    text=clean_text(turn["text"]),
                    start_time=turn["stime"],
                    end_time=turn["etime"],
                    speaker=turn["speaker"],
                    word_list=turn["w_list"],
                    nlp=self.nlp,
                )
                for turn in x
                if len(turn["text"]) > 1
            ]
        )
        with ThreadPoolExecutor() as executor:
            df["interviews"] = df["interviews"].apply(
                lambda utterances: list(
                    executor.map(lambda u: process_utterance(u), utterances)
                )
            )

        return df

    def compute_sentence_vectors(self, trf_model, w2v_model, fasttext_model):
        self.corpus["interviews"] = self.corpus["interviews"].apply(
            lambda x: [
                utterance.compute_sentence_vectors(trf_model, w2v_model, fasttext_model)
                for utterance in x
            ]
        )

    def save_corpus(self, file_path):
        """
        Save the Corpus object to a file using pickle.

        Parameters:
        - file_path (str): The path to the file where the Corpus object will be saved.
        """
        try:
            # convert utterances to dict
            self.corpus["interviews"] = self.corpus["interviews"].apply(
                lambda x: [utterance._to_dict() for utterance in x]
            )
            self.corpus.to_pickle(file_path)
        except Exception as e:
            self.logger.error(f"Error saving Corpus object: {e}")

    def load_corpus_from_pickle(self, file_path):
        # Todo
        try:
            loaded_corpus = pd.read_pickle(file_path)
            # convert dict to utterances
            loaded_corpus["interviews"] = loaded_corpus["interviews"].apply(
                lambda x: [
                    Utterance(
                        text=None, start_time=None, end_time=None, speaker=None
                    )._from_dict(utterance)
                    for utterance in x
                ]
            )
        except Exception as e:
            self.logger.error(f"Error loading Corpus object: {e}")
            raise e
        return loaded_corpus


def average_sentence_encoding(model, token_list):
    """
    Compute the average word embedding for a sentence.

    Parameters:
    - model (gensim.models.Word2Vec): The Word2Vec model to use for computing the word embeddings.
    - sentence (str): The sentence to encode.

    Returns:
    - np.ndarray: The average word embedding for the sentence.
    """
    # Split the sentence into words

    # Filter out words that are not in the Word2Vec model's vocabulary
    words_in_vocab = [
        word.lower() for word in token_list if word.lower() in model.key_to_index
    ]

    # compute the embedings
    word_embeddings = [model[word] for word in words_in_vocab]

    # Calculate the average vector
    avg_vector = np.mean(word_embeddings, axis=0)

    return avg_vector


def process_utterance(utterance):
    return utterance.process_text()
