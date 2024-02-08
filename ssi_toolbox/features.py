import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from ssi_toolbox.utils import flatten_list

import textdistance


def utterance_count(interview):
    return len(interview)


def speaker_turn_count(interview, speaker_id):
    return sum(utterance.speaker == speaker_id for utterance in interview)


def utterance_duration(utterance):
    return float(utterance.e_time) - float(utterance.s_time)


def average_utterance_duration(interview):
    return sum(utterance_duration(utterance) for utterance in interview) / len(
        interview
    )


def average_utterance_duration_by_speaker(interview, speaker_id):
    return sum(
        utterance_duration(utterance)
        for utterance in interview
        if utterance.speaker == speaker_id
    ) / speaker_turn_count(interview, speaker_id)


def utterance_length(utterance, unit="lemma"):
    if unit == "lemma":
        return len(flatten_list(utterance.lemma_sentence_vect))
    elif unit == "token":
        return len(flatten_list(utterance.token_sentence_vect))
    elif unit == "sentences":
        return len(utterance.sentences)
    elif unit == "char":
        return len(utterance.text)


def average_utterance_length_by_speaker(
    interview, unit="lemma", speaker_id="enqueteur"
):
    return np.mean(
        [
            utterance_length(utterance, unit=unit)
            for utterance in interview
            if utterance.speaker == speaker_id
        ]
    )

def extract_enq_text(interview) : 
    enq_text = []
    for utt in interview : 
        if utt['speaker'] == "enqueteur" : 
            enq_text.append(utt['text'])
    return enq_text

def utterance_length_by_speaker(interview, unit="lemma", speaker_id="enqueteur"):
    return [
        utterance_length(utterance, unit=unit)
        for utterance in interview
        if utterance.speaker == speaker_id
    ]


def compute_previous_sentences_similarity(interview, unit="lemma"):
    previous_sent_sim = []
    for i in range(1, len(interview)):
        try:
            if unit == "lemma":
                sent1 = interview[i - 1].lemma_sentence_vect[
                    -1
                ]  # last sentence of previous utterance
                sent2 = interview[i].lemma_sentence_vect[
                    0
                ]  # first sentence of current utterance
                sim = textdistance.jaccard.similarity(sent1, sent2)
            elif unit == "token":
                sent1 = interview[i - 1].token_sentence_vect[-1]
                sent2 = interview[i].token_sentence_vect[0]
                sim = textdistance.jaccard.similarity(sent1, sent2)
            elif unit == "trf_vect":
                sent1 = interview[i - 1].trf_sentence_vect[-1]
                sent2 = interview[i].trf_sentence_vect[0]
                sim = cosine_similarity([sent1], [sent2])[0][0]
            elif unit == "w2v_vect":
                sent1 = interview[i - 1].w2v_sentence_vect[-1]
                sent2 = interview[i].w2v_sentence_vect[0]
                sim = cosine_similarity([sent1], [sent2])[0][0]
            elif unit == "fasttext_vect":
                sent1 = interview[i - 1].fasttext_sentence_vect[-1]
                sent2 = interview[i].fasttext_sentence_vect[0]
                sim = cosine_similarity([sent1], [sent2])[0][0]
        except:
            sim = 0

        previous_sent_sim.append(sim)
    return previous_sent_sim


def compute_next_sentences_similarity(interview, unit="lemma"):
    next_sent_sim = []
    for i in range(len(interview) - 1):
        try:
            if unit == "lemma":
                sent1 = interview[i].lemma_sentence_vect[
                    -1
                ]  # last sentence of current utterance
                sent2 = interview[i + 1].lemma_sentence_vect[
                    0
                ]  # first sentence of next utterance
                sim = textdistance.jaccard.similarity(sent1, sent2)
            elif unit == "token":
                sent1 = interview[i].token_sentence_vect[-1]
                sent2 = interview[i + 1].token_sentence_vect[0]
                sim = textdistance.jaccard.similarity(sent1, sent2)
            elif unit == "trf_vect":
                sent1 = interview[i].trf_sentence_vect[-1]
                sent2 = interview[i + 1].trf_sentence_vect[0]
                sim = cosine_similarity([sent1], [sent2])[0][0]
            elif unit == "w2v_vect":
                sent1 = interview[i].w2v_sentence_vect[-1]
                sent2 = interview[i + 1].w2v_sentence_vect[0]
                sim = cosine_similarity([sent1], [sent2])[0][0]
            elif unit == "fasttext_vect":
                sent1 = interview[i].fasttext_sentence_vect[-1]
                sent2 = interview[i + 1].fasttext_sentence_vect[0]
                sim = cosine_similarity([sent1], [sent2])[0][0]
        except:
            sim = 0

        next_sent_sim.append(sim)
    return next_sent_sim


def conversation_flow(interview, unit="lemma"):
    sims = []
    for i in range(len(interview) - 1):
        if unit == "lemma":
            sent1 = flatten_list(
                interview[i].lemma_sentence_vect
            )  # all lemma of current utterances
            sent2 = flatten_list(
                interview[i + 1].lemma_sentence_vect
            )  # all lemma of previous utterances
            sim = textdistance.jaccard.similarity(sent1, sent2)
        elif unit == "token":
            sent1 = flatten_list(interview[i].token_sentence_vect)
            sent2 = flatten_list(interview[i + 1].token_sentence_vect)
            sim = textdistance.jaccard.similarity(sent1, sent2)
        elif unit == "trf_vect":
            sent1 = np.mean(interview[i].trf_sentence_vect, axis=0)
            sent2 = np.mean(interview[i + 1].trf_sentence_vect, axis=0)
            sim = cosine_similarity([sent1], [sent2])[0][0]
        elif unit == "w2v_vect":
            sent1 = np.mean(interview[i].w2v_sentence_vect, axis=0)
            sent2 = np.mean(interview[i + 1].w2v_sentence_vect, axis=0)
            sim = cosine_similarity([sent1], [sent2])[0][0]
        elif unit == "fasttext_vect":
            sent1 = np.mean(interview[i].fasttext_sentence_vect, axis=0)
            sent2 = np.mean(interview[i + 1].fasttext_sentence_vect, axis=0)
            sim = cosine_similarity([sent1], [sent2])[0][0]
        sims.append(sim)
    return sims


def inter_speaker_turn_duration(interview):
    inter_speaker_durations = [
        float(interview[i + 1].s_time) - float(interview[i].e_time)
        for i in range(len(interview) - 1)
        if interview[i].speaker != interview[i + 1].speaker
    ]
    return inter_speaker_durations


def predict_dialog_act(interview, model, labels_dict):
    for utt in interview:
        if utt['speaker'] == "enqueteur":
            pred = model([utt['text']])[0]
            utt['dialog_act'] = [[f for f, p in zip(labels_dict, ps) if p] for ps in [pred]]
    return interview


def predict_dialog_themes(interview, model, labels_dict):
    for utt in interview:
        if utt['speaker'] == "enqueteur":
            probas = model.predict_proba([utt['text']])
            argmax_indices  = np.argmax(probas,axis=1)
            preds = np.zeros_like(probas)
            row = np.arange(probas.shape[0])
            preds[row, argmax_indices] = 1
            pred = preds[0]

            utt['dialog_theme'] = [[f for f, p in zip(labels_dict, ps) if p] for ps in [pred]]
            print(utt['dialog_theme'])
    return interview


def get_dialog_act_list(interview):
    dialog_act_list = []
    for utt in interview:
        if utt['speaker'] == "enqueteur":
            dialog_act_list.append(flatten_list(utt['dialog_act']))
    return flatten_list(dialog_act_list)

def get_dialog_theme_list(interview):
    dialog_act_list = []
    for utt in interview:
        if utt['speaker'] == "enqueteur":
            dialog_act_list.append(flatten_list(utt['dialog_theme']))
    return flatten_list(dialog_act_list)


def compute_normalized_count(df, column_name):
    df_exploded = df.explode(column_name)
    dialog_act_counts = df_exploded[column_name].value_counts(normalize=True)

    for dialog_act in dialog_act_counts.index:
        col_name = f"{dialog_act}_normalized_counts"
        df[col_name] = df[column_name].apply(lambda x: x.count(dialog_act) / len(x) if dialog_act in x else 0)
    return df



def compute_features(corpus):
    # Add features to the Corpus DataFrame
    corpus["utterance_count"] = corpus["interviews"].apply(lambda x: utterance_count(x))
    corpus["enqueteur_turn_count"] = corpus["interviews"].apply(
        lambda x: speaker_turn_count(x, speaker_id="enqueteur")
    )
    corpus["enqueteur_turn_ratio"] = (
        corpus["enqueteur_turn_count"] / corpus["utterance_count"]
    )
    corpus["average_duration"] = corpus["interviews"].apply(
        lambda x: average_utterance_duration(x)
    )
    corpus["average_duration_enqueteur"] = corpus["interviews"].apply(
        lambda x: average_utterance_duration_by_speaker(x, speaker_id="enqueteur")
    )
    corpus["inter_speaker_turn_duration"] = corpus["interviews"].apply(
        lambda x: inter_speaker_turn_duration(x)
    )

    for unit in ["lemma", "token", "sentences"]:
        corpus[f"enqueteur_utterance_length_{unit}"] = corpus["interviews"].apply(
            lambda x: utterance_length_by_speaker(x, unit=unit, speaker_id="enqueteur")
        )
        corpus[f"enqueteur_average_utterance_length_{unit}"] = corpus[
            "interviews"
        ].apply(
            lambda x: average_utterance_length_by_speaker(
                x, unit=unit, speaker_id="enqueteur"
            )
        )

    for unit in ["lemma", "token", "trf_vect", "w2v_vect", "fasttext_vect"]:
        try:
            corpus[f"previous_sentences_similarity_{unit}"] = corpus[
                "interviews"
            ].apply(lambda x: compute_previous_sentences_similarity(x, unit=unit))
            corpus[f"next_sentences_similarity_{unit}"] = corpus["interviews"].apply(
                lambda x: compute_next_sentences_similarity(x, unit=unit)
            )
            corpus[f"All sentences_similarity_{unit}"] = corpus["interviews"].apply(
                lambda x: conversation_flow(x, unit=unit)
            )
        except:
            pass

    return corpus
