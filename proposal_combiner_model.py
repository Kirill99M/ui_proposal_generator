from typing import List, Tuple

import tensorflow as tf

import tensorflow_hub as hub
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.svm import SVC
from joblib import dump

model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def parse_file(file_path: str) -> Tuple[List[str], List[str], List[str]]:
    with open(file_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
        labels, s1, s2 = zip(*[
            (tokens[0][0], tokens[3], tokens[4])
            for tokens in [line.strip().split('\t') for line in lines]
        ])
        return labels, s1, s2


def get_embedding(text):
    embedding = model([text])
    return np.array(embedding).flatten()


def get_distances(embeddings_s1, embeddings_s2):
    distances = [cosine(embedding_s1, embedding_s2) for embedding_s1, embedding_s2 in
                 zip(embeddings_s1, embeddings_s2)]
    return np.array(distances).reshape(-1, 1)


labels_train, s1_train, s2_train = parse_file("msr_paraphrase_train.txt")
labels_test, s1_test, s2_test = parse_file("msr_paraphrase_test.txt")
with tf.device('/CPU:0'):
    embeddings_s1_train = [get_embedding(text) for text in s1_train]
    embeddings_s2_train = [get_embedding(text) for text in s2_train]
    distances_train = get_distances(embeddings_s1_train, embeddings_s2_train)
    svm_model = SVC(kernel='rbf')
    svm_model.fit(distances_train, labels_train)
    dump(svm_model, 'svm_model.joblib')
