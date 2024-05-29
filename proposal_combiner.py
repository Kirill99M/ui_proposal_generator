from joblib import load
from proposal_combiner_model import get_embedding
from proposal_combiner_model import get_distances

loaded_model = load('svm_model.joblib')


def combine(proposal_1: str, proposal_2: str, proposal_3: str):
    lines_1 = proposal_1.splitlines()
    lines_2 = proposal_2.splitlines()
    lines_3 = proposal_3.splitlines()
    for line in lines_1:
        embedding_1 = get_embedding(line)
        for line_2 in lines_2:
            embedding_2 = get_embedding(line_2)
            distance = get_distances(embedding_1, embedding_2)
            result = loaded_model.predict(distance)
            if result == '1':
                lines_2.remove(line_2)

        for line_3 in lines_3:
            embedding_3 = get_embedding(line_3)
            distance = get_distances(embedding_1, embedding_3)
            result = loaded_model.predict(distance)
            if result == '1':
                lines_3.remove(line_3)
