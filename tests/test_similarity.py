

def test_vect_legth(corpus):
    for j in range(len(corpus.corpus['interviews'])):
        for i in range(len(corpus.corpus['interviews'][j])):
            assert len(corpus.corpus['interviews'][j][i].trf_sentence_vect) == len(corpus.corpus['interviews'][j][i].sentences)
            assert len(corpus.corpus['interviews'][j][i].w2v_sentence_vect) == len(corpus.corpus['interviews'][j][i].sentences)
            assert len(corpus.corpus['interviews'][j][i].fasttext_sentence_vect) == len(corpus.corpus['interviews'][j][i].sentences)