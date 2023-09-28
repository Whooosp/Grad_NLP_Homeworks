import sys

import nltk

from nltk.corpus import brown
import numpy
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from typing import List


# Load the Brown corpus with Universal Dependencies tags
# proportion is a float
# Returns a tuple of lists (sents, tags)
def load_training_corpus(proportion=1.0):
    brown_sentences = brown.tagged_sents(tagset='universal')
    num_used = int(proportion * len(brown_sentences))

    corpus_sents, corpus_tags = [None] * num_used, [None] * num_used
    for i in range(num_used):
        corpus_sents[i], corpus_tags[i] = zip(*brown_sentences[i])
    return corpus_sents, corpus_tags


# Generate word n-gram features
# words is a list of strings
# i is an int
# Returns a list of strings
def get_ngram_features(words: List[str], i: int) -> List[str]:
    gram_words = [words[j] if 0 <= j < len(words) else '<s>' if j < i else '</s>' for j in range(i - 2, i + 3)]
    return [f'prevbigram-{gram_words[1]}',
            f'nextbigram-{gram_words[3]}',
            f'prevskip-{gram_words[0]}',
            f'nextskip-{gram_words[4]}',
            f'prevtrigram-{gram_words[1]}-{gram_words[0]}',
            f'nexttrigram-{gram_words[3]}-{gram_words[4]}',
            f'centertrigram-{gram_words[1]}-{gram_words[3]}']


def get_wordshape(word: str, short=False):
    word = ''.join('d' if c.isdigit() else 'X' if c.isupper() else 'x' for c in word)

    if short:
        new_word = word[0]
        for c in word[1:]:
            if c != new_word[-1]:
                new_word += c
        word = new_word

    return word


# Generate word-based features
# word is a string
# returns a list of strings
def get_word_features(word: str) -> List[str]:
    conditions = {'capital': word[0].isupper(),
                  'allcaps': word.isupper(),
                  f'wordshape-{get_wordshape(word)}': True,
                  f'short-wordshape-{get_wordshape(word, True)}': True,
                  'number': any(c.isdigit() for c in word),
                  'hyphen': any(c == '-' for c in word)}
    prefixes = [f'prefix{j}-{word[:j]}' for j in range(1, min(4, len(word)))]
    suffixes = [f'suffix{j}-{word[-j:]}' for j in range(1, min(4, len(word)))]
    return [f'word-{word}'] + [key for key, val in conditions.items() if val] + prefixes + suffixes


# Wrapper function for get_ngram_features and get_word_features
# words is a list of strings
# i is an int
# prevtag is a string
# Returns a list of strings
def get_features(words: List[str], i: int, prevtag: str) -> List[str]:
    ngram_features = [s.lower() for s in get_ngram_features(words, i)]
    word_features = [s if 'wordshape' in s else s.lower() for s in get_word_features(words[i])]
    return ngram_features+word_features+[f'tagbigram-{prevtag.lower()}']


# Remove features that occur fewer than a given threshold number of time corpus_features is a list of lists,
# where each sublist corresponds to a sentence and has elements that are lists of strings (feature names) threshold
# is an int Returns a tuple (corpus_features, common_features)
def remove_rare_features(corpus_features, threshold=5):
    pass


# Build feature and tag dictionaries
# common_features is a set of strings
# corpus_tags is a list of lists of strings (tags)
# Returns a tuple (feature_dict, tag_dict)
def get_feature_and_label_dictionaries(common_features, corpus_tags):
    pass


# Build the label vector Y
# corpus_tags is a list of lists of strings (tags)
# tag_dict is a dictionary {string: int}
# Returns a Numpy array
def build_Y(corpus_tags, tag_dict):
    pass


# Build a sparse input matrix X corpus_features is a list of lists, where each sublist corresponds to a sentence and
# has elements that are lists of strings (feature names) feature_dict is a dictionary {string: int} Returns a
# Scipy.sparse csr_matrix
def build_X(corpus_features, feature_dict):
    pass


# Train an MEMM tagger on the Brown corpus
# proportion is a float
# Returns a tuple (model, feature_dict, tag_dict)
def train(proportion=1.0):
    pass


# Load the test set
# corpus_path is a string
# Returns a list of lists of strings (words)
def load_test_corpus(corpus_path):
    with open(corpus_path) as inf:
        lines = [line.strip().split() for line in inf]
    return [line for line in lines if len(line) > 0]


# Predict tags for a test sentence
# test_sent is a list containing a single list of strings
# model is a trained LogisticRegression
# feature_dict is a dictionary {string: int}
# reverse_tag_dict is a dictionary {int: string}
# Returns a tuple (Y_start, Y_pred)
def get_predictions(test_sent, model, feature_dict, reverse_tag_dict):
    pass


# Perform Viterbi decoding using predicted log probabilities
# Y_start is a Numpy array of size (1, T)
# Y_pred is a Numpy array of size (n-1, T, T)
# Returns a list of strings (tags)
def viterbi(Y_start, Y_pred):
    pass


# Predict tags for a test corpus using a trained model
# corpus_path is a string
# model is a trained LogisticRegression
# feature_dict is a dictionary {string: int}
# tag_dict is a dictionary {string: int}
# Returns a list of lists of strings (tags)
def predict(corpus_path, model, feature_dict, tag_dict):
    pass


def main(args):
    words = ['the', 'Happy', 'cat']
    print(get_ngram_features(words, 2))
    print(get_wordshape('HeO2223'))
    print(get_features(words, 1, 'DT'))
    # model, feature_dict, tag_dict = train(0.25)
    #
    # predictions = predict('test.txt', model, feature_dict, tag_dict)
    # for test_sent in predictions:
    #     print(test_sent)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
