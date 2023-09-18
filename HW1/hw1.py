import re
import sys
import heapq
import nltk
import numpy as np
from sklearn.linear_model import LogisticRegression

# nltk.download('averaged_perceptron_tagger')

negation_words = {'not', 'no', 'never', 'nor', 'cannot'}
negation_enders = {'but', 'however', 'nevertheless', 'nonetheless'}
sentence_enders = {'.', '?', '!', ';'}


# Loads a training or test corpus
# corpus_path is a string
# Returns a list of (string, int) tuples
def load_corpus(corpus_path):
    corpus = []
    with open(corpus_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            text, label = line.split('\t')
            snippets = text.split()
            corpus.append((snippets, label))
    return corpus


# Checks whether or not a word is a negation word
# word is a string
# Returns a boolean
def is_negation(word: str):
    if word in negation_words:
        return True
    if word.endswith("n't"):
        return True
    return False


# Modifies a snippet to add negation tagging
# snippet is a list of strings
# Returns a list of strings
def tag_negation(snippet):
    tagged_snippet = nltk.pos_tag(snippet)
    negating = False
    new_snippet = []
    for i, item in enumerate(tagged_snippet):
        word, tag = item
        if is_negation(word):
            if i == len(tagged_snippet) - 1 or tagged_snippet[i + 1][0] != 'only':
                negating = True
                new_snippet.append(word)
                continue
        if word in sentence_enders | negation_enders or tag in ['JJR', 'RBR']:
            negating = False
        new_word = 'NOT_' + word if negating else word
        new_snippet.append(new_word)

    return new_snippet


# Assigns to each unigram an index in the feature vector
# corpus is a list of tuples (snippet, label)
# Returns a dictionary {word: index}
def get_feature_dictionary(corpus):
    i = 0
    feat_dict = {}
    for snippet, label in corpus:
        for word in snippet:
            if word not in feat_dict:
                feat_dict[word] = i
                i += 1
    return feat_dict


# Converts a snippet into a feature vector
# snippet is a list of tuples (word, pos_tag)
# feature_dict is a dictionary {word: index}
# Returns a Numpy array
def vectorize_snippet(snippet, feature_dict):
    vector = np.zeros(len(feature_dict))
    for word in snippet:
        if word in feature_dict:
            vector[feature_dict[word]] += 1
    return vector


# Trains a classification model (in-place)
# corpus is a list of tuples (snippet, label)
# feature_dict is a dictionary {word: label}
# Returns a tuple (X, Y) where X and Y are Numpy arrays
def vectorize_corpus(corpus, feature_dict):
    X, Y = np.empty((len(corpus), len(feature_dict))), np.empty(len(corpus))
    i = 0
    for snippet, label in corpus:
        X[i, :] = vectorize_snippet(snippet, feature_dict)
        Y[i] = label
        i += 1
    return X, Y


# Performs min-max normalization (in-place)
# X is a Numpy array
# No return value
def normalize(X):
    for c in range(X.shape[1]):
        min_val, max_val = min(X[:, c]), max(X[:, c])
        # print(X[:, c])
        # print(min_val, max_val)
        vfunc = np.vectorize(lambda f: (f - min_val) / (max_val - min_val) if max_val > min_val else 0)
        X[:, c] = vfunc(X[:, c])
    return


# Trains a model on a training corpus
# corpus_path is a string
# Returns a LogisticRegression
def train(corpus_path):
    model = LogisticRegression()
    corpus = [(tag_negation(snippet), label) for snippet, label in load_corpus(corpus_path)]
    feat_dict = get_feature_dictionary(corpus)
    X, Y = vectorize_corpus(corpus, feat_dict)
    normalize(X)
    model.fit(X, Y)
    return model, feat_dict


# Calculate precision, recall, and F-measure
# Y_pred is a Numpy array
# Y_test is a Numpy array
# Returns a tuple of floats
def evaluate_predictions(Y_pred, Y_test):
    # print(np.dstack((Y_pred, Y_test)))
    tp, fp, fn = np.array([(pred == true == 1, pred != true == 0, pred != true == 1)
                           for pred, true in np.dstack((Y_pred, Y_test))[0]]).sum(axis=0)

    precision, recall = tp / (tp + fp), tp / (tp + fn)
    f_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f_score
    # print(tp, fp, fn)
    # print(temp)
    # pass


# Evaluates a model on a test corpus and prints the results
# model is a LogisticRegression
# corpus_path is a string
# Returns a tuple of floats
def test(model, feature_dict, corpus_path):
    corpus = [(tag_negation(snippet), label) for snippet, label in load_corpus(corpus_path)]
    X, y = vectorize_corpus(corpus, feature_dict)
    normalize(X)
    y_pred = model.predict(X)
    return evaluate_predictions(y_pred, y)


# Selects the top k highest-weight features of a logistic regression model
# logreg_model is a trained LogisticRegression
# feature_dict is a dictionary {word: index}
# k is an int
def get_top_features(logreg_model, feature_dict, k=1):
    words = list(feature_dict.keys())
    # print(logreg_model.coef_)
    top_features = [(words[i], weight) for i, weight in
                    heapq.nlargest(k, enumerate(logreg_model.coef_[0]), key=lambda x: abs(x[1]))]
    return top_features
    # pass


def main(args):
    model, feature_dict = train('train.txt')

    print(test(model, feature_dict, 'test.txt'))

    weights = get_top_features(model, feature_dict, k=5)
    for weight in weights:
        print(weight)

    # mini_corpus = load_corpus('train.txt')[:10]
    # print(mini_corpus)
    # print([is_negation(word) for word in "I couldn't believe it's not butter!".split()])
    # print(tag_negation("I could n't believe it 's not butter, but it is !".split()))
    # print(tag_negation("I could n't believe it 's not better butter , it is !".split()))
    # feat_dict = get_feature_dictionary(mini_corpus)
    # print(feat_dict)
    # X, Y = vectorize_corpus(mini_corpus, feat_dict)
    # print((X, Y))
    # normalize(X)
    # print(X)
    # model, feature_dict = train('train.txt')
    # print(model, feature_dict)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
