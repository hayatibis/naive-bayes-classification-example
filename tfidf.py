from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import os

corpus = []

pos_prob_vector = []
neg_prob_vector = []

positiveCount = 0
negativeCount = 0

for file in os.listdir("MRDataset/train/pos"):
    f = open("MRDataset/train/pos/" + file, encoding='utf-8')
    context = f.read()
    corpus.append(context)
    positiveCount += 1

for file in os.listdir("MRDataset/train/neg"):
    f = open("MRDataset/train/neg/" + file, encoding='utf-8')
    context = f.read()
    corpus.append(context)
    negativeCount += 1

count_vect = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.008, max_df=0.4)
transformer = TfidfTransformer();

train_V = count_vect.fit_transform(corpus)
train_tf = transformer.fit_transform(train_V)

features = count_vect.get_feature_names()
# print(list)
print(len(features))

print("pos" + str(positiveCount))
print("neg" + str(negativeCount))

# print(train_V)
# print(train_tf)

train_tf = train_tf.toarray()

n_word_pos = np.sum(train_tf[:positiveCount], axis=0)
n_word_neg = np.sum(train_tf[positiveCount:], axis=0)

total_word_pos = np.sum(n_word_pos)
total_word_neg = np.sum(n_word_neg)

pos_prob_vector = ((n_word_pos + 1) / (total_word_pos + len(features)))
neg_prob_vector = ((n_word_neg + 1) / (total_word_neg + len(features)))

print(neg_prob_vector.shape)
print(pos_prob_vector.shape)

# Test
test = []
testFileNames = []
testLabels = []

for file in os.listdir("MRDataset/test/pos"):
    f = open("MRDataset/test/pos/" + file, encoding='utf-8')
    context = f.read()
    testFileNames.append(file)
    test.append(context)
    testLabels.append("pos")

for file in os.listdir("MRDataset/test/neg"):
    f = open("MRDataset/test/neg/" + file, encoding='utf-8')
    context = f.read()
    testFileNames.append(file)
    test.append(context)
    testLabels.append("neg")

test = count_vect.transform(test)


def naiveBayes(pos_prob_vector, neg_prob_vector, test):
    estimatedLabels = []
    for item in test:
        test_V = item.toarray()
        posProb = np.dot(test_V, pos_prob_vector)
        negProb = np.dot(test_V, neg_prob_vector)
        if posProb > negProb:
            estimatedLabels.append("pos")
        else:
            estimatedLabels.append("neg")

    return estimatedLabels


LabelResult = naiveBayes(pos_prob_vector, neg_prob_vector, test)


def accuracy(EstimatedLabels, TrueLabels):
    count = 0
    for i in range(len(EstimatedLabels)):
        if EstimatedLabels[i] == TrueLabels[i]:
            count += 1
    return (count / len(EstimatedLabels)) * 100


acc = accuracy(LabelResult, testLabels)

output = open("result.txt", "w")

output.write('ACCURACY :' + str(acc))

for i in range(len(LabelResult)):
    output.write(
        str( '\n'+  testFileNames[i]) + ' TrueLabel :' + str(testLabels[i]) + ' EstimatedLabel :' + str(LabelResult[i]))

output.close()




