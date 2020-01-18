from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import ensemble
import pandas, xgboost, numpy
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
from ttictoc import TicToc
from gensim.models import Word2Vec
import os
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

####################################
#   Dataset Prep
####################################

corpus_dir = "corpus"

# get file names
topic = input("Please enter the topic to work on: ")
topic_path = os.path.join(corpus_dir, topic)
list_files = os.listdir(topic_path)
pos_file = os.path.join(topic_path, [path for path in list_files if "non" not in path][0])
neg_file = os.path.join(topic_path, [path for path in list_files if "non" in path][0])

# load the dataset
print("Dataset is being read and processed...")
pos_df = pandas.read_csv(filepath_or_buffer=pos_file)
pos_data = pos_df['opcode'].tolist()
pos_data = [" ".join(data.split(",")) for data in pos_data]

neg_df = pandas.read_csv(filepath_or_buffer=neg_file)
neg_data = neg_df['opcode'].tolist()
neg_data = [" ".join(data.split(",")) for data in neg_data]


X = pos_data + neg_data
y = list(numpy.ones(len(pos_data))) + list(numpy.zeros(len(neg_data)))

# create a dataframe using texts and lables
trainDF = pandas.DataFrame()
trainDF['text'] = X
trainDF['label'] = y

# split the dataset into training and validation datasets
test_size = 0.25
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'], test_size=test_size)
print("Working on train set size: {} and test set size: {}".format(len(train_x), len(valid_x)))

# label encode the target variable
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)


####################################
#   Feature Engineering
####################################

print("Count vectorizer is created...")

# create a count vectorizer object
count_vect = CountVectorizer(analyzer='word')
count_vect.fit(trainDF['text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

print("Tf-idf vectorizer is created...")

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf = tfidf_vect.transform(train_x)
xvalid_tfidf = tfidf_vect.transform(valid_x)

print("N-gram tf-idf vectorizer is created...")

# ngram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(5,5), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram = tfidf_vect_ngram.transform(valid_x)


print("############################################")

# load the pre-trained word-embedding vectors
word2vec = Word2Vec.load("test_w2v.model")

vocabulary = list(word2vec.wv.vocab)

embeddings_index = {}
for voc, vals in zip(vocabulary, word2vec[vocabulary]):
    embeddings_index[voc] = vals

# create a tokenizer
token = text.Tokenizer()
token.fit_on_texts(trainDF['text'])
word_index = token.word_index

max_len = 1000

# convert text to sequence of tokens and pad them to ensure equal length vectors
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=max_len)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=max_len)

# create token-embedding mapping
embedding_matrix = numpy.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index[word]
    embedding_matrix[i] = embedding_vector

####################################
#   Model Building
####################################


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    t = TicToc()  ## TicToc("name")
    t.tic()
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    t.toc()
    print("Time elapsed to train: {}".format(t.elapsed))
    # return metrics.f1_score(predictions, valid_y) * 100
    tn, fp, fn, tp = metrics.confusion_matrix(predictions, valid_y).ravel()
    print("Confusion matrix results:")
    print("True neg: {}\nFalse pos: {}\nFalse neg: {}\nTrue pos:{}".format(tn, fp, fn, tp))
    print("---")
    return metrics.accuracy_score(predictions, valid_y) * 100


# Naive Bayes on Count Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print("NB, Count Vectors: ", accuracy)
print()

# Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print("NB, WordLevel TF-IDF: ", accuracy)
print()

# Naive Bayes on Ngram Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("NB, N-Gram Vectors: ", accuracy)
print("############################################")

# ############################################

# Linear Classifier on Count Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print("LR, Count Vectors: ", accuracy)
print()

# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print("LR, WordLevel TF-IDF: ", accuracy)
print()

# Linear Classifier on Ngram Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("LR, N-Gram Vectors: ", accuracy)
print("############################################")

# ############################################

# SVM on Ngram Level TF IDF Vectors
accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("SVM, N-Gram Vectors: ", accuracy)
print("############################################")

# ############################################

# RF on Count Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
print("RF, Count Vectors: ", accuracy)
print()

# RF on Word Level TF IDF Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
print("RF, WordLevel TF-IDF: ", accuracy)
print()


accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("RF, N-Gram Vectors: ", accuracy)
print("############################################")

# ############################################

# Extereme Gradient Boosting on Count Vectors
accuracy = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc())
print("Xgb, Count Vectors: ", accuracy)
print()

# Extereme Gradient Boosting on Word Level TF IDF Vectors
accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())
print("Xgb, WordLevel TF-IDF: ", accuracy)
print()

accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("Xgb, N-Gram Vectors: ", accuracy)
print("############################################")

# ############################################

def create_model_architecture(input_size):
    # create input layer
    input_layer = layers.Input((input_size,), sparse=False)

    # create hidden layer
    hidden_layer = layers.Dense(100, activation="relu")(input_layer)

    # create output layer
    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)

    classifier = models.Model(inputs=input_layer, outputs=output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return classifier


classifier = create_model_architecture(xtrain_tfidf_ngram.shape[1])
accuracy = train_model(classifier, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, is_neural_net=True)
print("NN, Ngram Level TF IDF Vectors", accuracy)
print("############################################")

############################################

def create_cnn():
    # Add an Input Layer
    input_layer = layers.Input((max_len,))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 100, weights=[embedding_matrix], trainable=False)(
        input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model


classifier = create_cnn()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print("CNN, Word Embeddings", accuracy)


############################################

def create_rnn_lstm():
    # Add an Input Layer
    input_layer = layers.Input((max_len,))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 100, weights=[embedding_matrix], trainable=False)(
        input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = layers.LSTM(100)(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model


classifier = create_rnn_lstm()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print("RNN-LSTM, Word Embeddings", accuracy)

############################################

def create_rnn_gru():
    # Add an Input Layer
    input_layer = layers.Input((max_len,))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 100, weights=[embedding_matrix], trainable=False)(
        input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the GRU Layer
    lstm_layer = layers.GRU(100)(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model


classifier = create_rnn_gru()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print("RNN-GRU, Word Embeddings", accuracy)

############################################

def create_bidirectional_rnn():
    # Add an Input Layer
    input_layer = layers.Input((max_len,))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 100, weights=[embedding_matrix], trainable=False)(
        input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the LSTM Layer
    lstm_layer = layers.Bidirectional(layers.GRU(100))(embedding_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(lstm_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model


classifier = create_bidirectional_rnn()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print("RNN-Bidirectional, Word Embeddings", accuracy)

############################################

def create_rcnn():
    # Add an Input Layer
    input_layer = layers.Input((max_len,))

    # Add the word embedding Layer
    embedding_layer = layers.Embedding(len(word_index) + 1, 100, weights=[embedding_matrix], trainable=False)(
        input_layer)
    embedding_layer = layers.SpatialDropout1D(0.3)(embedding_layer)

    # Add the recurrent layer
    rnn_layer = layers.Bidirectional(layers.GRU(50, return_sequences=True))(embedding_layer)

    # Add the convolutional Layer
    conv_layer = layers.Convolution1D(100, 3, activation="relu")(embedding_layer)

    # Add the pooling Layer
    pooling_layer = layers.GlobalMaxPool1D()(conv_layer)

    # Add the output Layers
    output_layer1 = layers.Dense(50, activation="relu")(pooling_layer)
    output_layer1 = layers.Dropout(0.25)(output_layer1)
    output_layer2 = layers.Dense(1, activation="sigmoid")(output_layer1)

    # Compile the model
    model = models.Model(inputs=input_layer, outputs=output_layer2)
    model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return model


classifier = create_rcnn()
accuracy = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print("CNN, Word Embeddings", accuracy)