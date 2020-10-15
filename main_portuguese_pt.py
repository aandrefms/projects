# encoding: utf-8
import yaml
import nltk
import numpy as np
import tensorflow as tf
import random
import os
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
# import warnings
import warnings
warnings.filterwarnings("ignore")


# Usando try para verificar/carregar os mmodelos e variáveis
try:
    from keras.models import load_model
    model = load_model('./model_chatbot.hdf5')
    with open('variavel_x', 'rb') as f:
        docs_x = pickle.load(f)
    with open('variavel_y', 'rb') as f:
        docs_y = pickle.load(f)
    with open('variavel_le', 'rb') as f:
        le = pickle.load(f)


except:
    # Carregando os dados
    directory = r'./input/'
    data = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), encoding="utf-8") as f:
            arquivo = yaml.load(f, Loader=yaml.FullLoader)
            data.append(arquivo)
    # Criando listas vazias para receber as palavras e as categorias
    words = []
    classes = []
    docs_x = []
    docs_y = []
    # Looping pelos dados para armazenar as palavras
    for categoria in data:
        a = len(docs_x)
        for item in categoria['conversations']:
            wrds = nltk.word_tokenize(item[0])
            words.extend(wrds)
            docs_x.append(wrds)
        for i in range(a, len(docs_x)):
            docs_y.append(categoria['categories'][0])
        if categoria['categories'] not in classes:
            classes.append(categoria['categories'][0])

    # Limpando os dados
    '''stemmer = nltk.stem.RSLPStemmer()
    words = [stemmer.stem(w.lower()) for w in words if w not in '?']'''
    words = [w.lower() for w in words if w not in '?']
    words = sorted(list(set(words)))
    labels = sorted(classes)

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    # Misturando os dados
    c = list(zip(docs_x,docs_y))
    random.shuffle(c)
    docs_x, docs_y = zip(*c)

    # Salvando variáveis
    with open('variavel_x', 'wb') as f:
        pickle.dump(docs_x, f)
    with open('variavel_y', 'wb') as f:
        pickle.dump(docs_y, f)

    # Aplicando pré-processamento de dados (NLP)
    tokenizer = Tokenizer(num_words=300)
    tokenizer.fit_on_texts(docs_x)
    list_tokenized_train = tokenizer.texts_to_sequences(docs_x)
    maxlen = 20
    X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(labels)

    y = le.transform(docs_y)
    y = np.array(y)
    with open('variavel_le', 'wb') as f:
        pickle.dump(le, f)

    # Construindo a rede neural(Modelando)
    embed_size = 64
    model = Sequential()
    model.add(Embedding(300, embed_size))
    model.add(Bidirectional(LSTM(64, return_sequences = True)))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))

    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    batch_size = 16
    metric = 'val_accuracy'
    checkpointer = ModelCheckpoint(filepath = 'model_chatbot.hdf5', verbose=1, save_best_only = True, monitor=metric)
    model.fit(X_t,y, batch_size=batch_size, epochs=500, validation_split=0.2, callbacks=[checkpointer])


# Função para receber a frase e processa-la
def cleaning_text(frase):
    stemmer = nltk.stem.RSLPStemmer()
    text = [stemmer.stem(w.lower()) for w in frase.split(" ")]
    text = " ".join(text)
    return text


resp1 = []
data_files = []
pasta = r'./input/'
for filename in os.listdir(pasta):
    with open(os.path.join(pasta, filename), encoding="utf-8") as f:
        arq = yaml.load(f, Loader=yaml.FullLoader)
        data_files.append(arq)

# Função para armazenar as intenções
def get_label(categoria):
    # Loading data
    resp2 = []
    for label in data_files:
        if label['categories'][0] == categoria:
            for item1 in label['conversations']:
                resp2.append(item1[1])
    return random.choice(resp2)


tokenizer = Tokenizer(num_words=300)
tokenizer.fit_on_texts(docs_x)

# Função principal do chatbot
def chat():
    print('Comece falando com o Bot!')
    while True:
        list_resp = []
        inp = input('You :')
        if inp.lower() == 'quit':
            break
        list_resp.append(inp)
        resposta = tokenizer.texts_to_sequences(list_resp)
        resposta = pad_sequences(resposta, maxlen=20)
        resp = np.argmax(model.predict(resposta) > 0.5, axis=-1).astype("int32")
        resp_final = le.inverse_transform(resp)
        resp_final = get_label(resp_final)
        print(f'Bot: {resp_final}')


chat()
