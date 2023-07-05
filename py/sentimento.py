import string

import nltk
from flask import Flask
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

# fazendo download do Tokenizador (utilizado para dividir as palavras do texto)
nltk.download("punkt")
# fazendo download dos StopWords (utiizado para remover palavras desnecessárias (ex: a, de, uma, um))
nltk.download("stopwords")

# método utilizado para processar/tratar o texto
def preprocess_text(sentence):
  # tirando as pontuações
  sentence = sentence.translate(str.maketrans("", "", string.punctuation))

  # colocando tudo em caixa baixa
  sentence = sentence.lower()

  # cria um vetor com as palavras do texto
  words = word_tokenize(sentence)

  # colocando apenas palavras em pt
  stop_words = set(stopwords.words("portuguese"))

  # removendo os "stop words (Uma, e, o..)" e colocano o resto no vetor
  words = [word for word in words if word not in stop_words]

  # objeto que vai mudar o verbo das palavras
  stemmer = PorterStemmer()

  # troca o verbo das palavras para o primitivo ex:(andando = andar)
  words = [stemmer.stem(word) for word in words]

  # adiciona na sentence os words filtrados
  sentence = ' '.join(words)

  return sentence

# criando uma base de teste
train_data = [
    ("Eu amo este produto", "positivo"),
    ("Este produto é horrível", "negativo"),
    ("O filme foi incrível", "positivo"),
    ("Não gostei do serviço", "negativo")
]

# TF = quantidade de ocorrências de uma palavra no texto / quantidade de palavras do texto
# IDF = quantidade de textos / numero de textos que contem a palavra em análise
# TF * IDF = para cada texto haverá um vetor, onde a palavra será identificada pelo resultado dessa multiplicação
vectorizer = TfidfVectorizer(preprocessor=preprocess_text)

# vetorizando as frases para um formato entendivel para treinamento
train_features = vectorizer.fit_transform([x[0] for x in train_data])
print(train_features)

# cria um vetor com os sentimentos de cada  um dos textos
train_labels = [x[1] for x in train_data]

# cria um modelo de classificação para no final encontrar o melhor hiperplano entre as classes
classifier = svm.SVC(kernel="linear")

# treinando o classificador com as frases e os sentimentos -- treinar o svc para aumentar a margem entre as classes (sentimentos)
classifier.fit(train_features, train_labels)

# método utilizado para classificar o sentimento de um texto
def predict_sentiment(sentence):
  # realizando o pré-processamento do texto
  sentence = preprocess_text(sentence)

  # vetorizando o texto
  features = vectorizer.transform([sentence])

  # classificando o sentimento de acordo com o exto vetorizado
  sentiment = classifier.predict(features)[0]

  return f"<h2> {sentiment} </h2>"

# criando um app utilizando flask
app = Flask(__name__)

# criando uma rota para receber um texto de sentimento
@app.route("/sentimento/<frase>")
def start(frase):
  return predict_sentiment(frase)

# rodando a api na porta 5000
if __name__ == "__main__":
  app.run()