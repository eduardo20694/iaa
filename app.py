from flask import Flask, request, jsonify
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# 🔹 Carregar modelo DistilBERT para Perguntas e Respostas
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# 🔹 Carregar modelo para Embeddings (Sentence-BERT)
embedding_model = SentenceTransformer('paraphrase-MiniLM-L12-v2')

# 🔹 Banco de Perguntas e Respostas (Definido diretamente no código)
perguntas_respostas = {
    "Qual é o seu nome?": "Meu nome é Eduardo Rodrigues Sparremberger.",
    "Quantos anos você tem?": "Eu tenho 22 anos.",
    "Onde você mora?": "Eu moro em Itati.",
    "Qual é a sua formação acadêmica?": "Eu sou estudante de Análise e Desenvolvimento de Sistemas e estudo sobre JavaScript.",
    "Onde você estuda?": "Eu estudo na Universidade FASUL, no curso de Análise e Desenvolvimento de Sistemas.",
    "Você tem irmãos?": "Sim, eu tenho um irmão mais velho chamado Daniel.",
    "Você tem filhos? Ou esposa?": "Eu estou esperando uma filha chamada Mavie. Sim, tenho esposa chamada Eziane da Silva Eberhardt!",
    "Quais são seus objetivos de vida?": "Meu objetivo é me formar em tecnologia, trabalhar com IA e viajar pelo mundo.",
    "Você tem algum sonho que quer realizar?": "Sim, quero aprender a programar em várias linguagens e criar meu próprio negócio.",
    "Me fale sobre Eziane?": "Eziane da Silva Eberhardt é minha esposa e mãe da minha filha Mavie.",
    "Mavie?": "Mavie é filha de Eduardo Rodrigues Sparremberger e Eziane da Silva Eberhardt.",
    "Alan?": "Alan é filho de Eziane da Silva Eberhardt.",
    "Quantos filhos?": "Uma filha chamada Mavie.",
    "De quem é a Mavie?": "Mavie é filha de Eduardo Rodrigues Sparremberger e Eziane da Silva Eberhardt.",
    "Onde eu moro?": "Eu moro em Itati, Rio Grande do Sul.",
    "Quem vai ganhar o Gauchão?": "Claro que o Grêmio, Sr. Eduardo!!!"
}

# 🔹 Função para gerar embeddings das perguntas
def gerar_embeddings(perguntas):
    return np.array(embedding_model.encode(perguntas, normalize_embeddings=True))  # Normalização melhora precisão

# 🔹 Função para encontrar a melhor resposta
def encontrar_resposta(pergunta_usuario):
    # Criar lista de perguntas armazenadas
    perguntas = list(perguntas_respostas.keys())
    respostas = list(perguntas_respostas.values())

    # Gerar embeddings das perguntas
    perguntas_embeddings = gerar_embeddings(perguntas)
    
    # Gerar embedding da pergunta do usuário
    embedding_pergunta_usuario = embedding_model.encode([pergunta_usuario], normalize_embeddings=True)

    # Calcular a similaridade entre a pergunta do usuário e as perguntas armazenadas
    similaridades = cosine_similarity(embedding_pergunta_usuario, perguntas_embeddings)[0]
    
    # Encontrar a pergunta mais similar
    indice_mais_similar = np.argmax(similaridades)
    maior_similaridade = similaridades[indice_mais_similar]

    # Se a similaridade for alta o suficiente, retornar a resposta armazenada
    if maior_similaridade > 0.6:  # Limite de similaridade
        return respostas[indice_mais_similar]
    
    # Caso contrário, usar DistilBERT para responder
    context = " ".join(respostas)
    result = qa_pipeline(question=pergunta_usuario, context=context)
    
    return result['answer'] if result['score'] > 0.5 else "Desculpe, não consegui encontrar uma resposta precisa para a sua pergunta."

# 🔹 Endpoint da API para perguntas e respostas

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "mensagem": "Bem-vindo à API de Perguntas e Respostas!",
        "instrucoes": "Use o endpoint POST /pergunta para enviar perguntas no formato JSON.",
        "exemplo": {
            "url": "/pergunta",
            "formato": {"pergunta": "sua pergunta aqui"}
        }
    }), 200

@app.route('/pergunta', methods=['POST'])
def responder_pergunta():
    # Obter pergunta do corpo da requisição
    data = request.json
    pergunta_usuario = data.get('pergunta', '')

    if pergunta_usuario:
        resposta = encontrar_resposta(pergunta_usuario)
        return jsonify({'resposta': resposta}), 200
    else:
        return jsonify({'error': 'Pergunta não fornecida'}), 400

# 🔹 Inicializar a API Flask
if __name__ == '__main__':
    # Usar a variável de ambiente PORT fornecida pelo Render (ou 5000 como fallback para desenvolvimento local)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
