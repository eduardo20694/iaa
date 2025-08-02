from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging

# Configuração básica de logging
logging.basicConfig(level=logging.INFO)

# Inicializa app Flask e CORS
app = Flask(__name__)
CORS(app)

# Carrega modelo de embeddings (modelo leve e eficiente)
embedding_model = SentenceTransformer('paraphrase-MiniLM-L12-v2')

# Perguntas e respostas fixas
perguntas_respostas = {
    "Qual é o seu nome?": "Meu nome é Eduardo Rodrigues Sparremberger.",
    "Quantos anos você tem?": "Eu tenho 23 anos.",
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
    "Quem vai ganhar o Gauchão?": "Claro que o Grêmio, Sr. Eduardo!!!",
    "Qual seu signo?": "Peixes",
    "Qual sua data de nascimento?": "11:03:2002",
    "Onde você nasceu?": "Torres",
    "Qual nome da sua mãe?": "Eraci Rodrigues Sparremberger",
    "Qual nome do seu pai?": "Enio Klippel Sparremberger",
    "Qual foi o primeiro dia que Eduardo e Eziane se conversaram?": "Dia 23/03/2022",
    "Qual foi o primeiro encontro?": "Dia 15/04/2022",
    "Qual é o seu apelido?": "Me chamam de Dudu.",
    "Qual é a sua altura?": "Eu tenho 1,93m de altura.",
    "Qual a sua cor favorita?": "Minha cor favorita é azul.",
    "Qual é o seu prato favorito?": "Eu gosto muito de churrasco gaúcho."
}

# Pré-calcula perguntas, respostas e embeddings
perguntas = list(perguntas_respostas.keys())
respostas = list(perguntas_respostas.values())
logging.info("Calculando embeddings das perguntas fixas...")
perguntas_embeddings = embedding_model.encode(perguntas, normalize_embeddings=True)
logging.info("Embeddings calculadas.")

def encontrar_resposta(pergunta_usuario: str) -> str:
    """Encontra a resposta mais próxima pela similaridade do embedding."""
    if not pergunta_usuario.strip():
        return "Por favor, envie uma pergunta válida."

    embedding_usuario = embedding_model.encode([pergunta_usuario], normalize_embeddings=True)
    similaridades = cosine_similarity(embedding_usuario, perguntas_embeddings)[0]
    indice = np.argmax(similaridades)
    maior_similaridade = similaridades[indice]

    logging.info(f"Similaridade encontrada: {maior_similaridade:.3f} para pergunta '{perguntas[indice]}'")

    LIMIAR = 0.6
    if maior_similaridade >= LIMIAR:
        return respostas[indice]
    else:
        return "Desculpe, não consegui encontrar uma resposta precisa para a sua pergunta."

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
    data = request.json
    pergunta = data.get('pergunta', '') if data else ''
    resposta = encontrar_resposta(pergunta)
    return jsonify({'resposta': resposta})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"Iniciando servidor na porta {port}...")
    app.run(host='0.0.0.0', port=port)
