from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

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

# 🔹 Função principal para interagir com a IA
def interagir_com_ia():
    print("🤖 IA: Olá! Como posso ajudá-lo?")
    
    while True:
        pergunta_usuario = input("\nVocê: ")
        
        if pergunta_usuario.lower() in ['sair', 'exit', 'quit']:
            print("🤖 IA: Até logo!")
            break
        
        resposta = encontrar_resposta(pergunta_usuario)
        print(f"🤖 IA: {resposta}")

# 🔹 Executando a interação com a IA
if __name__ == "__main__":
    interagir_com_ia()
