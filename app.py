import streamlit as st
import tensorflow as tf
import numpy as np

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Análise de Sentimentos",
    page_icon="🤖",
    layout="centered"
)

# --- CARREGAMENTO DOS MODELOS (com cache para performance) ---
@st.cache_resource
def carregar_modelos():
    """
    Carrega o modelo Keras original para obter a camada de vetorização
    e o modelo TFLite para inferência.
    """
    try:
        # Carrega o modelo original para pegar a camada de pré-processamento
        model_original = tf.keras.models.load_model('modelo_final_tweet.keras')
        # ATENÇÃO: Use o nome correto da sua camada de TextVectorization aqui!
        text_vectorization_layer = model_original.get_layer('text_vectorization') 
        print("Camada de Vetorização carregada.")

        # Carrega o modelo TFLite e aloca os tensores
        interpreter = tf.lite.Interpreter(model_path='modelo.tflite')
        interpreter.allocate_tensors()
        print("Interpretador TFLite carregado.")
        
        return text_vectorization_layer, interpreter
    except Exception as e:
        st.error(f"Erro ao carregar os modelos: {e}")
        st.error("Verifique se os arquivos 'modelo_tweet.keras' e 'modelo.tflite' estão na mesma pasta que o app.py.")
        return None, None

# Carrega os modelos e lida com possíveis erros
vectorizer, interpreter = carregar_modelos()

# --- FUNÇÃO DE PREDIÇÃO ---
def prever_sentimento(texto):
    """
    Recebe um texto, pré-processa e retorna o sentimento e a probabilidade.
    """
    if not texto.strip():
        return None, None

    # Detalhes de entrada e saída do modelo TFLite
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 1. Pré-processar o texto para tokens numéricos
    tokens = vectorizer([texto])
    
    # 2. Enviar os tokens para o modelo TFLite
    interpreter.set_tensor(input_details[0]['index'], tokens)
    
    # 3. Executar a inferência
    interpreter.invoke()
    
    # 4. Obter o resultado
    predicao = interpreter.get_tensor(output_details[0]['index'])
    
    probabilidade = predicao[0][0]
    sentimento = "Positivo" if probabilidade > 0.5 else "Negativo"
    
    return sentimento, probabilidade

# --- INTERFACE DO USUÁRIO (UI) ---
st.title("🤖 Análise de Sentimentos de Texto")
st.write(
    "Este aplicativo utiliza um modelo de Deep Learning (LSTM) para classificar "
    "o sentimento de um texto em Inglês como Positivo ou Negativo."
)

st.markdown("---")

# Exemplos prontos
st.subheader("💡 Experimente com exemplos prontos")
# ALTERAÇÃO: Mudei de 3 colunas para 2
col1, col2 = st.columns(2)

frase_positiva = "this is a great tweet and the model is going to work perfectly"
frase_negativa = "what an awful post, I'm not happy with this at all"
# ALTERAÇÃO: Variável 'frase_neutra' foi removida

with col1:
    if st.button("Exemplo Positivo"):
        st.session_state.texto_usuario = frase_positiva
with col2:
    if st.button("Exemplo Negativo"):
        st.session_state.texto_usuario = frase_negativa

# ALTERAÇÃO: O bloco de código da 'col3' com o botão "Exemplo Neutro" foi completamente removido.

# Área de texto para o usuário
texto_input = st.text_area(
    "Ou digite seu próprio texto abaixo (em Inglês):", 
    value=st.session_state.get("texto_usuario", ""), 
    height=150,
    key="texto_usuario" 
)

# Botão para analisar
if st.button("Analisar Sentimento", type="primary"):
    if interpreter and vectorizer and texto_input:
        with st.spinner("Analisando..."):
            sentimento, probabilidade = prever_sentimento(texto_input)

            if sentimento == "Positivo":
                st.success(f"**Sentimento: {sentimento}**")
                st.progress(probabilidade)
                st.metric(label="Confiança na Previsão", value=f"{probabilidade:.2%}")
            else:
                st.error(f"**Sentimento: {sentimento}**")
                st.progress(1 - probabilidade)
                st.metric(label="Confiança na Previsão", value=f"{1 - probabilidade:.2%}")

    elif not texto_input:
        st.warning("Por favor, digite um texto para analisar.")

st.markdown("---")
st.info("Lembre-se: O modelo foi treinado com um conjunto de dados específico e sua precisão pode variar com textos fora desse contexto.")