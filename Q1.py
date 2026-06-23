import pypdf
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Ocultar avisos do terminal para manter a interface limpa
warnings.filterwarnings("ignore")

# 1. Extração do texto do documento PDF
leitor_pdf = pypdf.PdfReader("CONTRATO DE LOCAÇÃO RESIDENCIAL.pdf")
texto_contrato = ""
for pagina in leitor_pdf.pages:
    texto_contrato += pagina.extract_text() + " "

print("Carregando modelo e tokenizador (BERT)...")

# 2. Carregamento explícito de Tokenizador e Modelo
nome_modelo = "pierreguillou/bert-base-cased-squad-v1.1-portuguese"
tokenizador = AutoTokenizer.from_pretrained(nome_modelo)
modelo = AutoModelForQuestionAnswering.from_pretrained(nome_modelo)

# 3. Definição da Função de Inferência Tensorial
def buscar_resposta(pergunta, contexto):
    # Conversão do texto em matrizes numéricas (Tensores PyTorch)
    entradas = tokenizador(pergunta, contexto, return_tensors="pt", truncation=True, max_length=512)
    
    # Processamento pela rede neural (Forward Pass)
    with torch.no_grad():
        saidas = modelo(**entradas)
    
    # Identificação dos tokens de maior probabilidade para início e término
    indice_inicio = torch.argmax(saidas.start_logits)
    indice_fim = torch.argmax(saidas.end_logits)
    
    # Decodificação do tensor resultante para texto
    tokens_resposta = entradas["input_ids"][0][indice_inicio:indice_fim + 1]
    return tokenizador.decode(tokens_resposta, skip_special_tokens=True)


# 4. Interface de Busca Semântica Interativa
print("--- MODO DE BUSCA SEMÂNTICA INTERATIVA ---")
print("Digite 'sair' para encerrar o programa.\n")

while True:
    # Captura a entrada do usuário no terminal
    pergunta_usuario = input("Faça sua pergunta sobre o contrato: ")
    
    # Condição de parada do loop
    if pergunta_usuario.lower().strip() == 'sair':
        print("Encerrando a busca semântica.")
        break
    
    # Executa a função e exibe o resultado
    resposta_usuario = buscar_resposta(pergunta_usuario, texto_contrato)
    print(f"Resposta: {resposta_usuario}\n")