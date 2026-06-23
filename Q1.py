import pypdf
from transformers import pipeline
import warnings

# Ocultar avisos que podem poluir o terminal
warnings.filterwarnings("ignore")

# 1. Extração do texto do documento PDF
leitor_pdf = pypdf.PdfReader("CONTRATO DE LOCAÇÃO RESIDENCIAL.pdf")
texto_contrato = ""

for pagina in leitor_pdf.pages:
    texto_contrato += pagina.extract_text() + "\n"

print("Carregando o modelo de Inteligência Artificial (Isso pode levar alguns segundos na CPU)...")

# 2. Carregamento do modelo local forçando o uso do processador (CPU)
# O parâmetro device=-1 garante que rodará em qualquer computador genérico
qa_pipeline = pipeline(
    task="question-answering",
    model="pierreguillou/bert-base-cased-squad-v1.1-portuguese",
    device=-1
)

# 3. Definição das perguntas solicitadas no trabalho
perguntas = [
    "Qual o nome do Locador?",
    "Qual o CPF do Locatário?",
    "Qual o endereço do imóvel?",
    "Qual o valor do aluguel?"
]

# 4. Execução da busca semântica
print("\n--- INICIANDO BUSCA SEMÂNTICA ---")
for pergunta in perguntas:
    resultado = qa_pipeline(question=pergunta, context=texto_contrato)
    
    print(f"Pergunta: {pergunta}")
    print(f"Resposta: {resultado['answer']}\n")