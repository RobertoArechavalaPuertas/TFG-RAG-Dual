import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from retriever import cargar_vector_store, recuperar_contexto
from retriever_dsm5 import cargar_vector_store_dsm5, recuperar_contexto_dsm5

# Configuración
load_dotenv()

OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Prompt template dual
PROMPT_TEMPLATE = """Eres un asistente médico especializado. Dispones de dos fuentes de información:
1. El historial clínico del paciente {patient_id}.
2. Criterios diagnósticos y conocimiento clínico del DSM-5.

Reglas estrictas:
- Responde SIEMPRE en español.
- Para datos específicos del paciente (medicación, analíticas, diagnósticos, \
antecedentes), usa ÚNICAMENTE el historial clínico.
- Para criterios diagnósticos, definiciones clínicas o conocimiento psiquiátrico \
general, puedes complementar con el DSM-5.
- Si la información no está en ninguna de las dos fuentes, di exactamente: \
"No encuentro esa información en las fuentes disponibles."
- No inventes datos, medicaciones ni diagnósticos.
- Sé claro, estructurado y preciso.

HISTORIAL CLÍNICO DEL PACIENTE {patient_id}:
{contexto_historial}

CONOCIMIENTO DSM-5 RELEVANTE:
{contexto_dsm5}

PREGUNTA: {pregunta}

RESPUESTA:"""

prompt = PromptTemplate(
    input_variables=["patient_id", "contexto_historial", "contexto_dsm5", "pregunta"],
    template=PROMPT_TEMPLATE,
)


# Inicializar LLM
def cargar_llm() -> OllamaLLM:
    return OllamaLLM(
        model       = OLLAMA_MODEL,
        base_url    = OLLAMA_BASE_URL,
        temperature = 0.1,
    )


# Chain dual
def construir_chain(llm, vs_pacientes, vs_dsm5):
    def chain(patient_id: str, pregunta: str) -> str:
        contexto_historial = recuperar_contexto(patient_id, pregunta, vs_pacientes)

        contexto_dsm5 = recuperar_contexto_dsm5(pregunta, vs_dsm5)
        if not contexto_dsm5:
            contexto_dsm5 = "No hay información DSM-5 relevante para esta consulta."

        prompt_final = prompt.format(
            patient_id         = patient_id,
            contexto_historial = contexto_historial,
            contexto_dsm5      = contexto_dsm5,
            pregunta           = pregunta,
        )

        respuesta = llm.invoke(prompt_final)
        return respuesta.strip()

    return chain


# Prueba rápida
if __name__ == "__main__":
    print("Cargando vector stores y LLM...\n")
    vs_pacientes = cargar_vector_store()
    vs_dsm5      = cargar_vector_store_dsm5()
    llm          = cargar_llm()
    rag          = construir_chain(llm, vs_pacientes, vs_dsm5)

    pruebas = [
        (
            "P001",
            "¿Tiene esta paciente algún trastorno del estado de ánimo? "
            "¿Cumple criterios del DSM-5?",
        ),
        (
            "P002",
            "¿Qué medicación toma actualmente y para qué es cada fármaco?",
        ),
    ]

    for patient_id, pregunta in pruebas:
        print(f"{'='*60}")
        print(f"Paciente : {patient_id}")
        print(f"Pregunta : {pregunta}")
        print(f"{'='*60}")
        respuesta = rag(patient_id, pregunta)
        print(respuesta)
        print()
