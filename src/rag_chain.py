import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

from retriever import cargar_vector_store, recuperar_contexto
from retriever_dsm5 import cargar_vector_store_dsm5, recuperar_contexto_dsm5

# ── Configuración ─────────────────────────────────────────────────────────────
load_dotenv()

OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ── Prompt template dual ──────────────────────────────────────────────────────
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


# ── Inicializar LLM ───────────────────────────────────────────────────────────
def cargar_llm() -> OllamaLLM:
    """Conecta con Qwen 2.5 corriendo en Ollama."""
    return OllamaLLM(
        model       = OLLAMA_MODEL,
        base_url    = OLLAMA_BASE_URL,
        temperature = 0.1,
    )


# ── Chain dual ────────────────────────────────────────────────────────────────
def construir_chain(llm, vs_pacientes, vs_dsm5):
    """Devuelve una función que recibe patient_id + pregunta y devuelve respuesta.

    Recupera contexto de ambos vectorstores y los envía en una única
    llamada al LLM, siguiendo el diseño del tutor académico.
    """

    def chain(patient_id: str, pregunta: str) -> str:
        # Paso 1: recuperar contexto del historial clínico (filtrado por patient_id)
        contexto_historial = recuperar_contexto(patient_id, pregunta, vs_pacientes)

        # Paso 2: recuperar contexto del DSM-5 (base global, sin filtro)
        contexto_dsm5 = recuperar_contexto_dsm5(pregunta, vs_dsm5)

        # Paso 3: construir el prompt con ambos contextos
        prompt_final = prompt.format(
            patient_id         = patient_id,
            contexto_historial = contexto_historial,
            contexto_dsm5      = contexto_dsm5,
            pregunta           = pregunta,
        )

        # Paso 4: única llamada al LLM con contexto combinado
        respuesta = llm.invoke(prompt_final)
        return respuesta.strip()

    return chain


# ── Prueba rápida ─────────────────────────────────────────────────────────────
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
