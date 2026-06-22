import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Configuración
load_dotenv()

CHROMA_DB_PATH  = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# MMR: k chunks devueltos al LLM, fetch_k candidatos evaluados internamente.
# lambda_mult=0.6 → prioriza relevancia pero evita criterios redundantes
# entre trastornos relacionados (p. ej. varios trastornos depresivos).
NUM_RESULTADOS = 4
FETCH_K        = 20
LAMBDA_MULT    = 0.6

# Umbral de relevancia mínima (score LangChain con distancia L2).
# Empíricamente: preguntas sobre trastornos DSM puntúan entre -1 y -4;
# preguntas sobre medicación/analíticas del paciente (sin relación con el
# DSM) caen por debajo de -8. El umbral evita enviar contexto DSM inútil
# al LLM cuando la pregunta es puramente clínica (dosis, lab, seguimiento).
MIN_SCORE = -8.0


# Cargar vector store
def cargar_vector_store_dsm5() -> Chroma:
    embedding_fn = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory  = CHROMA_DB_PATH,
        embedding_function = embedding_fn,
        collection_name    = "dsm5_guia",
    )


# Recuperador principal
def recuperar_contexto_dsm5(pregunta: str, vector_store: Chroma) -> str:
    top1 = vector_store.similarity_search_with_relevance_scores(
        query=pregunta, k=1
    )
    if not top1 or top1[0][1] < MIN_SCORE:
        return ""

    resultados = vector_store.max_marginal_relevance_search(
        query       = pregunta,
        k           = NUM_RESULTADOS,
        fetch_k     = FETCH_K,
        lambda_mult = LAMBDA_MULT,
    )

    return "\n\n---\n\n".join([doc.page_content for doc in resultados])


# Prueba rápida (python3 src/retriever_dsm5.py)
if __name__ == "__main__":
    print("Cargando vector store DSM-5...")
    vs = cargar_vector_store_dsm5()
    print("Listo.\n")

    pruebas = [
        ("DSM",      "criterios diagnósticos para la depresión mayor"),
        ("DSM",      "síntomas del trastorno de ansiedad generalizada"),
        ("NO-DSM",   "¿Qué dosis de escitalopram toma el paciente?"),
        ("NO-DSM",   "resultados analítica glucosa colesterol"),
    ]

    for tipo, pregunta in pruebas:
        print(f"{'='*55}")
        print(f"[{tipo}] {pregunta}")
        print(f"{'='*55}")
        contexto = recuperar_contexto_dsm5(pregunta, vs)
        if contexto:
            print(contexto[:400])
        else:
            print("→ Sin contexto DSM-5 (threshold no superado)")
        print()
