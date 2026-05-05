import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# ── Configuración ─────────────────────────────────────────────────────────────
load_dotenv()

CHROMA_DB_PATH  = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# MMR: k chunks devueltos al LLM, fetch_k candidatos evaluados internamente.
# lambda_mult=0.6 → prioriza relevancia pero evita criterios redundantes
# entre trastornos relacionados (p. ej. varios trastornos depresivos).
NUM_RESULTADOS = 4
FETCH_K        = 20
LAMBDA_MULT    = 0.6


# ── Cargar vector store (una sola vez al importar) ────────────────────────────
def cargar_vector_store_dsm5() -> Chroma:
    """Conecta con la colección DSM-5 de ChromaDB ya indexada."""
    embedding_fn = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory  = CHROMA_DB_PATH,
        embedding_function = embedding_fn,
        collection_name    = "dsm5_guia",
    )


# ── Recuperador principal ─────────────────────────────────────────────────────
def recuperar_contexto_dsm5(pregunta: str, vector_store: Chroma) -> str:
    """Busca los chunks más relevantes del DSM-5 para una pregunta.

    Usa MMR en lugar de similarity_search para evitar recuperar múltiples
    chunks casi idénticos de trastornos relacionados entre sí.

    Args:
        pregunta:     Consulta en lenguaje natural
        vector_store: Instancia de ChromaDB con la colección 'dsm5_guia'

    Returns:
        String con los chunks más relevantes concatenados,
        listos para usarse como contexto en el prompt.
    """
    resultados = vector_store.max_marginal_relevance_search(
        query       = pregunta,
        k           = NUM_RESULTADOS,
        fetch_k     = FETCH_K,
        lambda_mult = LAMBDA_MULT,
    )

    if not resultados:
        return "No se encontró información relevante en el DSM-5."

    contexto = "\n\n---\n\n".join([doc.page_content for doc in resultados])
    return contexto


# ── Prueba rápida (solo se ejecuta con: python3 src/retriever_dsm5.py) ────────
if __name__ == "__main__":
    print("Cargando vector store DSM-5...")
    vs = cargar_vector_store_dsm5()
    print("Listo.\n")

    pruebas = [
        "criterios diagnósticos para la depresión mayor",
        "síntomas del trastorno de ansiedad generalizada",
        "criterios para el diagnóstico de diabetes tipo 2",  # fuera de ámbito
    ]

    for pregunta in pruebas:
        print(f"{'='*55}")
        print(f"Pregunta: {pregunta}")
        print(f"{'='*55}")
        contexto = recuperar_contexto_dsm5(pregunta, vs)
        print(contexto[:600])
        print()
