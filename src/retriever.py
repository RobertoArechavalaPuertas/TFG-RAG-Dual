import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# ── Configuración ─────────────────────────────────────────────────────────────
load_dotenv()

CHROMA_DB_PATH  = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

NUM_RESULTADOS = 8   # chunks que llegan al LLM
FETCH_K        = 20  # candidatos evaluados internamente por MMR
LAMBDA_MULT    = 0.8  # más peso a relevancia que a diversidad (vs 0.6 en DSM-5)

# Umbral de relevancia mínima (score LangChain con distancia L2).
# Empíricamente: preguntas relevantes puntúan entre -7 y -15;
# preguntas completamente fuera de ámbito puntúan por debajo de -20.
MIN_SCORE = -17.0


# ── Cargar vector store (una sola vez al importar) ────────────────────────────
def cargar_vector_store() -> Chroma:
    """Conecta con la base de datos ChromaDB ya indexada."""
    embedding_fn = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory  = CHROMA_DB_PATH,
        embedding_function = embedding_fn,
        collection_name    = "historiales_clinicos",
    )


# ── Recuperador principal ─────────────────────────────────────────────────────
def recuperar_contexto(patient_id: str, pregunta: str, vector_store: Chroma) -> str:
    """Busca los chunks más relevantes para un paciente y una pregunta.

    Pipeline: threshold de relevancia (descarta ruido) + MMR (diversidad de chunks).

    Returns:
        Chunks seleccionados concatenados, listos para el prompt.
    """
    filtro = {"patient_id": patient_id}

    top1 = vector_store.similarity_search_with_relevance_scores(
        query=pregunta, k=1, filter=filtro
    )
    if not top1 or top1[0][1] < MIN_SCORE:
        return f"No se encontró información relevante para el paciente {patient_id}."

    resultados = vector_store.max_marginal_relevance_search(
        query       = pregunta,
        k           = NUM_RESULTADOS,
        fetch_k     = FETCH_K,
        lambda_mult = LAMBDA_MULT,
        filter      = filtro,
    )

    contexto = "\n\n---\n\n".join([doc.page_content for doc in resultados])
    return contexto


# ── Prueba rápida (solo se ejecuta con: python3 src/retriever.py) ─────────────
if __name__ == "__main__":
    print("Cargando vector store...")
    vs = cargar_vector_store()
    print("Listo.\n")

    pruebas = [
        ("P001", "¿Qué medicación toma actualmente?"),
        ("P002", "¿Cuántos episodios psicóticos ha tenido?"),
        ("P003", "¿Cuál es el diagnóstico principal?"),
        # Caso fuera de ámbito: debe devolver "No se encontró información relevante"
        ("P001", "¿Cuánto mide el puente de Brooklyn?"),
    ]

    for patient_id, pregunta in pruebas:
        print(f"{'='*55}")
        print(f"Paciente: {patient_id} | Pregunta: {pregunta}")
        print(f"{'='*55}")
        resultado = recuperar_contexto(patient_id, pregunta, vs)
        # Solo mostrar los primeros 400 chars para no saturar la salida
        print(resultado[:400])
        print()
