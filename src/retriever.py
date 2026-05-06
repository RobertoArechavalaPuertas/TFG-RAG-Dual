import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# ── Configuración ─────────────────────────────────────────────────────────────
load_dotenv()

CHROMA_DB_PATH  = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
NUM_RESULTADOS  = 8  # chunks a recuperar por consulta


# ── Cargar vector store (una sola vez al importar) ────────────────────────────
def cargar_vector_store() -> Chroma:
    """Conecta con la base de datos ChromaDB ya indexada."""
    embedding_fn = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory = CHROMA_DB_PATH,
        embedding_function = embedding_fn,
        collection_name    = "historiales_clinicos",
    )


# ── Recuperador principal ─────────────────────────────────────────────────────
def recuperar_contexto(patient_id: str, pregunta: str, vector_store: Chroma) -> str:
    """Busca los chunks más relevantes para un paciente y una pregunta.

    Args:
        patient_id:   Código del paciente, p.ej. 'P001'
        pregunta:     Consulta en lenguaje natural
        vector_store: Instancia de ChromaDB ya cargada

    Returns:
        String con los chunks más relevantes concatenados,
        listos para usarse como contexto en el prompt.
    """
    # Filtro por patient_id — ChromaDB solo busca entre los chunks de este paciente
    filtro = {"patient_id": patient_id}

    resultados = vector_store.similarity_search(
        query  = pregunta,
        k      = NUM_RESULTADOS,
        filter = filtro,
    )

    if not resultados:
        return f"No se encontró información para el paciente {patient_id}."

    # Unir los chunks recuperados en un solo bloque de texto
    contexto = "\n\n---\n\n".join([doc.page_content for doc in resultados])
    return contexto


# ── Prueba rápida (solo se ejecuta con: python3 src/retriever.py) ─────────────
if __name__ == "__main__":
    print("Cargando vector store...")
    vs = cargar_vector_store()
    print("Listo.\n")

    # Casos de prueba para verificar el aislamiento por paciente
    pruebas = [
        ("P001", "¿Qué medicación toma este paciente?"),
        ("P002", "¿Cuál es el problema cardíaco del paciente?"),
        ("P003", "¿Cuál es el estado de la función renal?"),
    ]

    for patient_id, pregunta in pruebas:
        print(f"{'='*55}")
        print(f"Paciente: {patient_id}")
        print(f"Pregunta: {pregunta}")
        print(f"{'='*55}")
        contexto = recuperar_contexto(patient_id, pregunta, vs)
        print(contexto)
        print()
