import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# ── Cargar configuración desde .env ──────────────────────────────────────────
load_dotenv()

PACIENTES_PATH = Path(os.getenv("PACIENTES_PATH", "./datos/pacientes"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHUNK_SIZE     = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP", 50))

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Secciones estándar de los historiales clínicos, en orden de aparición.
# El patrón es flexible con acentos porque PyMuPDF a veces no los preserva.
SECCIONES = [
    ("datos_paciente",  r"1\.\s+Datos del paciente"),
    ("antecedentes",    r"2\.\s+Antecedentes personales"),
    ("historia_actual", r"3\.\s+Historia cl[ií]nica actual"),
    ("exploracion",     r"4\.\s+Exploraci[oó]n"),
    ("tratamiento",     r"5\.\s+Tratamiento actual"),
    ("laboratorio",     r"6\.\s+Resultados de laboratorio"),
    ("plan",            r"7\.\s+Plan de actuaci[oó]n"),
]


# ── 1. LOADER — extrae texto de un PDF ───────────────────────────────────────
def cargar_pdf(ruta_pdf: Path) -> str:
    """Abre un PDF y devuelve todo su texto como string."""
    texto = ""
    with fitz.open(ruta_pdf) as doc:
        for pagina in doc:
            texto += pagina.get_text()
    return texto


def extraer_patient_id(nombre_archivo: str) -> str:
    """Extrae el patient_id del nombre del archivo.

    Ejemplo: 'P001_Carmen_Ruiz_Velasco.pdf' → 'P001'
    """
    return nombre_archivo.split("_")[0]


# ── 2. SEGMENTACIÓN POR SECCIONES ────────────────────────────────────────────
def segmentar_por_secciones(texto: str) -> list[tuple[str, str]]:
    """Divide el texto del historial en segmentos por sección clínica.

    Returns:
        Lista de tuplas (nombre_seccion, texto_seccion). Las secciones que
        no se encuentran en el PDF se omiten silenciosamente.
    """
    posiciones = []
    for nombre, patron in SECCIONES:
        match = re.search(patron, texto, re.IGNORECASE)
        if match:
            posiciones.append((match.start(), nombre))

    posiciones.sort(key=lambda x: x[0])

    segmentos = []
    for i, (inicio, nombre) in enumerate(posiciones):
        fin = posiciones[i + 1][0] if i + 1 < len(posiciones) else len(texto)
        contenido = texto[inicio:fin].strip()
        if contenido:
            segmentos.append((nombre, contenido))

    return segmentos


# ── 3. SPLITTER — trocea el texto en chunks ───────────────────────────────────
def trocear_seccion(texto: str) -> list[str]:
    """Divide el texto de una sección en chunks con solapamiento."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_text(texto)


# ── 4. HELPERS DE INDEXACIÓN INCREMENTAL ─────────────────────────────────────
def conectar_vector_store(embedding_fn: SentenceTransformerEmbeddings) -> Chroma:
    """Conecta con la colección existente o la crea si no existe."""
    return Chroma(
        persist_directory  = CHROMA_DB_PATH,
        embedding_function = embedding_fn,
        collection_name    = "historiales_clinicos",
    )


def obtener_pacientes_indexados(vector_store: Chroma) -> set[str]:
    """Devuelve el conjunto de patient_ids ya presentes en ChromaDB."""
    result = vector_store._collection.get(include=["metadatas"])
    return {
        meta["patient_id"]
        for meta in result["metadatas"]
        if meta and "patient_id" in meta
    }


# ── 5. INDEXACIÓN INCREMENTAL ─────────────────────────────────────────────────
def indexar_pacientes():
    """Indexa solo los pacientes que no están aún en ChromaDB. Idempotente."""
    if not PACIENTES_PATH.exists():
        print(f"ERROR: No se encontró la carpeta {PACIENTES_PATH}")
        sys.exit(1)

    pdfs = list(PACIENTES_PATH.glob("*.pdf"))
    if not pdfs:
        print(f"ERROR: No hay PDFs en {PACIENTES_PATH}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  Indexación incremental — {len(pdfs)} PDFs encontrados")
    print(f"{'='*50}\n")

    print("Cargando modelo de embeddings...")
    embedding_fn = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store  = conectar_vector_store(embedding_fn)
    ya_indexados  = obtener_pacientes_indexados(vector_store)

    if ya_indexados:
        print(f"Pacientes ya en BD: {sorted(ya_indexados)}\n")

    nuevos       = 0
    omitidos     = 0
    total_chunks = 0

    for ruta_pdf in sorted(pdfs):
        nombre     = ruta_pdf.name
        patient_id = extraer_patient_id(nombre)

        if patient_id in ya_indexados:
            print(f"[{patient_id}] Ya indexado — omitiendo.")
            omitidos += 1
            continue

        print(f"[{patient_id}] Procesando: {nombre}")

        texto     = cargar_pdf(ruta_pdf)
        print(f"  → Texto extraído: {len(texto):,} caracteres")

        segmentos = segmentar_por_secciones(texto)
        print(f"  → Secciones detectadas: {[s for s, _ in segmentos]}")

        textos_batch    = []
        metadatos_batch = []
        ids_batch       = []

        for nombre_seccion, contenido_seccion in segmentos:
            chunks = trocear_seccion(contenido_seccion)
            for i, chunk in enumerate(chunks):
                textos_batch.append(chunk)
                metadatos_batch.append({
                    "patient_id":     patient_id,
                    "nombre_archivo": nombre,
                    "seccion":        nombre_seccion,
                    "chunk_index":    i,
                    "total_chunks":   len(chunks),
                })
                ids_batch.append(f"{patient_id}_{nombre_seccion}_{i:04d}")

        vector_store.add_texts(
            texts=textos_batch,
            metadatas=metadatos_batch,
            ids=ids_batch,
        )

        print(f"  ✓ {patient_id} indexado ({len(ids_batch)} chunks en {len(segmentos)} secciones)\n")
        nuevos       += 1
        total_chunks += len(ids_batch)

    print(f"{'='*50}")
    print(f"  ✓ Indexación completada")
    print(f"  Pacientes nuevos:   {nuevos}")
    print(f"  Pacientes omitidos: {omitidos}")
    print(f"  Chunks añadidos:    {total_chunks}")
    print(f"  Base de datos:      {CHROMA_DB_PATH}")
    print(f"{'='*50}\n")

    return vector_store


# ── Punto de entrada ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    indexar_pacientes()
