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

DSM5_PATH      = Path(os.getenv("DSM5_PATH", "./datos/DSM5/guia-de-consulta-del-dsm-v.pdf"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Las primeras 69 páginas son índice de clasificación CIE + prefacio:
# no aportan criterios diagnósticos recuperables.
# Los criterios reales empiezan en la página 70 (índice 69).
PAGINA_INICIO = 69

# Chunks más grandes que en el RAG de pacientes para preservar
# los criterios diagnósticos A/B/C/D completos dentro del mismo chunk.
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 150


# ── 1. LOADER — extrae texto saltando el índice de clasificación ──────────────
def cargar_dsm5(ruta_pdf: Path) -> tuple[str, int]:
    """Extrae texto del DSM-5 omitiendo las páginas de índice CIE."""
    texto = ""
    with fitz.open(ruta_pdf) as doc:
        total_paginas = len(doc)
        for i in range(PAGINA_INICIO, total_paginas):
            texto += doc[i].get_text()
    return texto, total_paginas


# ── 2. LIMPIEZA — elimina artefactos del PDF ──────────────────────────────────
def limpiar_texto(texto: str) -> str:
    """Elimina guiones de silabeo, números de página sueltos y saltos excesivos."""
    # Guiones de silabeo del PDF (soft hyphen U+00AD)
    texto = texto.replace("\xad", "")
    # Números de página solos al inicio de línea (p. ej. "17\n", "123\n")
    texto = re.sub(r"^\d{1,3}\n", "", texto, flags=re.MULTILINE)
    # Colapsar más de dos saltos de línea consecutivos
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


# ── 3. SPLITTER — trocea respetando la estructura del DSM-5 ──────────────────
def trocear_dsm5(texto: str) -> list[str]:
    """Divide el texto con separadores alineados a la jerarquía del DSM-5.

    Prioridad de corte:
      1. Criterios diagnósticos mayores (A., B., C. …)
      2. Sub-criterios numerados (1., 2., 3. …)
      3. Párrafos (\n\n), líneas, oraciones, palabras.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\nA.\n", "\nB.\n", "\nC.\n", "\nD.\n", "\nE.\n",
            "\n1.\n", "\n2.\n", "\n3.\n", "\n4.\n", "\n5.\n",
            "\n\n",
            "\n",
            ". ",
            " ",
        ],
    )
    return splitter.split_text(texto)


# ── 4. EMBEDDINGS + VECTOR STORE — indexa en ChromaDB ────────────────────────
def indexar_dsm5():
    """Proceso completo: carga el PDF del DSM-5 y lo indexa en ChromaDB."""

    if not DSM5_PATH.exists():
        print(f"ERROR: No se encontró el archivo {DSM5_PATH}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  Indexando DSM-5: Guía de consulta")
    print(f"{'='*50}\n")

    # Paso 1 — Loader
    print(f"Cargando PDF: {DSM5_PATH.name}")
    texto_crudo, total_paginas = cargar_dsm5(DSM5_PATH)
    paginas_procesadas = total_paginas - PAGINA_INICIO
    print(f"  → Páginas omitidas (índice CIE): {PAGINA_INICIO}")
    print(f"  → Páginas procesadas: {paginas_procesadas}/{total_paginas}")
    print(f"  → Texto extraído: {len(texto_crudo):,} caracteres")

    # Paso 2 — Limpieza
    texto = limpiar_texto(texto_crudo)
    print(f"  → Texto tras limpieza: {len(texto):,} caracteres")

    # Paso 3 — Splitter
    chunks = trocear_dsm5(texto)
    longitudes = [len(c) for c in chunks]
    print(f"  → Chunks generados: {len(chunks)}")
    print(f"  → Longitud media: {sum(longitudes) // len(longitudes)} chars")
    print(f"  → Longitud mín/máx: {min(longitudes)}/{max(longitudes)} chars")

    # Paso 4 — Embeddings + inserción en ChromaDB
    print("\nCargando modelo de embeddings...")
    print("(La primera vez descarga ~90 MB, ten paciencia)\n")
    embedding_fn = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

    metadatos = [
        {
            "source":       "dsm5",
            "chunk_index":  i,
            "total_chunks": len(chunks),
        }
        for i in range(len(chunks))
    ]

    print("Indexando en ChromaDB (colección 'dsm5_guia')...")
    vector_store = Chroma.from_texts(
        texts             = chunks,
        metadatas         = metadatos,
        embedding         = embedding_fn,
        persist_directory = CHROMA_DB_PATH,
        collection_name   = "dsm5_guia",
    )

    print(f"\n{'='*50}")
    print(f"  ✓ Indexación DSM-5 completada")
    print(f"  Total chunks indexados: {len(chunks)}")
    print(f"  Colección: dsm5_guia")
    print(f"  Base de datos: {CHROMA_DB_PATH}")
    print(f"{'='*50}\n")

    return vector_store


# ── Punto de entrada ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    indexar_dsm5()
