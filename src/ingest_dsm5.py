import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Configuración
load_dotenv()

DSM5_PATH      = Path(os.getenv("DSM5_PATH", "./datos/DSM5/manualDSM5.pdf"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Las primeras 69 páginas son índice de clasificación CIE + prefacio:
# no aportan criterios diagnósticos recuperables.
# Los criterios reales empiezan en la página 70 (índice 69).
PAGINA_INICIO = 69

# Las páginas 482-491 son el índice alfabético del libro (entradas tipo
# "trastorno depresivo persistente (distimia), 108–110"). Indexarlas
# introduce chunks completamente inútiles que contaminan los resultados.
PAGINA_FIN = 482

# Chunks más grandes que en el RAG de pacientes para preservar
# los criterios diagnósticos A/B/C/D completos dentro del mismo chunk.
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 150

# Código DSM-5: línea corta con formato "NNN.N (FXXX)".
# Solo se considera código de trastorno cuando ocupa la mayor parte de la
# línea (< 30 chars); los códigos inline en frases descriptivas se ignoran.
_DSM_CODE_RE = re.compile(r"^\s*\d{2,3}\.\d+\s*\([A-Z]\d+[\w\.]*\)\s*$")

# Número mínimo de páginas en las que una línea debe aparecer como primera
# línea para considerarse running header de capítulo.
_RUNNING_HEADER_MIN_FREQ = 4


# 1. Loader
def cargar_dsm5_paginas(ruta_pdf: Path) -> tuple[list[str], int]:
    paginas = []
    with fitz.open(ruta_pdf) as doc:
        total_paginas = len(doc)
        for i in range(PAGINA_INICIO, PAGINA_FIN):
            paginas.append(doc[i].get_text())
    return paginas, total_paginas


# 2. Limpieza de running headers
def limpiar_running_headers(paginas: list[str]) -> tuple[list[str], set[str]]:
    from collections import Counter

    frecuencias: Counter = Counter()
    for texto in paginas:
        lineas = [l.strip() for l in texto.split("\n")
                  if l.strip() and not l.strip().isdigit()]
        if lineas and 5 < len(lineas[0]) < 75:
            frecuencias[lineas[0]] += 1

    running_headers = {
        linea for linea, n in frecuencias.items()
        if n >= _RUNNING_HEADER_MIN_FREQ
    }

    paginas_limpias = []
    for texto in paginas:
        lineas_filtradas = [
            l for l in texto.split("\n")
            if l.strip() not in running_headers
        ]
        paginas_limpias.append("\n".join(lineas_filtradas))

    return paginas_limpias, running_headers


# 3. Detección de trastorno en página
def detectar_trastorno_en_pagina(lineas: list[str], trastorno_previo: str) -> str:
    for i, linea in enumerate(lineas):
        if not _DSM_CODE_RE.match(linea):
            continue

        # Buscar la línea anterior no vacía y no numérica
        candidatas = []
        for j in range(i - 1, max(-1, i - 5), -1):
            c = lineas[j].strip().rstrip(".")
            if not c or c.isdigit():
                continue
            candidatas.append(c)
            if len(candidatas) == 2:
                break

        if not candidatas:
            return trastorno_previo

        nombre = candidatas[0]
        # Si la línea más cercana es un fragmento de frase (minúscula),
        # usar la anterior como nombre del trastorno
        if nombre[0].islower() and len(candidatas) > 1:
            nombre = candidatas[1]

        # Rechazar si sigue siendo minúscula, empieza por paréntesis
        # (fragmento de nombre largo) o tiene menos de 2 palabras
        if nombre[0].islower() or nombre.startswith("(") or len(nombre.split()) < 2:
            return trastorno_previo

        return nombre

    return trastorno_previo


# 4. Segmentación por trastorno
def segmentar_por_trastorno(paginas: list[str]) -> list[tuple[str, str]]:
    trastorno_actual = "General"
    texto_acumulado  = ""
    secciones: list[tuple[str, str]] = []

    for texto_pagina in paginas:
        lineas = texto_pagina.split("\n")
        nuevo_trastorno = detectar_trastorno_en_pagina(lineas, trastorno_actual)

        if nuevo_trastorno != trastorno_actual:
            if texto_acumulado.strip():
                secciones.append((texto_acumulado, trastorno_actual))
            trastorno_actual = nuevo_trastorno
            texto_acumulado  = texto_pagina
        else:
            texto_acumulado += "\n" + texto_pagina

    if texto_acumulado.strip():
        secciones.append((texto_acumulado, trastorno_actual))

    return secciones


# 5. Limpieza de artefactos del PDF
def limpiar_texto(texto: str) -> str:
    # Guiones de silabeo del PDF (soft hyphen U+00AD)
    texto = texto.replace("\xad", "")
    # Números de página solos al inicio de línea (p. ej. "17\n", "123\n")
    texto = re.sub(r"^\d{1,3}\n", "", texto, flags=re.MULTILINE)
    # Colapsar más de dos saltos de línea consecutivos
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


# 6. Splitter
def trocear_dsm5(texto: str) -> list[str]:
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


# 7. Helpers de indexación idempotente
def _slugify(texto: str) -> str:
    slug = re.sub(r"[^\w\s]", "", texto.lower())
    slug = re.sub(r"\s+", "_", slug.strip())
    return slug[:40]  # limitar longitud


def _conectar_vector_store(embedding_fn: SentenceTransformerEmbeddings) -> Chroma:
    return Chroma(
        persist_directory = CHROMA_DB_PATH,
        embedding_function = embedding_fn,
        collection_name    = "dsm5_guia",
    )


def _ya_indexado(vector_store: Chroma) -> bool:
    return vector_store._collection.count() > 0


# 8. Indexación
def indexar_dsm5():
    if not DSM5_PATH.exists():
        print(f"ERROR: No se encontró el archivo {DSM5_PATH}")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  Indexando DSM-5: Guía de consulta")
    print(f"{'='*50}\n")

    print("Cargando modelo de embeddings...")
    embedding_fn = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = _conectar_vector_store(embedding_fn)

    if _ya_indexado(vector_store):
        total = vector_store._collection.count()
        print(f"  DSM-5 ya indexado ({total} chunks). Nada que hacer.\n")
        return vector_store

    print(f"Cargando PDF: {DSM5_PATH.name}")
    paginas, total_paginas = cargar_dsm5_paginas(DSM5_PATH)
    paginas_procesadas    = PAGINA_FIN - PAGINA_INICIO
    print(f"  → Páginas omitidas al inicio (índice CIE): {PAGINA_INICIO}")
    print(f"  → Páginas omitidas al final (índice alfabético): {total_paginas - PAGINA_FIN}")
    print(f"  → Páginas procesadas: {paginas_procesadas}/{total_paginas}")

    paginas, headers_eliminados = limpiar_running_headers(paginas)
    print(f"  → Running headers eliminados: {len(headers_eliminados)}")
    for h in sorted(headers_eliminados):
        print(f"      · {h}")

    secciones = segmentar_por_trastorno(paginas)
    print(f"  → Secciones (trastornos detectados): {len(secciones)}")

    textos_batch    = []
    metadatos_batch = []
    ids_batch       = []

    for sec_idx, (texto_seccion, trastorno) in enumerate(secciones):
        texto_limpio = limpiar_texto(texto_seccion)
        chunks = trocear_dsm5(texto_limpio)
        slug = _slugify(trastorno)
        for i, chunk in enumerate(chunks):
            chunk_prefijado = f"[{trastorno}]\n{chunk}"
            textos_batch.append(chunk_prefijado)
            metadatos_batch.append({
                "source":       "dsm5",
                "trastorno":    trastorno,
                "chunk_index":  i,
                "total_chunks": len(chunks),
            })
            # sec_idx garantiza unicidad aunque dos secciones tengan el mismo slug
            ids_batch.append(f"dsm5_{sec_idx:03d}_{slug}_{i:04d}")

    longitudes = [len(t) for t in textos_batch]
    print(f"  → Chunks generados: {len(textos_batch)}")
    print(f"  → Longitud media: {sum(longitudes) // len(longitudes)} chars")
    print(f"  → Longitud mín/máx: {min(longitudes)}/{max(longitudes)} chars")

    print("\nIndexando en ChromaDB (colección 'dsm5_guia')...")
    vector_store.add_texts(
        texts     = textos_batch,
        metadatas = metadatos_batch,
        ids       = ids_batch,
    )

    print(f"\n{'='*50}")
    print(f"  ✓ Indexación DSM-5 completada")
    print(f"  Total chunks indexados: {len(textos_batch)}")
    print(f"  Colección: dsm5_guia")
    print(f"  Base de datos: {CHROMA_DB_PATH}")
    print(f"{'='*50}\n")

    return vector_store


# Punto de entrada
if __name__ == "__main__":
    indexar_dsm5()
