# Sistema RAG Dual para Consulta de Historiales Clínicos

Trabajo de Fin de Grado — Ingeniería Informática  

---

## ¿Qué es este sistema?

Asistente de consulta clínica basado en **RAG dual** (Retrieval-Augmented Generation): ante una pregunta sobre un paciente, el sistema recupera contexto de dos fuentes independientes y genera una única respuesta estructurada mediante un modelo de lenguaje local.

- **RAG 1** — Historial clínico del paciente (colección ChromaDB `historiales_clinicos`)
- **RAG 2** — Criterios diagnósticos del DSM-5 (colección ChromaDB `dsm5_guia`)

El modelo de lenguaje (Qwen 2.5 7B via Ollama) nunca recibe una pregunta sin contexto recuperado. Si la consulta no supera el umbral de relevancia, el sistema la rechaza antes de llegar al LLM.

---

## Arquitectura

```
Consulta (patient_id + pregunta)
        │
        ├─► retriever.py       → umbral de relevancia → MMR(k=8, λ=0.8)
        │   colección: historiales_clinicos   ← PDFs de pacientes (7 secciones)
        │
        └─► retriever_dsm5.py  → umbral de relevancia → MMR(k=4, λ=0.6)
            colección: dsm5_guia              ← manual DSM-5
                    │
                    ▼
            rag_chain.py — prompt único con ambos contextos → OllamaLLM → respuesta
```

**Aislamiento de paciente**: ChromaDB filtra por `patient_id` a nivel de metadatos. Una consulta sobre P001 nunca puede recuperar chunks de P002, independientemente de la pregunta.

**Umbral de relevancia**: el retriever comprueba el score del chunk más cercano antes de ejecutar MMR. Preguntas fuera de ámbito (geografía, matemáticas, etc.) son rechazadas sin llegar al LLM.

---

## Requisitos previos

- Python 3.9 o superior
- [Ollama](https://ollama.com/) instalado y en ejecución
- Modelo `qwen2.5:7b` descargado en Ollama:

```bash
ollama pull qwen2.5:7b
```

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd TFG_Informatica

# 2. Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Indexación (primera vez)

Antes de lanzar el sistema hay que indexar los documentos en ChromaDB. Este paso es idempotente: si ya están indexados, no hace nada.

```bash
# Indexar historiales clínicos (datos/pacientes/*.pdf)
python3 src/ingest.py

# Indexar manual DSM-5 (datos/DSM5/manualDSM5.pdf)
python3 src/ingest_dsm5.py
```

La indexación tarda varios minutos la primera vez (descarga del modelo de embeddings incluida). Las siguientes ejecuciones son inmediatas.

---

## Ejecución

Ollama debe estar corriendo en una terminal separada:

```bash
ollama serve
```

En otra terminal, con el entorno virtual activo:

```bash
python3 src/main.py
```

### Flujo de uso

```
¿Ejecutar prueba de aislamiento antes de empezar? (s/n): n

ID del paciente (P001-P010): P001
Pregunta para P001: ¿Qué medicación toma actualmente y para qué es cada fármaco?

Buscando en historial y DSM-5, consultando al LLM...

───────────────────────────────────────────────────
  Paciente : P001 — Carmen Ruiz Velasco
  Pregunta : ¿Qué medicación toma actualmente...?
───────────────────────────────────────────────────

  [Respuesta del sistema]
```

Comandos especiales durante la sesión:

| Comando | Acción |
|---|---|
| `salir` | Cierra el sistema |
| `pacientes` | Muestra la lista de pacientes disponibles |

---

## Pacientes del sistema

| ID | Nombre | Diagnóstico principal |
|---|---|---|
| P001 | Carmen Ruiz Velasco | Depresión mayor + TAG |
| P002 | Antonio Herrera Lopez | Esquizofrenia paranoide |
| P003 | Maria Jose Martinez Aguilar | Trastorno bipolar tipo I |
| P004 | Alejandro Vega Romero | — |
| P005 | Laura Navarro Gutierrez | — |
| P006 | Marta Esteve Climent | — |

---

## Verificación de robustez

La batería de 27 comprobaciones automatizadas cubre recuperación, umbral de relevancia, aislamiento entre pacientes y calidad de los metadatos:

```bash
python3 src/verificacion.py
```

Salida esperada:

```
══════════════════════════════════════════════
  RESULTADO: 26 OK  |  1 FAIL  |  27 total
══════════════════════════════════════════════
```

---

## Tests de módulo

Cada módulo tiene un bloque `__main__` para prueba aislada:

```bash
python3 src/retriever.py      # recuperación RAG 1: umbral + MMR para P001, P002, P003
python3 src/retriever_dsm5.py # recuperación RAG 2: umbral + MMR contra DSM-5
python3 src/rag_chain.py      # pipeline completo para P001 y P002
```

---

## Configuración

Variables disponibles en `.env` (se crean con los valores por defecto si no existe el archivo):

| Variable | Valor por defecto | Descripción |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:7b` | Modelo LLM en Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Endpoint de Ollama |
| `CHROMA_DB_PATH` | `./chroma_db` | Directorio de la base de datos vectorial |
| `PACIENTES_PATH` | `./datos/pacientes` | Carpeta con los PDFs de pacientes |
| `DSM5_PATH` | `./datos/DSM5/manualDSM5.pdf` | Ruta al manual DSM-5 |
| `CHUNK_SIZE` | `500` | Tamaño de chunk para historiales |
| `CHUNK_OVERLAP` | `50` | Solapamiento entre chunks de historiales |

---

## Estructura del proyecto

```
TFG_Informatica/
├── datos/
│   ├── pacientes/          # PDFs de historiales clínicos (P001–P006)
│   └── DSM5/               # Manual DSM-5 en PDF
├── src/
│   ├── main.py             # Interfaz CLI principal
│   ├── rag_chain.py        # Pipeline RAG dual + prompt al LLM
│   ├── retriever.py        # Recuperador RAG 1 (historiales)
│   ├── retriever_dsm5.py   # Recuperador RAG 2 (DSM-5)
│   ├── ingest.py           # Indexación incremental de pacientes
│   ├── ingest_dsm5.py      # Indexación del DSM-5
│   └── verificacion.py     # Batería de 27 pruebas de robustez
├── chroma_db/              # Base de datos vectorial (generada al indexar)
├── requirements.txt
└── .env                    # Configuración local (no versionado)
```
