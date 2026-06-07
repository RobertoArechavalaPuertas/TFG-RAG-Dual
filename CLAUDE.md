# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the system

Ollama must be running before launching anything else:

```bash
ollama serve                  # in a separate terminal — must stay alive
python3 src/main.py           # interactive CLI (from project root)
```

All commands must be run from the project root with the venv active:

```bash
source venv/bin/activate
```

## Re-indexing

`ingest.py` and `ingest_dsm5.py` are both idempotent — they skip already-indexed content. Only needed when PDFs change or ChromaDB is deleted:

```bash
python3 src/ingest.py         # indexes datos/pacientes/ → collection "historiales_clinicos"
python3 src/ingest_dsm5.py    # indexes datos/DSM5/manualDSM5.pdf → collection "dsm5_guia"
```

## Robustness verification

```bash
python3 src/verificacion.py   # 27-check battery: retrieval, threshold, isolation, metadata
```

## Quick module tests

Each module has a `__main__` block for standalone testing:

```bash
python3 src/retriever.py      # tests threshold + MMR retrieval for P001, P002, P003
python3 src/retriever_dsm5.py # tests threshold + MMR search against DSM-5
python3 src/rag_chain.py      # tests full dual-RAG pipeline for P001, P002
```

## Architecture

The system is a **dual RAG pipeline** where every query pulls context from two independent ChromaDB collections before making a single LLM call:

```
User query (patient_id + question)
        │
        ├─► retriever.py       → threshold check → MMR(k=8, fetch_k=20, λ=0.8)
        │   collection: "historiales_clinicos"   ← patient records (section-chunked)
        │
        └─► retriever_dsm5.py  → threshold check → MMR(k=4, fetch_k=20, λ=0.6)
            collection: "dsm5_guia"              ← DSM-5 diagnostic criteria
                    │
                    ▼
            rag_chain.py — single prompt with both contexts → OllamaLLM
```

**Patient isolation** is enforced at the ChromaDB metadata filter level (`{"patient_id": "P00X"}`), not in application logic. The whitelist in `main.py` is a second, independent check: an unknown ID never reaches the LLM.

**Relevance threshold** — both retrievers check the score of the top-1 result before running MMR. Queries scoring below `MIN_SCORE` (off-topic or out-of-domain) are rejected before reaching the LLM. Thresholds are calibrated empirically: `-17.0` for patient records, `-8.0` for DSM-5.

**Chunking differs by source:**
- Patient records: split by the 7 clinical sections of each PDF (`datos_paciente`, `antecedentes`, `historia_actual`, `exploracion`, `tratamiento`, `laboratorio`, `plan`) using regex detection. Each chunk carries a `seccion` metadata field. 500-char chunks with overlap 50 applied within each section.
- DSM-5: split by disorder (132 sections detected via DSM codes). Each chunk is prefixed with `[disorder name]` so embeddings capture the disorder context even in standalone criterion chunks (A., B., C.…). 1200-char chunks, overlap 150.

**Retrieval strategy differs by source** — both use MMR, but with different `lambda_mult`: `0.8` for patient records (relevance priority), `0.6` for DSM-5 (more diversity to avoid near-duplicate criteria from related disorders).

**DSM-5 preprocessing** — 22 chapter-level running headers (e.g. "Trastornos depresivos") removed from page text before chunking. Pages 0–68 (CIE index) and 482–491 (alphabetical index) excluded.

## Configuration

All runtime paths and model settings live in `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:7b` | LLM served by Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `CHROMA_DB_PATH` | `./chroma_db` | Shared path for both collections |
| `PACIENTES_PATH` | `./datos/pacientes` | Patient PDF directory |
| `DSM5_PATH` | `./datos/DSM5/manualDSM5.pdf` | DSM-5 PDF |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `500` / `50` | Patient chunking (DSM-5 hardcodes its own) |

## Current patients (P001–P006)

| ID | Name | Main diagnosis |
|---|---|---|
| P001 | Carmen Ruiz Velasco | Depresión mayor + TAG |
| P002 | Antonio Herrera Lopez | Esquizofrenia paranoide |
| P003 | Maria Jose Martinez Aguilar | Trastorno bipolar tipo I |
| P004 | Alejandro Vega Romero | — |
| P005 | Laura Navarro Gutierrez | — |
| P006 | Marta Esteve Climent | — |

## Adding a new patient

1. Add the PDF to `datos/pacientes/` — filename must start with `PXXX_` (e.g. `P007_Nombre_Apellido.pdf`).
2. Run `python3 src/ingest.py` — only the new patient will be indexed (incremental).
3. Add the patient to `PACIENTES_VALIDOS` in `src/main.py`.

## Note on real clinical records

The current chunking strategy (`segmentar_por_secciones` in `ingest.py`) relies on the 7 numbered sections of the synthetic PDF format. If real EHR exports (Abucasis, SAP, etc.) arrive, the section regex and possibly the chunking strategy will need revision before re-indexing.
