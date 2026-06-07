"""Verificación de robustez del sistema RAG dual."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from retriever import cargar_vector_store, recuperar_contexto
from retriever_dsm5 import cargar_vector_store_dsm5, recuperar_contexto_dsm5
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

OK   = "✓"
FAIL = "✗"

resultados = []

def check(nombre, condicion, detalle=""):
    estado = OK if condicion else FAIL
    resultados.append((estado, nombre))
    print(f"  {estado}  {nombre}")
    if detalle and not condicion:
        print(f"       → {detalle}")


print("\nCargando vector stores...")
vs_pac  = cargar_vector_store()
vs_dsm5 = cargar_vector_store_dsm5()
print("Listo.\n")

# ══════════════════════════════════════════════════════
print("══════════════════════════════════════════════")
print("  RAG 1 — Historial clínico")
print("══════════════════════════════════════════════")

# [A] Recuperación relevante
print("\n[A] Recuperación relevante")

ctx = recuperar_contexto("P001", "¿Qué medicación toma actualmente?", vs_pac)
check("P001 · medicación → devuelve contexto",
      ctx and "No se encontró" not in ctx)
check("P001 · medicación → contiene escitalopram",
      "scitalopram" in ctx or "Escitalopram" in ctx,
      f"Fragmento: {ctx[:120]}")

ctx = recuperar_contexto("P002", "¿Cuántos episodios psicóticos ha tenido?", vs_pac)
check("P002 · episodios psicóticos → devuelve contexto",
      ctx and "No se encontró" not in ctx)
check("P002 · episodios psicóticos → contiene 'episodio' o 'psicótico'",
      "episodio" in ctx.lower() or "psicot" in ctx.lower(),
      f"Fragmento: {ctx[:120]}")

ctx = recuperar_contexto("P003", "¿Cuál es el diagnóstico principal?", vs_pac)
check("P003 · diagnóstico → devuelve contexto",
      ctx and "No se encontró" not in ctx)
check("P003 · diagnóstico → contiene 'bipolar'",
      "bipolar" in ctx.lower(),
      f"Fragmento: {ctx[:120]}")

ctx = recuperar_contexto("P006", "¿Qué analíticas tiene alteradas?", vs_pac)
check("P006 · analíticas → devuelve contexto",
      ctx and "No se encontró" not in ctx)

# [B] Threshold — consultas fuera de ámbito
print("\n[B] Threshold — consultas irrelevantes")

ctx = recuperar_contexto("P001", "¿Cuánto mide el puente de Brooklyn?", vs_pac)
check("P001 · puente Brooklyn → bloqueado",
      "No se encontró información relevante" in ctx,
      f"Devolvió: {ctx[:80]}")

ctx = recuperar_contexto("P001", "¿Cuál es la capital de Francia?", vs_pac)
check("P001 · capital Francia → bloqueado",
      "No se encontró información relevante" in ctx,
      f"Devolvió: {ctx[:80]}")

# [C] Aislamiento entre pacientes
print("\n[C] Aislamiento entre pacientes")

ctx_p1 = recuperar_contexto("P001", "¿Tiene este paciente esquizofrenia?", vs_pac)
ctx_p2 = recuperar_contexto("P002", "¿Tiene este paciente esquizofrenia?", vs_pac)
check("P001 no contiene 'esquizofrenia' en su contexto",
      "esquizofrenia" not in ctx_p1.lower() or "No se encontró" in ctx_p1,
      f"Fragmento P001: {ctx_p1[:120]}")
check("P002 sí contiene 'esquizofrenia' en su contexto",
      "esquizofrenia" in ctx_p2.lower(),
      f"Fragmento P002: {ctx_p2[:120]}")

ctx_p1b = recuperar_contexto("P001", "¿Ha tenido episodios maníacos?", vs_pac)
ctx_p3  = recuperar_contexto("P003", "¿Ha tenido episodios maníacos?", vs_pac)
check("P001 no contiene 'maníaco' en su contexto",
      "maniaco" not in ctx_p1b.lower() and "maníaco" not in ctx_p1b.lower(),
      f"Fragmento P001: {ctx_p1b[:120]}")
check("P003 sí contiene 'maníaco' o 'maniaco' en su contexto",
      "maniaco" in ctx_p3.lower() or "maníaco" in ctx_p3.lower(),
      f"Fragmento P003: {ctx_p3[:120]}")

# [D] Metadato 'seccion' en todos los chunks
print("\n[D] Metadato 'seccion' en chunks de pacientes")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
emb = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

raw_pac = Chroma(persist_directory=CHROMA_DB_PATH,
                 embedding_function=emb, collection_name="historiales_clinicos")
metas = raw_pac._collection.get(include=["metadatas"])["metadatas"]
con_seccion = sum(1 for m in metas if m.get("seccion"))
check(f"Todos los chunks tienen 'seccion' ({con_seccion}/{len(metas)})",
      con_seccion == len(metas))

secciones_presentes = {m.get("seccion") for m in metas if m.get("seccion")}
esperadas = {"datos_paciente", "antecedentes", "historia_actual",
             "exploracion", "tratamiento", "laboratorio", "plan"}
check(f"Las 7 secciones clínicas están presentes ({secciones_presentes})",
      esperadas == secciones_presentes)

# ══════════════════════════════════════════════════════
print("\n══════════════════════════════════════════════")
print("  RAG 2 — DSM-5")
print("══════════════════════════════════════════════")

# [E] Recuperación DSM relevante
print("\n[E] Recuperación relevante")

ctx = recuperar_contexto_dsm5("criterios diagnósticos trastorno depresión mayor", vs_dsm5)
check("DSM · depresión mayor → devuelve contexto", bool(ctx))
check("DSM · depresión mayor → chunk prefijado con '[...]'",
      ctx.startswith("["),
      f"Inicio: {ctx[:60]}")

ctx = recuperar_contexto_dsm5("síntomas esquizofrenia criterios DSM episodio psicótico", vs_dsm5)
check("DSM · esquizofrenia → devuelve contexto", bool(ctx))
check("DSM · esquizofrenia → contiene 'esquizofreni'",
      "esquizofreni" in ctx.lower(),
      f"Fragmento: {ctx[:120]}")

ctx = recuperar_contexto_dsm5("trastorno ansiedad generalizada preocupación criterios", vs_dsm5)
check("DSM · ansiedad generalizada → devuelve contexto", bool(ctx))

ctx = recuperar_contexto_dsm5("trastorno bipolar episodio maníaco criterios DSM", vs_dsm5)
check("DSM · bipolar → devuelve contexto", bool(ctx))
check("DSM · bipolar → contiene 'bipolar' o 'maníaco'",
      "bipolar" in ctx.lower() or "maníaco" in ctx.lower() or "maniaco" in ctx.lower(),
      f"Fragmento: {ctx[:120]}")

# [F] Threshold DSM
print("\n[F] Threshold DSM — consultas no diagnósticas")

ctx = recuperar_contexto_dsm5("glucosa colesterol analítica resultados laboratorio", vs_dsm5)
check("DSM · analítica → bloqueado (vacío)",
      ctx == "",
      f"Devolvió: {ctx[:80]}")

ctx = recuperar_contexto_dsm5("dosis escitalopram mirtazapina posología", vs_dsm5)
check("DSM · posología fármacos → bloqueado o contenido mínimo",
      ctx == "" or len(ctx) < 300,
      f"Longitud devuelta: {len(ctx)}")

# [G] Calidad de chunks DSM-5
print("\n[G] Calidad de chunks DSM-5")

raw_dsm = Chroma(persist_directory=CHROMA_DB_PATH,
                 embedding_function=emb, collection_name="dsm5_guia")
docs  = raw_dsm._collection.get(include=["documents"])["documents"]
metas_dsm = raw_dsm._collection.get(include=["metadatas"])["metadatas"]

headers_capitulo = [
    "Trastornos depresivos", "Trastornos neurocognitivos",
    "Trastornos de ansiedad", "Trastornos del desarrollo neurológico",
    "Trastornos relacionados con sustancias y trastornos adictivos",
]
contaminados = sum(1 for d in docs for h in headers_capitulo if f"\n{h}\n" in d)
check(f"Sin running headers de capítulo en chunks ({contaminados} contaminados)",
      contaminados == 0)

con_trastorno = sum(1 for m in metas_dsm if m.get("trastorno"))
total_dsm = len(metas_dsm)
check(f"Todos los chunks DSM tienen 'trastorno' ({con_trastorno}/{total_dsm})",
      con_trastorno == total_dsm)

ids = raw_dsm._collection.get()["ids"]
duplicados = len(ids) - len(set(ids))
check(f"Sin IDs duplicados en colección DSM-5 ({duplicados} duplicados)",
      duplicados == 0)

# ══════════════════════════════════════════════════════
ok   = sum(1 for e, _ in resultados if e == OK)
fail = sum(1 for e, _ in resultados if e == FAIL)
print(f"\n══════════════════════════════════════════════")
print(f"  RESULTADO: {ok} OK  |  {fail} FAIL  |  {ok + fail} total")
print(f"══════════════════════════════════════════════\n")
