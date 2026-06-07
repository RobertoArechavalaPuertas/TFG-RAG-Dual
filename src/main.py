import sys
import os
import textwrap
import shutil

# Añadir src al path para que los imports funcionen
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retriever import cargar_vector_store
from retriever_dsm5 import cargar_vector_store_dsm5
from rag_chain import cargar_llm, construir_chain

# ── Pacientes válidos en el sistema ──────────────────────────────────────────
PACIENTES_VALIDOS = {
    "P001": "Carmen Ruiz Velasco",
    "P002": "Antonio Herrera Lopez",
    "P003": "Maria Jose Martinez Aguilar",
    "P004": "Alejandro Vega Romero",
    "P005": "Laura Navarro Gutierrez",
    "P006": "Marta Esteve Climent",
}


# ── Helpers de interfaz ───────────────────────────────────────────────────────
def imprimir_bienvenida():
    print("\n" + "="*55)
    print("   SISTEMA RAG DUAL — HISTORIALES CLÍNICOS + DSM-5")
    print("   Hospital Universitario de Castellón")
    print("="*55)
    print("  Pacientes disponibles:")
    for pid, nombre in PACIENTES_VALIDOS.items():
        print(f"    {pid} — {nombre}")
    print("="*55)
    print("  Escribe 'salir' para terminar")
    print("  Escribe 'pacientes' para ver la lista")
    print("="*55 + "\n")


def imprimir_respuesta(patient_id: str, pregunta: str, respuesta: str):
    nombre = PACIENTES_VALIDOS[patient_id]
    ancho = min(shutil.get_terminal_size().columns, 100)
    print(f"\n{'─'*55}")
    print(f"  Paciente : {patient_id} — {nombre}")
    print(f"  Pregunta : {pregunta}")
    print(f"{'─'*55}")
    wrapped = "\n".join(
        textwrap.fill(linea, width=ancho) if linea.strip() else ""
        for linea in respuesta.split("\n")
    )
    print(f"\n{wrapped}\n")


# ── Validación de patient_id ──────────────────────────────────────────────────
def validar_patient_id(patient_id: str) -> bool:
    return patient_id.upper() in PACIENTES_VALIDOS


# ── Prueba de aislamiento ─────────────────────────────────────────────────────
def ejecutar_prueba_aislamiento(rag):
    print("\n" + "="*55)
    print("  TEST DE AISLAMIENTO")
    print("="*55)
    print("  Preguntando a P001 sobre un infarto")
    print("  (información que solo existe en P002)")
    print("="*55 + "\n")

    # P002 (Antonio) tiene esquizofrenia; P001 (Carmen) no.
    # Preguntar a P001 sobre esquizofrenia debe devolver que no hay información.
    pregunta  = "¿Tiene este paciente diagnóstico de esquizofrenia o episodios psicóticos?"
    respuesta = rag("P001", pregunta)

    print(f"Pregunta : {pregunta}")
    print(f"\nRespuesta:\n{respuesta}")

    print("\n" + "="*55)
    print("  Ahora la misma pregunta para P002 (debe responder SÍ)")
    print("="*55 + "\n")

    respuesta2 = rag("P002", pregunta)
    print(f"Pregunta : {pregunta}")
    print(f"\nRespuesta:\n{respuesta2}")
    print("\n" + "="*55 + "\n")


# ── Bucle principal ───────────────────────────────────────────────────────────
def main():
    print("\nInicializando sistema RAG dual...")
    print("Cargando bases de datos vectoriales y modelo LLM...\n")

    vs_pacientes = cargar_vector_store()
    vs_dsm5      = cargar_vector_store_dsm5()
    llm          = cargar_llm()
    rag          = construir_chain(llm, vs_pacientes, vs_dsm5)

    print("Sistema listo.\n")
    imprimir_bienvenida()

    respuesta_test = input("¿Ejecutar prueba de aislamiento antes de empezar? (s/n): ").strip().lower()
    if respuesta_test == "s":
        ejecutar_prueba_aislamiento(rag)

    while True:
        print()

        patient_id = input("ID del paciente (P001-P010): ").strip().upper()

        if patient_id == "SALIR":
            print("\nSistema cerrado. Hasta luego.\n")
            break

        if patient_id == "PACIENTES":
            imprimir_bienvenida()
            continue

        if not validar_patient_id(patient_id):
            print(f"\n  ERROR: '{patient_id}' no existe en el sistema.")
            print(  "  IDs válidos: P001–P010\n")
            continue

        pregunta = input(f"Pregunta para {patient_id}: ").strip()

        if not pregunta:
            print("  Por favor escribe una pregunta.")
            continue

        if pregunta.lower() == "salir":
            print("\nSistema cerrado. Hasta luego.\n")
            break

        print("\nBuscando en historial y DSM-5, consultando al LLM...\n")
        respuesta = rag(patient_id, pregunta)
        imprimir_respuesta(patient_id, pregunta, respuesta)


if __name__ == "__main__":
    main()
