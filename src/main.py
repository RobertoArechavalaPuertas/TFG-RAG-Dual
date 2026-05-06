import sys
import os

# Añadir src al path para que los imports funcionen
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retriever import cargar_vector_store
from retriever_dsm5 import cargar_vector_store_dsm5
from rag_chain import cargar_llm, construir_chain

# ── Pacientes válidos en el sistema ──────────────────────────────────────────
PACIENTES_VALIDOS = {
    "P001": "Carmen Vidal Soler",
    "P002": "Miquel Àngel Roca Fuster",
    "P003": "Rosa Maria Beltrán Ortiz",
    "P004": "Andreu Mas Castelló",
    "P005": "Esperança Tormos Vidal",
    "P006": "Marta Esteve Climent",
    "P007": "Adrián Ferrer Blasco",
    "P008": "Lucía Herrero Navarro",
    "P009": "Daniel Ortega Campos",
    "P010": "Sara Molina Grau",
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
    print(f"\n{'─'*55}")
    print(f"  Paciente : {patient_id} — {nombre}")
    print(f"  Pregunta : {pregunta}")
    print(f"{'─'*55}")
    print(f"\n{respuesta}\n")


# ── Validación de patient_id ──────────────────────────────────────────────────
def validar_patient_id(patient_id: str) -> bool:
    """Verifica que el patient_id existe en el sistema.
    
    Esta es la primera línea de defensa del aislamiento:
    si el ID no existe, la consulta no llega nunca al LLM.
    """
    return patient_id.upper() in PACIENTES_VALIDOS


# ── Prueba de aislamiento ─────────────────────────────────────────────────────
def ejecutar_prueba_aislamiento(rag):
    """Demuestra formalmente que el aislamiento por patient_id funciona.
    
    Pregunta algo que existe en P002 (IAM, infarto) usando el ID de P001.
    El sistema debe responder que no encuentra esa información.
    """
    print("\n" + "="*55)
    print("  TEST DE AISLAMIENTO")
    print("="*55)
    print("  Preguntando a P001 sobre un infarto")
    print("  (información que solo existe en P002)")
    print("="*55 + "\n")

    pregunta  = "¿Ha tenido este paciente algún infarto de miocardio o problema cardíaco grave?"
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

    # Preguntar si se quiere ejecutar el test de aislamiento
    respuesta_test = input("¿Ejecutar prueba de aislamiento antes de empezar? (s/n): ").strip().lower()
    if respuesta_test == "s":
        ejecutar_prueba_aislamiento(rag)

    # Bucle de consultas
    while True:
        print()

        # Pedir patient_id
        patient_id = input("ID del paciente (P001-P010): ").strip().upper()

        if patient_id == "SALIR":
            print("\nSistema cerrado. Hasta luego.\n")
            break

        if patient_id == "PACIENTES":
            imprimir_bienvenida()
            continue

        # Validación — primera línea de defensa del aislamiento
        if not validar_patient_id(patient_id):
            print(f"\n  ERROR: '{patient_id}' no existe en el sistema.")
            print(  "  IDs válidos: P001–P010\n")
            continue

        # Pedir pregunta
        pregunta = input(f"Pregunta para {patient_id}: ").strip()

        if not pregunta:
            print("  Por favor escribe una pregunta.")
            continue

        if pregunta.lower() == "salir":
            print("\nSistema cerrado. Hasta luego.\n")
            break

        # Ejecutar RAG
        print("\nBuscando en historial y DSM-5, consultando al LLM...\n")
        respuesta = rag(patient_id, pregunta)
        imprimir_respuesta(patient_id, pregunta, respuesta)


if __name__ == "__main__":
    main()
