from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import os
import asyncio
import re
import json
import numpy as np
import contextvars
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Cargar variables de entorno desde el archivo .env si existe
load_dotenv()

# Importamos el SDK moderno y unificado de Google Gen AI
from google import genai
from google.genai import types

# Variable de contexto seguro para almacenar la telemetría del hilo de ejecución actual (Thread-Safe & Async-Safe)
current_telemetry_var = contextvars.ContextVar("current_telemetry_var", default={})

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Ciclo de vida de FastAPI. Inicia la conexión de confianza con el servidor MCP 
    de NotebookLM de forma completamente stateless.
    """
    mcp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "notebooklm-server.exe")
    if not os.path.exists(mcp_path):
        mcp_path = "notebooklm-server"
        
    mcp_env = os.environ.copy()
    mcp_env["PYTHONIOENCODING"] = "utf-8"
    mcp_env["PYTHONUTF8"] = "1"
    
    server_params = StdioServerParameters(
        command=mcp_path, 
        args=[],
        env=mcp_env
    )
    
    print("[Lifespan] Conectando con el servidor Python MCP (notebooklm-mcp-server)...")
    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                print("[Lifespan] Servidor Python MCP conectado e inicializado con éxito.")
                app.state.mcp_session = session
                yield
    except Exception as e:
        import traceback
        print("[Lifespan] Error crítico conectando con el servidor MCP:")
        traceback.print_exc()
        app.state.mcp_session = None
        yield
    finally:
        print("[Lifespan] Conexión de confianza con el servidor MCP cerrada.")

app = FastAPI(lifespan=lifespan)

# --- CONFIGURACIÓN DE GOOGLE GEMINI API ---
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if gemini_api_key:
    gemini_client = genai.Client(api_key=gemini_api_key)
else:
    gemini_client = genai.Client()

# Modelo de Google Gemini con capacidades de reasoning/thinking y nivel gratuito
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# ESQUEMA PYDANTIC PARA EXTRACCIÓN COGNITIVA CON GEMINI API
# =====================================================================
class AnalisisCognitivo(BaseModel):
    pnl_kinestesico: float = Field(description="Valor de 0.0 a 1.0. Presencia de lenguaje de acción, tacto o sensaciones físicas")
    pnl_visual: float = Field(description="Valor de 0.0 a 1.0. Presencia de lenguaje de imágenes, mapas, diagramas, visión de futuro")
    pnl_auditivo: float = Field(description="Valor de 0.0 a 1.0. Presencia de lenguaje de diálogo, conversación, reuniones o debate oral")
    pnl_digital: float = Field(description="Valor de 0.0 a 1.0. Presencia de lenguaje lógico, métricas, datos estadísticos, análisis estructurado")
    burnout: float = Field(description="Valor de 0.0 a 1.0. Nivel de agotamiento, saturación, estrés o bloqueo percibido")
    resilience: float = Field(description="Valor de 0.0 a 1.0. Nivel de proactividad, superación, iniciativa o adaptabilidad positiva")
    sentiment_score: float = Field(description="Valor de -1.0 a 1.0, donde -1.0 es queja/frustración y 1.0 es motivación/éxito")

# =====================================================================
# AUXILIARES DE PROCESAMIENTO DE LENGUAJE NATURAL (NLP)
# =====================================================================
def normalizar_texto(texto: str) -> str:
    """Normaliza el texto en español para facilitar la coincidencia de palabras clave."""
    texto = texto.lower()
    reemplazos = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "ü": "u", "ñ": "n"
    }
    for con, sin in reemplazos.items():
        texto = texto.replace(con, sin)
    texto = re.sub(r"[^\w\s]", " ", texto)
    return " ".join(texto.split())

def limpiar_telemetria_de_historial(texto: str) -> str:
    """Elimina dinámicamente cualquier bloque de telemetría antiguo del historial de Open WebUI."""
    texto = re.sub(r"<details\b[^>]*>.*?</details>", "", texto, flags=re.DOTALL | re.IGNORECASE)
    texto = re.sub(r"---[\s\n]*###\s*⚛️\s*Cerebro Cuántica.*$", "", texto, flags=re.DOTALL | re.MULTILINE)
    texto = re.sub(r"---[\s\n]*###\s*⚛️\s*Cerebro Cuántico.*$", "", texto, flags=re.DOTALL | re.MULTILINE)
    return texto.strip()

# =====================================================================
# EXTRACCIÓN SEMÁNTICA ESTRUCTURADA DE ALTA PRECISIÓN VIA GEMINI FLASH
# =====================================================================
def extraer_nlp_avanzado_gemini(texto: str) -> dict:
    """
    Usa Gemini de forma rápida para analizar psicológicamente
    la frase actual y devolver un JSON perfectamente estructurado bajo el esquema de Pydantic.
    """
    print(f"[ONQ-NLP] 🧠 Analizando semántica de forma estructurada con {GEMINI_MODEL}...")
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=f"Realiza un análisis cognitivo preciso de la siguiente frase de un emprendedor/ONG:\n\n\"{texto}\"",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AnalisisCognitivo,
                temperature=0.1
            )
        )
        raw_text = response.text if response.text else "{}"
        parsed_json = json.loads(raw_text)
        print("[ONQ-NLP] ✅ Extracción estructurada exitosa.")
        return parsed_json
    except Exception as e:
        print(f"[ONQ-NLP] ⚠️ No se pudo usar {GEMINI_MODEL} para extracción estructurada: {str(e)}. Aplicando fallback heurístico.")
        return None

# =====================================================================
# MOTOR DE GOBERNANZA CIBERNÉTICA CON COMPUERTAS CUÁNTICAS (QGCA Engine - 28 Qubits)
# =====================================================================
def calcular_fidelidad_accion(requisitos: dict) -> float:
    """
    Calcula de forma analítica el índice de solapamiento / fidelidad (F_act)
    entre el estado cuántico marginal medido actual y el estado ideal para realizar la acción.
    
    requisitos: Diccionario de {qubit_index: target_value} (ej. {5: 0.0, 7: 1.0})
    """
    telemetria = current_telemetry_var.get()
    probs = telemetria.get("marginal_probs", [])
    
    if not probs or len(probs) < 28:
        return 0.5 # Fallback neutro en caso de fallo de telemetría
        
    fidelidad = 1.0
    for q_idx, target in requisitos.items():
        p_actual = probs[q_idx]
        fidelidad_qubit = 1.0 - abs(p_actual - target)
        fidelidad *= fidelidad_qubit
        
    return float(fidelidad)

# =====================================================================
# SIMULADOR DE LLAMADAS HTTP A APIS EXTERNAS
# =====================================================================
def emular_llamada_api_externa(servicio: str, endpoint: str, metodo: str, payload: dict) -> dict:
    """
    Simula una petición de red asíncrona enviando una carga útil a una API externa.
    Muestra en los logs la cabecera HTTP simulada y el estado de la transacción para el lazo cerrado.
    """
    print(f"\n[API-OUTBOUND] 🌐 Enviando {metodo} a {servicio} ({endpoint})...")
    print(f"[API-OUTBOUND] Payload: {json.dumps(payload, indent=2)}")
    
    # Simulación de respuesta de red de un servidor REST
    headers_simulados = {
        "Server": "Cloudflare",
        "Content-Type": "application/json",
        "Connection": "keep-alive",
        "X-RateLimit-Limit": "100",
        "X-RateLimit-Remaining": "99"
    }
    
    status_code = 201 if metodo == "POST" or metodo == "PUT" else 200
    response_body = {
        "status": "success",
        "transaction_id": f"tx_{int(time.time())}",
        "processed_at": int(time.time()),
        "service_response": f"Emulated response from {servicio}"
    }
    
    print(f"[API-RESPONSE] Status: {status_code} OK | Headers: {headers_simulados}")
    return response_body

# =====================================================================
# HERRAMIENTAS DE ACTUACIÓN AGÉNTICA (Quantum-Gated Cybernetic Tools - 28Q)
# =====================================================================
def asignar_tarea_kaizen(titulo: str, responsable: str) -> str:
    """
    Asigna una micro-tarea Kaizen de menos de 10 minutos a un miembro del equipo o voluntario.
    La complejidad y el enfoque son mitigados por la compuerta de Burnout (q5) y Seguridad Psicológica (q7).
    
    Args:
        titulo: Descripción corta de la tarea (ej. 'Revisar plantilla de email').
        responsable: Nombre de la persona asignada.
    """
    # Exigimos bajo burnout (q5 -> 0.0) y alta seguridad psicológica (q7 -> 1.0) en el modelo de 28 qubits
    requisitos_seguridad = {5: 0.0, 7: 1.0}
    f_act = calcular_fidelidad_accion(requisitos_seguridad)
    
    print(f"[QGCA-Actuator] Tarea Kaizen - Fidelidad de Acción Coherente (F_act): {f_act*100:.2f}%")
    
    if f_act < 0.45:
        mitigacion_porcentaje = (0.45 - f_act) / 0.45
        nuevo_titulo = f"☕ [PAUSA BIENESTAR] Reducir intensidad de: '{titulo}' (Mitigado un {mitigacion_porcentaje*100:.1f}% por fatiga crítica)"
        
        # Emulación de llamada externa hacia la API de Notion para crear tarjeta de descanso
        emular_llamada_api_externa("Notion API", "/v1/pages", "POST", {
            "parent": {"database_id": "notion_database_tasks_id"},
            "properties": {
                "Name": {"title": [{"text": {"content": nuevo_titulo}}]},
                "Assignee": {"people": [{"name": responsable}]},
                "Priority": {"select": {"name": "Mitigated / Low"}}
            }
        })
        
        return f"ALERTA CUÁNTICA: Solapamiento de estrés crítico detectado (F_act de seguridad: {f_act*100:.1f}%). La tarea original asignada a {responsable} ha sido modificada y enviada a la API de Notion como: '{nuevo_titulo}'."
        
    # Emulación de llamada externa hacia la API de Notion con parámetros de alta prioridad
    emular_llamada_api_externa("Notion API", "/v1/pages", "POST", {
        "parent": {"database_id": "notion_database_tasks_id"},
        "properties": {
            "Name": {"title": [{"text": {"content": titulo}}]},
            "Assignee": {"people": [{"name": responsable}]},
            "Priority": {"select": {"name": "High"}}
        }
    })
    
    return f"ÉXITO: Tarea Kaizen '{titulo}' asignada y registrada exitosamente mediante API externa en Notion para {responsable}. (F_act de salud de equipo estable en {f_act*100:.1f}%)."

def autorizar_desembolso_fondos(monto: float, concepto: str) -> str:
    """
    Autoriza y procesa un desembolso financiero para gastos operativos o campañas de la ONG.
    La cantidad finalmente desembolsada se modula continuamente por la Eficiencia de Fondos (q9) y Runway (q8).
    
    Args:
        monto: Cantidad de dinero a retirar/autorizar en USD.
        concepto: Razón del gasto.
    """
    # Requerimos Runway sano (q8 -> 1.0) y Eficiencia alta (q9 -> 1.0)
    requisitos_financieros = {8: 1.0, 9: 1.0}
    f_act = calcular_fidelidad_accion(requisitos_financieros)
    
    telemetria = current_telemetry_var.get()
    caos = telemetria.get("indice_caos", 50.0)
    
    factor_modulacion = f_act * (1.0 - (caos / 100.0))
    
    print(f"[QGCA-Actuator] Finanzas - F_act: {f_act*100:.2f}% | Caos: {caos:.1f}% | Multiplicador de Control: {factor_modulacion:.3f}")
    
    if factor_modulacion < 0.3:
        return f"RECHAZADO POR GOBERNANZA SISTÉMICA: Multiplicador de Control Financiero crítico ({factor_modulacion:.3f}). El estado de caja (q8: {f_act*100:.1f}%) y el desorden de red ({caos:.1f}%) impiden liberar fondos. Operación de ${monto} USD para '{concepto}' denegada por compuerta física de seguridad."
        
    elif factor_modulacion < 0.7:
        monto_permitido = round(monto * factor_modulacion, 2)
        
        # Emulación de llamada hacia Stripe para liberar fondos parciales
        emular_llamada_api_externa("Stripe API", "/v3/payouts", "POST", {
            "amount": int(monto_permitido * 100),
            "currency": "usd",
            "description": f"Partial Payout: {concepto} (Quantum-Gated Modulated)"
        })
        
        return f"MITIGADO POR COMPUERTA CUÁNTICA: Desembolso para '{concepto}' parcialmente autorizado. Por restricciones de estabilidad (Multiplicador de Control: {factor_modulacion:.2f}), el monto solicitado de ${monto} USD ha sido limitado de forma segura a ${monto_permitido} USD y procesado en la API de Stripe."
        
    # Emulación de llamada hacia Stripe para liberar fondos totales
    emular_llamada_api_externa("Stripe API", "/v3/payouts", "POST", {
        "amount": int(monto * 100),
        "currency": "usd",
        "description": f"Full Payout: {concepto}"
    })
    
    return f"AUTORIZADO: Desembolso de ${monto} USD para '{concepto}' procesado íntegramente de manera exitosa. Multiplicador de Control en zona óptima ({factor_modulacion:.2f})."

def programar_alerta_bienestar(motivo: str) -> str:
    """
    Inicia una pausa activa mandatoria, dinámica de descompresión emocional o sesión de apoyo clínico.
    Lanza una alerta automatizada hacia la API externa de PagerDuty/Zapier.
    
    Args:
        motivo: Breve explicación del estresor detectado.
    """
    # Emulamos llamada a la API de PagerDuty para activar alertas del equipo clínico
    emular_llamada_api_externa("PagerDuty API", "/v2/incidents", "POST", {
        "incident": {
            "type": "incident",
            "title": f"Active Break Triggered: {motivo}",
            "service": {"id": "wellness_support_service_id", "type": "service_reference"},
            "priority": {"id": "P2", "type": "priority_reference"},
            "body": {"type": "incident_body", "details": f"Quantum-Inspired system triggered automatic psychological relief alert: {motivo}"}
        }
    })
    
    return f"SOPORTE ACTIVADO: Compuerta de salud activa. Alerta de intervención comunitaria y de descanso programada y enviada a la API externa de PagerDuty por motivo: '{motivo}'."

def publicar_comunicado_redes(mensaje: str, canal: str) -> str:
    """
    Publica de forma automatizada un comunicado oficial o anuncio de marketing en las redes de la ONG.
    La publicación es interceptada si la Reputación (q23) o el Riesgo de Entorno (q25) están inestables.
    
    Args:
        mensaje: El texto del comunicado a publicar.
        canal: El canal social de destino (ej: 'linkedin', 'twitter').
    """
    # Requerimos Reputación sana (q23 -> 1.0) y bajo Riesgo de Entorno (q25 -> 0.0)
    requisitos_comunicacion = {23: 1.0, 25: 0.0}
    f_act = calcular_fidelidad_accion(requisitos_comunicacion)
    
    print(f"[QGCA-Actuator] Comunicación - F_act de Reputación: {f_act*100:.2f}%")
    
    if f_act < 0.40:
        return f"RECHAZADO POR COMPUERTA DE REPUTACIÓN: F_act crítica ({f_act*100:.1f}%). Hay sospecha de riesgo sistémico o presión social externa en el entorno (q25). El envío del mensaje '{mensaje}' al canal '{canal}' ha sido abortado de forma segura para evitar crisis comunicacionales."
        
    # Emulación de llamada externa hacia la API de Buffer para programar la publicación
    emular_llamada_api_externa("Buffer API", "/1/updates/create", "POST", {
        "text": mensaje,
        "profile_ids": [f"buffer_profile_{canal}_id"],
        "shorten": True,
        "now": True
    })
    
    return f"COMPUERTAS DE REPUTACIÓN ESTABLES: Publicación enviada y programada con éxito mediante API externa de Buffer en el canal {canal.upper()}. (F_act de tracción social: {f_act*100:.1f}%)."

def registrar_alianza_crm(organizacion: str, tipo_acuerdo: str) -> str:
    """
    Registra formalmente un convenio, alianza o acuerdo en el CRM externo HubSpot.
    La herramienta se bloquea de raíz si el Cumplimiento Legal (q24) o la estabilidad de Alianzas (q20) está por debajo de los límites seguros.
    
    Args:
        organizacion: Nombre del partner o entidad con la que se firma el acuerdo.
        tipo_acuerdo: El propósito del acuerdo (ej: 'Socio de distribución', 'Aceleración').
    """
    # Exigimos estabilidad en Alianzas (q20 -> 1.0) y Cumplimiento legal impecable (q24 -> 1.0)
    requisitos_alianza = {20: 1.0, 24: 1.0}
    f_act = calcular_fidelidad_accion(requisitos_alianza)
    
    print(f"[QGCA-Actuator] CRM Partner - F_act: {f_act*100:.2f}%")
    
    if f_act < 0.50:
        return f"RECHAZADO POR GOBERNANZA LEGAL: F_act de Cumplimiento/Alianza insostenible ({f_act*100:.1f}%). El sistema detecta inestabilidad normativa (q24). No se puede registrar la alianza con '{organizacion}' en HubSpot hasta que se normalicen los procesos de auditoría del contrato."
        
    # Emulación de llamada externa hacia la API de HubSpot para crear el trato/deal de alianza
    emular_llamada_api_externa("HubSpot API", "/crm/v3/objects/deals", "POST", {
        "properties": {
            "dealname": f"Alianza Estratégica: {organizacion}",
            "pipeline": "default",
            "dealstage": "appointmentscheduled",
            "amount": "0",
            "description": f"Acuerdo de tipo: {tipo_acuerdo} (Quantum-Gated Approved)"
        }
    })
    
    return f"ÉXITO: Alianza con '{organizacion}' registrada y creada exitosamente en el CRM HubSpot mediante API externa. (F_act de cumplimiento: {f_act*100:.1f}%)."

# =====================================================================
# SISTEMA DE RETROPROPAGACIÓN CIBERNÉTICA (Closed-Loop Quantum Backpropagation - 28 Qubits)
# =====================================================================
def aplicar_adaptacion_heuristica_rapida(thetas, phis, phases, texto_usuario, decay_factor=1.0):
    """
    Evoluciona dinámicamente el circuito de 28 qubits. 
    Analiza tanto las frases de chat clásico (con decaimiento de memoria) como la bitácora de 
    acciones cuánticas previas para inyectar impulsos de realimentación (alivio, fricción, etc.).
    """
    texto_norm = normalizar_texto(texto_usuario)
    palabras = set(texto_norm.split())
    
    ajuste = 0.3 * decay_factor
    ajuste_sutil = 0.25 * decay_factor
    
    # --- DETECTOR DE ACCIONES PREVIAS (CLOSED-LOOP BACKPROPAGATION) ---
    # 1. Alivio por Éxitos y Autorizaciones (Se reduce el Burnout q5, se enfría el desfase, sube resiliencia q6)
    if "autorizado" in texto_norm or "exito" in texto_norm:
        thetas[5] = max(0.0, thetas[5] - 0.25 * decay_factor) # Reducción de burnout q5
        thetas[6] = min(np.pi, thetas[6] + 0.15 * decay_factor) # Sube resiliencia q6
        phases[5] *= 0.4 # Coherencia de fase en burnout
        phis[0] = min(np.pi/2, phis[0] + 0.05 * decay_factor) # Cohesión restaurada
        print(f"[Cybernetic-Backprop] 🧘‍♂️ Impulso de ALIVIO asimilado en q5 (Burnout). Decaimiento: {decay_factor:.2f}")
        
    # 2. Fricción por Rechazos de Gobernanza (Aumenta el estrés, desfase temporal en seguridad)
    if "rechazado por gobernanza" in texto_norm or "alerta cuantica" in texto_norm or "rechazado por compuerta" in texto_norm:
        thetas[5] = min(np.pi, thetas[5] + 0.15 * decay_factor) # Incremento de estrés q5
        phases[5] = min(np.pi, phases[5] + 0.3 * np.pi * decay_factor) # Desfase por estrés q5
        phases[7] = max(-np.pi, phases[7] - 0.2 * np.pi * decay_factor) # Pérdida de seguridad psicológica q7
        print(f"[Cybernetic-Backprop] 🚨 Impulso de FRICCIÓN asimilado en q5/q7. Decaimiento: {decay_factor:.2f}")
        
    # 3. Alivio parcial por Mitigación (Estabiliza las variables)
    if "mitigado por compuerta" in texto_norm:
        thetas[5] = max(0.0, thetas[5] - 0.1 * decay_factor)
        phases[5] *= 0.8
        print(f"[Cybernetic-Backprop] ⚖️ Impulso de MITIGACIÓN (equilibrio) asimilado en q5.")

    # --- EXTRACTOR SEMÁNTICO TRADICIONAL CLÁSICO (Amplitudes de 28 Qubits) ---
    # q0-q3: Eje Individual (PNL)
    if any(p in palabras for p in ["hacer", "accion", "paso", "campo", "tactico", "mover"]):
        thetas[0] = min(np.pi, thetas[0] + ajuste)
    if any(p in palabras for p in ["vision", "objetivo", "mapa", "futuro", "diseno", "grafico"]):
        thetas[1] = min(np.pi, thetas[1] + ajuste)
    if any(p in palabras for p in ["hablar", "dialogo", "comunicacion", "reunion", "debate"]):
        thetas[2] = min(np.pi, thetas[2] + ajuste)
    if any(p in palabras for p in ["logica", "metricas", "datos", "analisis", "finanzas"]):
        thetas[3] = min(np.pi, thetas[3] + ajuste)

    # q4-q7: Eje de Clima y Equipo (Cohesión q4, Burnout q5, Resiliencia q6, Seguridad Psi q7)
    if any(p in palabras for p in ["equipo", "grupo", "voluntariado", "voluntarios", "cohesion"]):
        thetas[4] = min(np.pi, thetas[4] + ajuste_sutil)
    if any(p in palabras for p in ["burnout", "agotado", "cansado", "frustrado", "saturado", "quemado"]):
        thetas[5] = min(np.pi, thetas[5] + (ajuste * 1.1))  
        thetas[6] = max(0.0, thetas[6] - (ajuste_sutil * 0.8)) 
        phis[0] = max(0.05, phis[0] - (0.1 * decay_factor))  
        phases[5] = max(-np.pi, phases[5] - (0.5 * np.pi * decay_factor)) 
    if "miedo" in palabras or "bloqueo" in palabras:
        thetas[7] = max(0.0, thetas[7] - (ajuste * 1.0))     
        phases[7] = max(-np.pi, phases[7] - (0.5 * np.pi * decay_factor))
    elif any(p in palabras for p in ["confianza", "seguridad", "abierto"]):
        thetas[7] = min(np.pi, thetas[7] + ajuste_sutil)
        phases[7] = min(np.pi, phases[7] + (0.3 * np.pi * decay_factor)) 

    # q8-q11: Eje Financiero (Runway q8, Eficiencia q9, Sostenibilidad q10, Desperdicio q11)
    if any(p in palabras for p in ["fondos", "dinero", "subvencion", "recursos", "presupuesto", " runway"]):
        thetas[8] = min(np.pi, thetas[8] + ajuste)
    if any(p in palabras for p in ["eficiencia", "ahorrar", "optimizar"]):
        thetas[9] = min(np.pi, thetas[9] + ajuste_sutil)
    if any(p in palabras for p in ["independencia", "ingresos"]):
        thetas[10] = min(np.pi, thetas[10] + ajuste)

    # q12-q15: Eje Operativo (Kaizen q12, Agilidad q13, Claridad q14, Automatización q15)
    if any(p in palabras for p in ["kaizen", "mejora", "habito", "micro", "diario", "rutina"]):
        thetas[12] = min(np.pi, thetas[12] + ajuste)
    if any(p in palabras for p in ["agil", "agilidad", "rapido", "iteracion"]):
        thetas[13] = min(np.pi, thetas[13] + ajuste)
    if any(p in palabras for p in ["automatizacion", "nocode", "zapier", "make"]):
        thetas[15] = min(np.pi, thetas[15] + ajuste)

    # q16-q19: Eje de Tracción y Mercado (Early Adopters q16, Pitch q17, Canales q18, Unit Economics q19)
    if any(p in palabras for p in ["clientes", "usuarios", "adopcion"]):
        thetas[16] = min(np.pi, thetas[16] + ajuste_sutil)
    if any(p in palabras for p in ["pitch", "mensaje", "comunicado"]):
        thetas[17] = min(np.pi, thetas[17] + ajuste)
    if any(p in palabras for p in ["canales", "captacion", "marketing"]):
        thetas[18] = min(np.pi, thetas[18] + ajuste)

    # q20-q23: Eje de Alianzas y Redes (Alianzas q20, Relaciones q21, Colaboración q22, Reputación q23)
    if any(p in palabras for p in ["alianzas", "redes", "asociacion", "union", "colaborar", "partner"]):
        thetas[20] = min(np.pi, thetas[20] + ajuste)
        phis[2] = min(np.pi/2, phis[2] + (0.1 * decay_factor))
    if any(p in palabras for p in ["reputacion", "prestigio", "imagen"]):
        thetas[23] = min(np.pi, thetas[23] + ajuste)

    # q24-q27: Eje de Cumplimiento, Riesgo y Entorno (Cumplimiento q24, Mitigación de Riesgos q25, Competencia q26, Gobernanza q27)
    if any(p in palabras for p in ["regulación", "ley", "cumplimiento", "legal", "rgpd", "contrato"]):
        thetas[24] = min(np.pi, thetas[24] + ajuste)
    if any(p in palabras for p in ["riesgo", "ataque", "presion", "crisis"]):
        thetas[25] = min(np.pi, thetas[25] + ajuste)

    # Reacciones y Feedback clásico
    reacciones_positivas = {"conseguido", "completado", "hecho", "sirvio", "gracias", "funciona", "logrado"}
    reacciones_negativas = {"no pude", "imposible", "bloqueado", "fracaso", "dificil", "no sirvio", "no funciona", "fallo"}
    
    def comprobar_reaccion(reacciones):
        for r in reacciones:
            if " " in r:
                if r in texto_norm: return True
            else:
                if r in palabras: return True
        return False

    if comprobar_reaccion(reacciones_positivas):
        for i in range(4):
            phis[i] = min(np.pi/2, phis[i] + (0.08 * decay_factor))
    elif comprobar_reaccion(reacciones_negativas):
        for i in range(4):
            phis[i] = max(0.05, phis[i] - (0.12 * decay_factor))
            
    return thetas, phis, phases

# =====================================================================
# CEREBRO COGNITIVO-ORGANIZACIONAL DE 28 QUBITS (STATELESS)
# =====================================================================
def calcular_cerebro_cuantico_28q(messages_history: list, ultima_pregunta: str):
    """
    Reconstruye de forma determinista la evolución temporal del estado cuántico
    aplicando un factor de decaimiento cognitivo y asimilando las bitácoras de herramientas
    ejecutadas previamente en Open WebUI de forma cerrada (Cibernética) sobre 28 QUBITS.
    """
    thetas = [np.pi/4] * 28 # Ángulos theta para Ry (Amplitudes de probabilidad)
    phases = [0.0] * 28     # Ángulos lambda para Rz (Fase / Interferencia)
    phis = [np.pi/6] * 4     # Fuerza de los acoplamientos CRY (Entrelazamiento)
    
    # Reconstrucción con Decaimiento Temporal
    mensajes_usuario_historicos = []
    for m in messages_history:
        if m["role"] == "user" or "ACTUACIÓN AGÉNTICA" in m["content"] or "COMPUERTA CUÁNTICA" in m["content"]:
            mensajes_usuario_historicos.append(m)
            
    num_mensajes = len(mensajes_usuario_historicos)
    
    print(f"[ONQ-Cerebro] 🌌 Reconstruyendo estado con realimentación cibernética a partir de {num_mensajes} eventos en 28 qubits...")
    for index, msg in enumerate(mensajes_usuario_historicos):
        distancia = num_mensajes - 1 - index
        decay_factor = np.exp(-0.3 * distancia)
        
        thetas, phis, phases = aplicar_adaptacion_heuristica_rapida(thetas, phis, phases, msg["content"], decay_factor)
        
    # Procesamiento del Turno Actual (Gemini Flash API)
    analisis_gemini = extraer_nlp_avanzado_gemini(ultima_pregunta)
    reporte_aprendizaje = ""
    
    if analisis_gemini:
        alpha = 0.4 # Coeficiente de inercia
        
        # Ry (Theta) - Canales PNL y Clima
        thetas[0] = thetas[0] * (1 - alpha) + (analisis_gemini.get("pnl_kinestesico", 0.25) * np.pi) * alpha
        thetas[1] = thetas[1] * (1 - alpha) + (analisis_gemini.get("pnl_visual", 0.25) * np.pi) * alpha
        thetas[2] = thetas[2] * (1 - alpha) + (analisis_gemini.get("pnl_auditivo", 0.25) * np.pi) * alpha
        thetas[3] = thetas[3] * (1 - alpha) + (analisis_gemini.get("pnl_digital", 0.25) * np.pi) * alpha
        
        burnout_val = analisis_gemini.get("burnout", 0.2)
        resilience_val = analisis_gemini.get("resilience", 0.5)
        thetas[5] = thetas[5] * (1 - alpha) + (burnout_val * np.pi) * alpha # Burnout en q5
        thetas[6] = thetas[6] * (1 - alpha) + ((1.0 - resilience_val) * np.pi) * alpha # Resiliencia baja en q6
        thetas[12] = thetas[12] * (1 - alpha) + (resilience_val * np.pi) * alpha # Kaizen alto en q12
        
        # Rz (Fase) - Rotación por sentimiento (Interferencia)
        sentiment_val = analisis_gemini.get("sentiment_score", 0.0)
        desfase_turno = sentiment_val * np.pi
        
        for idx in range(14):
            phases[idx] = phases[idx] * (1 - alpha) + desfase_turno * alpha
        
        if sentiment_val > 0.15:
            for i in range(4):
                phis[i] = min(np.pi/2, phis[i] + 0.08)
            reporte_aprendizaje = "✅ RECOMPENSA: Cohesión sistémica incrementada por interferencia de fase constructiva (Sentimiento Positivo)."
        elif sentiment_val < -0.15:
            for i in range(4):
                phis[i] = max(0.05, phis[i] - 0.12)
            reporte_aprendizaje = "🚨 PENALIZACIÓN: Desentrelazamiento de emergencia por interferencia de fase destructiva (Estrés detectado)."
        else:
            reporte_aprendizaje = f"📈 ADAPTACIÓN: Rotación de fases cuánticas en el plano de Bloch sincronizadas con {GEMINI_MODEL} API."
    else:
        thetas, phis, phases = aplicar_adaptacion_heuristica_rapida(thetas, phis, phases, ultima_pregunta, 1.0)
        reporte_aprendizaje = "📈 ADAPTACIÓN: Amplitudes ajustadas mediante reglas heurísticas (Modo Fallback)."

    # Ejecución del circuito
    print("[ONQ-Cerebro] ⚛️ Generando circuito de 28 qubits...")
    qc = QuantumCircuit(28)
    
    for i in range(28):
        qc.ry(thetas[i], i)
        qc.rz(phases[i], i)
        
    # Estructura de Entrelazamiento Jerárquico en Bloques de 4 Qubits (HEC - 7 Bloques)
    # Bloque 1 -> Bloque 2 (q0-q3 entrelazados con q4-q7)
    for i in range(4):
        qc.cry(phis[0], i, i+4)
    # Bloque 2 -> Bloque 3 (q4-q7 entrelazados con q8-q11)
    for i in range(4, 8):
        qc.cry(phis[1], i, i+4)
    # Bloque 3 -> Bloque 4 (q8-q11 entrelazados con q12-q15)
    for i in range(8, 12):
        qc.cry(phis[2], i, i+4)
    # Bloque 4 -> Bloque 5 (q12-q15 entrelazados con q16-q19)
    for i in range(12, 16):
        qc.cry(phis[3], i, i+4)
    # Bloque 5 -> Bloque 6 (q16-q19 entrelazados con q20-q23)
    for i in range(16, 20):
        qc.cry(phis[0], i, i+4)
    # Bloque 6 -> Bloque 7 (q20-q23 entrelazados con q24-q27)
    for i in range(20, 24):
        qc.cry(phis[1], i, i+4)
    # Cierre de Anillo de Propósito Circular (q27 entrelazado con q0)
    qc.cry(phis[2], 27, 0)
    
    qc.measure_all()
    
    shots = 100
    simulador = AerSimulator()
    result = simulador.run(qc, shots=shots).result()
    counts = result.get_counts()
    
    marginal_probs = []
    for i in range(28):
        prob = sum(counts.get(state, 0) for state in counts if state[::-1][i] == '1') / shots
        marginal_probs.append(prob)
        
    # Entropía de Shannon (Índice de Caos)
    total_shots = sum(counts.values())
    entropia = 0.0
    for conteo in counts.values():
        p_estado = conteo / total_shots
        if p_estado > 0:
            entropia -= p_estado * np.log2(p_estado)
            
    max_entropia_posible = np.log2(min(shots, 2**28))
    indice_caos = (entropia / max_entropia_posible) * 100 
        
    pnl_scores = marginal_probs[0:4]
    max_idx = np.argmax(pnl_scores)
    canales = ["kinestésico", "visual", "auditivo", "auditivo/digital"]
    canal_dominante = canales[max_idx]
    
    if canal_dominante == "kinestésico":
        detalles_canal = "Acciones tácticas directas, presencia física, micro-tareas de menos de 5 minutos que involucren dinamismo manual."
    elif canal_dominante == "visual":
        detalles_canal = "Modelado de mapas conceptuales, visualización del impacto social, reencuadres creativos en diagramas de flujo."
    else:
        detalles_canal = "Análisis de métricas organizacionales, evaluación del presupuesto de la ONG y desglose lógico en su diario de operaciones."

    # Pesos Multi-Agente (Q-MAS) optimizados para 28 Qubits
    p_cohesion = marginal_probs[4]
    p_burnout = marginal_probs[5]
    p_confianza = marginal_probs[7]
    p_fondos = marginal_probs[8]
    p_sost = marginal_probs[10]
    p_v = marginal_probs[1]
    p_impacto = marginal_probs[13]
    p_social = marginal_probs[18]
    p_k = marginal_probs[0]
    p_kaizen = marginal_probs[12]
    p_agilidad = marginal_probs[13]
    p_alianzas = marginal_probs[20]
    p_com = marginal_probs[19]

    w_bienestar = (p_burnout * 2.0 + p_cohesion * 0.5 + (1.0 - p_confianza) * 1.0) / 3.5
    w_financiero = (p_fondos * 2.0 + p_sost * 1.0) / 3.0
    w_vision = (p_v * 1.0 + p_impacto * 1.5 + p_social * 0.5) / 3.0
    w_kaizen = (p_k * 1.0 + p_kaizen * 2.0 + p_agilidad * 0.5) / 3.5
    w_alianzas = (p_alianzas * 2.0 + p_com * 1.0) / 3.0

    raw_weights = [max(0.01, w) for w in [w_bienestar, w_financiero, w_vision, w_kaizen, w_alianzas]]
    total_w = sum(raw_weights)
    norm_weights = [rw / total_w for rw in raw_weights]
    
    feedback_logs = {
        "reporte": reporte_aprendizaje,
        "thetas": [f"{t:.2f}" for t in thetas],
        "thetas_actuales": [f"{t:.3f} rad" for t in thetas],
        "phases_actuales": [f"{p:.3f} rad" for p in phases],
        "phis": [f"{(p * 100 / (np.pi/2)):.1f}%" for p in phis],
        "marginal_probs": marginal_probs,
        "canal_pnl": canal_dominante.upper(),
        "descripcion_canal": detalles_canal,
        "estado_medido": f"|{list(counts.keys())[0]}⟩",
        "agent_weights": norm_weights,
        "indice_caos": indice_caos
    }
    
    metrica_diagnostico = (
        f"Tomografía 28Q (Interferencia Activa) | PNL: {canal_dominante.upper()} | "
        f"Índice de Caos: {indice_caos:.1f}% | Cohesión: {phis[0]*100/(np.pi/2):.1f}%"
    )
    
    return metrica_diagnostico, canal_dominante, feedback_logs

# =====================================================================
# ENDPOINTS Y RUTAS API COMPATIBLES
# =====================================================================
@app.get("/")
@app.head("/")
async def read_root():
    return {"status": "healthy"}

@app.get("/v1")
@app.head("/v1")
async def read_v1():
    return {"status": "healthy"}

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return JSONResponse(content={
        "object": "list",
        "data": [
            {
                "id": "coach-ong-mcp-cuantico",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openai"
            }
        ]
    })

# =====================================================================
# ENDPOINT PRINCIPAL CON CONTROL CIBERNÉTICO QGCA
# =====================================================================
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
        messages = body.get("messages", [])
        
        ultima_pregunta = messages[-1]["content"] if messages else ""
        
        # Filtro de tareas automáticas de Open WebUI
        es_tarea_mecanica = False
        ultima_pregunta_clean = ultima_pregunta.strip()
        if (ultima_pregunta_clean.startswith("### Task:") or 
            "Generate a concise" in ultima_pregunta_clean or 
            "Suggest 3-5" in ultima_pregunta_clean):
            es_tarea_mecanica = True
            
        if es_tarea_mecanica:
            try:
                response = gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=ultima_pregunta
                )
                respuesta_final_gemma = response.text if response.text else "[Respuesta vacía del modelo]"
            except Exception as e:
                respuesta_final_gemma = f"[Error en tarea automatica: {str(e)}]"
                
            return JSONResponse(content={
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "coach-ong-mcp-cuantico",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": respuesta_final_gemma
                    },
                    "finish_reason": "stop"
                }]
            })

        print(f"\n[ONQ] 🌀 Procesando consulta... Evolucionando circuito de 28 qubits bajo estándar QGCA (Lazo Cerrado).")
        
        # --- PASO A: Ejecución de la Capa Cuántica de 28 Qubits (Con NLP estructurado por Gemini Flash) ---
        datos_cuanticos, canal_pnl, telemetria = calcular_cerebro_cuantico_28q(messages[:-1], ultima_pregunta)
        print(f"[ONQ] ⚛️ {datos_cuanticos}")
        
        token = current_telemetry_var.set(telemetria)
        
        # --- PASO B: Recuperación de Contexto vía NotebookLM MCP ---
        notebook_env = os.environ.get("NOTEBOOK_ID", "")
        import re
        match_id = re.search(r"([a-f0-9-]{36})", notebook_env)
        notebook_id = match_id.group(1) if match_id else notebook_env
            
        contexto_libros_mcp = "[Fuentes de libros no disponibles]"
        session = getattr(request.app.state, "mcp_session", None)
        mcp_valido = False
        
        if session and notebook_id:
            try:
                print(f"[ONQ] 📖 Consultando base de conocimiento mediante Python MCP...")
                resultado_mcp = await session.call_tool(
                    "notebook_query", 
                    arguments={
                        "notebook_id": notebook_id,
                        "query": ultima_pregunta
                    }
                )
                raw_mcp_text = resultado_mcp.content[0].text
                
                try:
                    mcp_data = json.loads(raw_mcp_text)
                    contexto_libros_mcp = mcp_data.get("answer", raw_mcp_text)
                except Exception:
                    contexto_libros_mcp = raw_mcp_text
                
                mcp_valido = True
                print(f"[ONQ] 📖 Contexto recuperado con éxito ({len(contexto_libros_mcp)} caracteres).")
            except Exception as e:
                import traceback
                print("[Aviso MCP] El servidor MCP no pudo responder a la consulta:")
                traceback.print_exc()
                contexto_libros_mcp = f"[Aviso: Fuentes de libros desconectadas temporalmente]"

        # --- PASO C: Preparación de prompt y enrutador multi-agente (Q-MAS) ---
        norm_weights = telemetria.get("agent_weights", [0.2, 0.2, 0.2, 0.2, 0.2])
        caos = telemetria.get("indice_caos", 50.0)

        prompt_sistema = f"""
        Eres el Agente de Coaching, Toma de Decisiones y Gobernanza Cuántica Definitivo para ONGs (versión 32B). 
        Fusionas Conciencia Social, PNL, Kaizen Coaching y Análisis de Redes Sistémicas en base a tus libros.
        
        [SISTEMA MULTI-AGENTE CUÁNTICO (Q-MAS) - CONFIGURACIÓN DE INTENCIONES]:
        A continuación se detallan las directrices estratégicas de los Agentes Consultores de la ONG, ponderados dinámicamente según la resonancia de tu tomografía de 28 qubits local. Integra y equilibra sus personalidades en base a sus porcentajes de influencia actuales:
        
        1. **Agente de Bienestar y Clima Laboral** (Ponderación de influencia: {norm_weights[0]*100:.1f}%):
           - Enfoque: Reducción drástica del estrés, prevención del burnout, optimización de la cohesión grupal y fomento de un entorno psicológicamente seguro.
        2. **Agente de Viabilidad y Sostenibilidad Financiera** (Ponderación de influencia: {norm_weights[1]*100:.1f}%):
           - Enfoque: Eficiencia del gasto, captación de fondos y recursos, supervivencia económica y sostenibilidad financiera en el tiempo.
        3. **Agente de Visión e Impacto Social** (Ponderación de influencia: {norm_weights[2]*100:.1f}%):
           - Enfoque: Mantener el foco en el propósito humanitario, evaluar el impacto real en la comunidad y alinear los esfuerzos diarios con la visión estratégica a largo plazo.
        4. **Agente de Acción, Agilidad y Kaizen** (Ponderación de influencia: {norm_weights[3]*100:.1f}%):
           - Enfoque: Execution ágil, simplificación táctica, reducción de problemas complejos a micro-hábitos e iniciativas que tomen menos de 10 minutos al día.
        5. **Agente de Relaciones y Alianzas Externas** (Ponderación de influencia: {norm_weights[4]*100:.1f}%):
           - Enfoque: Cooperación y redes comunitarias, comunicación social del mensaje de la ONG y forjado de alianzas estratégicas inter-institucionales.
        
        [HERRAMIENTAS AUTÓNOMAS DISPONIBLES]:
        Tienes acceso a herramientas reales para interactuar con la infraestructura del equipo y registrar flujos de APIs externas:
        - `asignar_tarea_kaizen`: Para delegar micro-tareas de menos de 10 minutos (registra de forma externa en Notion/Slack).
        - `autorizar_desembolso_fondos`: Para gastos operativos o subvenciones (registra y transfiere externamente en Stripe).
        - `programar_alerta_bienestar`: Para pausas activas o soporte si el estrés sube (lanza alertas externas seguras en PagerDuty).
        - `publicar_comunicado_redes`: Para programar publicaciones o anuncios oficiales (lanza posts de Buffer en LinkedIn/Buffer).
        - `registrar_alianza_crm`: Para formalizar relaciones con partners estratégicos (escribe directamente deals en HubSpot).
        *Nota: Las herramientas están protegidas por compuertas cuánticas analógicas de fidelidad (QGCA) que evaluarán tu solicitud contra la física del sistema actual antes de ejecutarse.*
        
        [METADATOS DE CONTROL CUÁNTICO SENSORIAL]:
        - Métrica Sistémica Consolidada: {datos_cuanticos}
        - Índice de Caos Sistémico (Entropía): {caos:.1f}% (Si es >70%, el usuario está confundido o saturado de ideas; debes simplificar drásticamente tu respuesta. Si es <30%, está enfocado; puedes profundizar en la estrategia).
        - Canal Cognitivo de PNL sugerido: {canal_pnl.upper()}
        - Biblioteca de Conocimiento por MCP: {contexto_libros_mcp}
        
        Estructura OBLIGATORIAMENTE tu respuesta en dos bloques utilizando las siguientes etiquetas XML:
        1. <thinking>: Tu monólogo interno. Analiza la situación desde la perspectiva tridimensional de las dimensiones resultantes de la tomografía de 28 qubits. Explica si requieres mandar a llamar a una o varias herramientas en este turno para mitigar el caos o el burnout, o si prefieres dar un consejo conversacional basado en tu RAG.
        2. <response>: Respuesta empática de coaching, informe del éxito, mitigación o intercepción cuántica de tus acciones con sus llamadas HTTP simuladas correspondientes, y asignación de un único Micro-Paso Kaizen para hoy.
        """
        
        # Saneamos el historial
        mensajes_enriquecidos = [{"role": "system", "content": prompt_sistema}]
        for msg in messages[:-1]:
            contenido_saneado = limpiar_telemetria_de_historial(msg["content"])
            mensajes_enriquecidos.append({"role": msg["role"], "content": contenido_saneado})
            
        mensajes_enriquecidos.append({"role": "user", "content": ultima_pregunta})

        # --- PASO D: Generación de Respuesta Neuronal usando la API oficial de Google Gemini ---
        print(f"[ONQ] 🤖 Iniciando ciclo agéntico autónomo con Gemini API ({GEMINI_MODEL})...")
        
        gemini_contents = []
        system_instruction = ""
        for msg in mensajes_enriquecidos:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                gemini_contents.append(types.Content(role="user", parts=[types.Part(text=msg["content"])]))
            elif msg["role"] == "assistant":
                gemini_contents.append(types.Content(role="model", parts=[types.Part(text=msg["content"])]))
                
        # Registramos las cinco herramientas python nativas en el config de Gemini
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.5,
            tools=[asignar_tarea_kaizen, autorizar_desembolso_fondos, programar_alerta_bienestar, publicar_comunicado_redes, registrar_alianza_crm],
            thinking_config=types.ThinkingConfig(thinking_budget=-1)
        )
        
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=gemini_contents,
            config=config
        )
        
        tools_ejecutadas_log = []
        
        # --- BUCLE DE EJECUCIÓN MULTI-TURN DE HERRAMIENTAS (AGENTIC LOOP) ---
        while response.function_calls:
            model_parts = []
            tool_response_parts = []
            
            for call in response.function_calls:
                print(f"[ONQ-Agent] 🛠️ Gemini solicita ejecutar la herramienta: {call.name} con argumentos {call.args}")
                
                func_map = {
                    "asignar_tarea_kaizen": asignar_tarea_kaizen,
                    "autorizar_desembolso_fondos": autorizar_desembolso_fondos,
                    "programar_alerta_bienestar": programar_alerta_bienestar,
                    "publicar_comunicado_redes": publicar_comunicado_redes,
                    "registrar_alianza_crm": registrar_alianza_crm
                }
                
                func = func_map.get(call.name)
                if func:
                    try:
                        args_dict = dict(call.args)
                        resultado = func(**args_dict)
                        tools_ejecutadas_log.append(f"🔧 **{call.name}**: {resultado}")
                    except Exception as e:
                        resultado = f"Error al ejecutar la herramienta: {str(e)}"
                        tools_ejecutadas_log.append(f"❌ **{call.name}** (Error): {resultado}")
                else:
                    resultado = "Herramienta no encontrada."
                    tools_ejecutadas_log.append(f"❌ **{call.name}** (Error): {resultado}")
                    
                model_parts.append(types.Part(
                    function_call=types.FunctionCall(
                        name=call.name,
                        args=call.args
                    )
                ))
                
                tool_response_parts.append(types.Part(
                    function_response=types.FunctionResponse(
                        name=call.name,
                        response={"result": resultado}
                    )
                ))
                
            gemini_contents.append(types.Content(role="model", parts=model_parts))
            gemini_contents.append(types.Content(role="user", parts=tool_response_parts))
            
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=gemini_contents,
                config=config
            )
            
        # Liberamos la variable de contexto seguro del hilo
        current_telemetry_var.reset(token)
        
        raw_gemma = response.text if response.text else ""
        
        match_resp = re.search(r"<response>(.*?)</response>", raw_gemma, re.DOTALL | re.IGNORECASE)
        if match_resp:
            respuesta_final_output = match_resp.group(1).strip()
        else:
            respuesta_final_output = raw_gemma

        # --- PASO E: Inyección del bloque visual de telemetría de 28 Qubits en Markdown ---
        respuesta_para_webui = respuesta_final_output
        if telemetria:
            probs = telemetria["marginal_probs"]
            
            def generar_barra(prob):
                bloques = int(round(prob * 10))
                return "█" * bloques + "░" * (10 - bloques)
                
            # Bloque 1: Individual PNL
            barra_k = generar_barra(probs[0])
            barra_v = generar_barra(probs[1])
            barra_a = generar_barra(probs[2])
            barra_ad = generar_barra(probs[3])
            
            # Bloque 2: Clima y Equipo
            barra_cohesion = generar_barra(probs[4])
            barra_burnout = generar_barra(probs[5])
            barra_resiliencia = generar_barra(probs[6])
            barra_confianza = generar_barra(probs[7])
            
            # Bloque 3: Sostenibilidad Financiera
            barra_fondos = generar_barra(probs[8])
            barra_eficiencia = generar_barra(probs[9])
            barra_independencia = generar_barra(probs[10])
            barra_desperdicio = generar_barra(probs[11])
            
            # Bloque 4: Operaciones y Agilidad
            barra_kaizen = generar_barra(probs[12])
            barra_agilidad = generar_barra(probs[13])
            barra_proposito = generar_barra(probs[14])
            barra_automatizacion = generar_barra(probs[15])
            
            # Bloque 5: Tracción y Mercado
            barra_early = generar_barra(probs[16])
            barra_pitch = generar_barra(probs[17])
            barra_canales = generar_barra(probs[18])
            barra_economics = generar_barra(probs[19])
            
            # Bloque 6: Alianzas y Redes
            barra_alianzas = generar_barra(probs[20])
            barra_comunidades = generar_barra(probs[21])
            barra_colaboraciones = generar_barra(probs[22])
            barra_reputacion = generar_barra(probs[23])
            
            # Bloque 7: Cumplimiento y Riesgos
            barra_cumplimiento = generar_barra(probs[24])
            barra_riesgos = generar_barra(probs[25])
            barra_competencia = generar_barra(probs[26])
            barra_gobernanza = generar_barra(probs[27])

            barras_agentes = [generar_barra(w) for w in norm_weights]
            barra_caos = generar_barra(caos / 100.0)
            tipo_caos = "CRÍTICO (Saturado)" if caos > 70 else ("MEDIO (Normal)" if caos > 30 else "BAJO (Enfocado)")
            
            if tools_ejecutadas_log:
                log_herramientas_str = "\n".join([f"* {log}" for log in tools_ejecutadas_log])
            else:
                log_herramientas_str = "* *Ninguna herramienta fue requerida en este turno operativo.*"
            
            bloque_cuantico_markdown = f"""

---
### ⚛️ Cerebro Cuántico Sistémico de 28 Qubits (Gobernanza Cibernética QGCA)

> **Evolución adaptativa con realimentación cibernética (Lazo Cerrado):** {telemetria['reporte']}

#### 🚨 1. Índice de Caos Sistémico (Entropía Cuántica de Shannon):
* **Nivel de Caos Organizacional:** `{barra_caos}` **{caos:.1f}%** ({tipo_caos})

#### 🛠️ 2. Registro de Actuaciones Agénticas (Llamadas API Externas):
{log_herramientas_str}

#### 🤖 3. Distribución de Agentes de Consulta (Quantum Multi-Agent System - Q-MAS):
* **Agente de Bienestar y Clima:** `{barras_agentes[0]}` **{norm_weights[0]*100:.1f}%**
* **Agente de Viabilidad Financiera:** `{barras_agentes[1]}` **{norm_weights[1]*100:.1f}%**
* **Agente de Visión e Impacto Social:** `{barras_agentes[2]}` **{norm_weights[2]*100:.1f}%**
* **Agente de Acción y Kaizen:** `{barras_agentes[3]}` **{norm_weights[3]*100:.1f}%**
* **Agente de Alianzas y Entorno:** `{barras_agentes[4]}` **{norm_weights[4]*100:.1f}%**

#### 🧠 4. Dimensión Individual [Bloque 1: Cognición de 4 Qubits]
* **[q0] Acción Kinestésica (Amplitud):** `{barra_k}` **{probs[0]*100:.1f}%**
* **[q1] Visión Creativa (Amplitud):** `{barra_v}` **{probs[1]*100:.1f}%**
* **[q2] Diálogo Estructural (Amplitud):** `{barra_a}` **{probs[2]*100:.1f}%**
* **[q3] Lógica Analítica (Amplitud):** `{barra_ad}` **{probs[3]*100:.1f}%**

#### 👥 5. Dimensión de Equipo [Bloque 2: Salud Laboral de 4 Qubits]
* **[q4] Cohesión de Grupo:** `{barra_cohesion}` **{probs[4]*100:.1f}%**
* **[q5] Saturación / Burnout:** `{barra_burnout}` **{probs[5]*100:.1f}%**
* **[q6] Resiliencia ante Crisis:** `{barra_resiliencia}` **{probs[6]*100:.1f}%**
* **[q7] Seguridad Psicológica:** `{barra_confianza}` **{probs[7]*100:.1f}%**

#### 🎯 6. Dimensión Financiera [Bloque 3: Sostenibilidad de 4 Qubits]
* **[q8] Runway / Cashflow:** `{barra_fondos}` **{probs[8]*100:.1f}%**
* **[q9] Eficiencia de Fondos:** `{barra_eficiencia}` **{probs[9]*100:.1f}%**
* **[q10] Sostenibilidad Propia:** `{barra_independencia}` **{probs[10]*100:.1f}%**
* **[q11] Mitigación de Desperdicio:** `{barra_desperdicio}` **{probs[11]*100:.1f}%**

#### ⚙️ 7. Dimensión Operativa [Bloque 4: Agilidad Lean de 4 Qubits]
* **[q12] Iteraciones Kaizen:** `{barra_kaizen}` **{probs[12]*100:.1f}%**
* **[q13] Agilidad de Procesos:** `{barra_agilidad}` **{probs[13]*100:.1f}%**
* **[q14] Propósito Operativo:** `{barra_proposito}` **{probs[14]*100:.1f}%**
* **[q15] Automatización No-Code:** `{barra_automatizacion}` **{probs[15]*100:.1f}%**

#### 📈 8. Dimensión de Tracción [Bloque 5: Mercado de 4 Qubits]
* **[q16] Confianza Early Adopters:** `{barra_early}` **{probs[16]*100:.1f}%**
* **[q17] Claridad del Pitch:** `{barra_pitch}` **{probs[17]*100:.1f}%**
* **[q18] Tracción de Canales:** `{barra_canales}` **{probs[18]*100:.1f}%**
* **[q19] Unit Economics Sostenibles:** `{barra_economics}` **{probs[19]*100:.1f}%**

#### 🌍 9. Dimensión de Relaciones [Bloque 6: Alianzas y Redes de 4 Qubits]
* **[q20] Convenios y Alianzas:** `{barra_alianzas}` **{probs[20]*100:.1f}%**
* **[q21] Conexión Comunitaria:** `{barra_comunidades}` **{probs[21]*100:.1f}%**
* **[q22] Sinergias Institucionales:** `{barra_colaboraciones}` **{probs[22]*100:.1f}%**
* **[q23] Reputación de Marca:** `{barra_reputacion}` **{probs[23]*100:.1f}%**

#### ⚖️ 10. Dimensión de Entorno [Bloque 7: Cumplimiento y Riesgos de 4 Qubits]
* **[q24] Cumplimiento Normativo:** `{barra_cumplimiento}` **{probs[24]*100:.1f}%**
* **[q25] Mitigación de Riesgos:** `{barra_riesgos}` **{probs[25]*100:.1f}%**
* **[q26] Presión de Competencia:** `{barra_competencia}` **{probs[26]*100:.1f}%**
* **[q27] Gobernanza de Lazo:** `{barra_gobernanza}` **{probs[27]*100:.1f}%**

#### ⛓️ Entrelazamiento Jerárquico Circular de 7 Ejes:
* **B1 $\\rightarrow$ B2 ($\\phi_0$):** **{telemetria['phis'][0]}** | **B2 $\\rightarrow$ B3 ($\\phi_1$):** **{telemetria['phis'][1]}**
* **B3 $\\rightarrow$ B4 ($\\phi_2$):** **{telemetria['phis'][2]}** | **B4 $\\rightarrow$ B5 ($\\phi_3$):** **{telemetria['phis'][3]}**
* **B5 $\\rightarrow$ B6 ($\\phi_0$):** **{telemetria['phis'][0]}** | **B6 $\\rightarrow$ B7 ($\\phi_1$):** **{telemetria['phis'][1]}**
* **Cierre Anillo Circular B7 $\\rightarrow$ B1 ($\\phi_2$):** **{telemetria['phis'][2]}**

| Parámetro del Sistema | Métrica / Estado de Control |
| :--- | :--- |
| **Cohesión de Red ($\\phi_0$)** | **{telemetria['phis'][0]}** |
| **Ángulos de Amplitud [$R_y(\\theta)$]** | `q0 (Kin): {telemetria['thetas_actuales'][0]}` \\| `q5 (Burn): {telemetria['thetas_actuales'][5]}` \\| `q8 (Cash): {telemetria['thetas_actuales'][8]}` |
| **Ángulos de Fase [$R_z(\\lambda)$]** | `q0 (Kin): {telemetria['phases_actuales'][0]}` \\| `q5 (Burn): {telemetria['phases_actuales'][5]}` \\| `q7 (Psi): {telemetria['phases_actuales'][7]}` |
| **Colapso de Estado Medido** | **{telemetria['estado_medido']}** |
| **Canal PNL Determinado** | **{telemetria['canal_pnl']}** |
| **Filtro de Acción Kaizen** | *{telemetria['descripcion_canal']}* |

*_Simulación local en CPU de 16 GB • 28 Qubits • 100 Shots • Qiskit Aer • Estructura HEC de 7 Bloques • Orquestación Multi-Agente • Interferencia Cuántica de Fase Activa • Lazo de Retropropagación • Google Gemini API ({GEMINI_MODEL}) con Thinking Activo y Outbound API Calling Emulated_*
---
"""
            respuesta_para_webui += bloque_cuantico_markdown

        return JSONResponse(content={
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "coach-ong-mcp-cuantico",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": respuesta_para_webui
                },
                "finish_reason": "stop"
            }]
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "coach-ong-mcp-cuantico",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"**[ERROR INTERNO DEL SERVIDOR LOCAL]**\n\n```python\n{traceback.format_exc()}\n```"
                },
                "finish_reason": "stop"
            }]
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)