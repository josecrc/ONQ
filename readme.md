⚛️ ONQ: Quantum-Inspired Cognitive Steering Engine (28-Qubit QGCA)

ONQ (Quantum-Inspired Cognitive Steering Engine) es un framework middleware de gobernanza híbrida cuántico-clásica diseñado para controlar la deriva cognitiva, mitigar la autocomplacencia (sycophancy) y regular emocionalmente agentes autónomos basados en Modelos de Lenguaje de Gran Escala (LLMs).

A diferencia de las arquitecturas de agentes tradicionales basadas en grafos rígidos o condicionales lógicos discretos (if/else), ONQ mapea el estado dinámico multidimensional de una organización o usuario en un espacio de Hilbert de 28 qubits. El sistema utiliza simulación cuántica local ultra-eficiente en CPU (Qiskit Aer) para evaluar continuamente la viabilidad de transacciones en red y modular el tono, la brevedad y el alcance de las herramientas (Tool Calling) de Google Gemini.

🚀 Innovaciones Clave de la Archictectura

                        [ INPUT: CONSULTA DEL USUARIO ]
                                       │
                                       ▼ (Gemini API Extraer JSON)
                        ┌──────────────────────────────┐
                        │ Análisis Cognitivo por LLM   │
                        │ (PNL, Burnout, Resiliencia)  │
                        └──────────────┬───────────────┘
                                       │
                                       ▼ (Rotaciones en Plano de Bloch)
                        ┌──────────────────────────────┐
                        │   Interferencia de Fase Z    │
                        │     y Amplitudes Ry(θ)       │
                        └──────────────┬───────────────┘
                                       │
                                       ▼ (Qiskit Aer local < 5 MB RAM)
                        ┌──────────────────────────────┐
                        │ Simulación de 28 Qubits (HEC)│
                        └──────────────┬───────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼ (Análisis Marginal)                 ▼ (Entropía de Shannon)
        ┌────────────────────────┐            ┌────────────────────────┐
        │   Gobernanza QGCA      │            │   Índice de Caos (%)   │
        │ Intercepta APIs Ext.   │            │ Modula el Prompt / LLM │
        └────────────────────────┘            └────────────────────────┘


1. Tablero Cuántico de 28 Qubits (7 Bloques Operacionales)

El circuito se estructura de forma perfectamente simétrica en 7 bloques de 4 qubits cada uno, modelando dependencias dinámicas cruzadas mediante compuertas de entrelazamiento controlado ($CRY$):

Bloque 1 

$$q0-q3$$

: Eje Individual / Cognición PNL (Acción Kinestésica, Visión, Diálogo, Análisis Lógico).

Bloque 2 

$$q4-q7$$

: Clima y Salud de Equipo (Cohesión, Burnout, Resiliencia, Seguridad Psicológica).

Bloque 3 

$$q8-q11$$

: Sostenibilidad Financiera (Runway, Eficiencia, Independencia, Mitigación de Desperdicio).

Bloque 4 

$$q12-q15$$

: Operaciones y Agilidad Lean (Kaizen de 10 min, Agilidad, Propósito, Automatización).

Bloque 5 

$$q16-q19$$

: Tracción y Mercado (Early Adopters, Claridad del Pitch, Canales de Captación, Economics).

Bloque 6 

$$q20-q23$$

: Alianzas y Redes Externas (Convenios, Relaciones Comunitarias, Sinergias, Reputación).

Bloque 7 

$$q24-q27$$

: Cumplimiento, Riesgo y Entorno (Cumplimiento Legal, Mitigación de Riesgos, Competencia, Gobernanza).

2. Gobernanza Cibernética de Herramientas (QGCA)

Cada herramienta (tool) agéntica está resguardada por una compuerta cuántica. El sistema calcula la Fidelidad de Proyección de Acción ($F_{\text{act}}$) entre el estado actual medido y el target ideal de la acción:

$$F_{\text{act}}(A) = \prod_{q \in \mathcal{Q}} \left[ 1 - \vert{}P(q) - T(q)\vert{} \right]$$

Si $F_{\text{act}}$ cae por debajo de los límites de seguridad (ej. burnout crítico o inestabilidad reputacional), el actuador intercepta la llamada de red, bloqueando el flujo o mitigando dinámicamente la intensidad de la acción de forma analógica y continua antes de llamar a la infraestructura real.

3. Retropropagación en Lazo Cerrado (Closed-Loop Backpropagation)

Al ser un sistema libre de estado (stateless), el estado cuántico se reconstruye deterministamente turno a turno analizando sintácticamente el historial del chat. El motor procesa las actuaciones de herramientas pasadas de forma cerrada:

Éxitos / Aprobaciones emiten un impulso de alivio que reduce el burnout ($q_5$), enfría el desfasamiento y eleva la cohesión.

Bloqueos / Fricciones inyectan desfasamientos destructivos de $+\pi$ rad en la fase del plano de Bloch, simulando el impacto del bloqueo operativo en la moral de la organización.

4. Control de Complejidad por Entropía de Shannon

Calculamos la entropía resultante de la tomografía cuántica de 100 disparos (shots):

$$H(X) = -\sum_{i} P(x_i) \log_2 P(x_i)$$

Si el índice de caos $C = (H(X) / H_{\text{max}}) \cdot 100$ supera el $70\%$, el prompt del sistema de Gemini se reconfigura en caliente para forzar consejos de extrema brevedad y micro-pasos Kaizen minimalistas, evitando la parálisis por análisis del usuario.

🛠️ Herramientas de Red con APIs Externas Emuladas

ONQ incluye actuadores diseñados para interactuar con la infraestructura real de tu equipo:

asignar_tarea_kaizen (Notion/Slack API): Registra micro-tareas de menos de 10 minutos. Gobernado por burnout ($q_5$) y seguridad psicológica ($q_7$).

autorizar_desembolso_fondos (Stripe API): Procesa transacciones de caja. Gobernado por runway ($q_8$) y eficiencia ($q_9$).

programar_alerta_bienestar (PagerDuty API): Lanza guardias y alertas de apoyo emocional y descanso.

publicar_comunicado_redes (Buffer API): Programa comunicados oficiales. Gobernado por reputación ($q_{23}$) y riesgo ($q_{25}$).

registrar_alianza_crm (HubSpot API): Formaliza acuerdos con partners. Gobernado por alianzas ($q_{20}$) y cumplimiento legal ($q_{24}$).

🛠️ Instalación y Quickstart

Prerrequisitos

Python 3.10 o superior

Memoria RAM: Optimizado para ordenadores estándar. Gracias a la simulación por Matrix Product State en Qiskit Aer, el consumo en ejecución de 28 qubits es insignificante (< 5 MB), corriendo de forma instantánea.

Configuración e Inicio

Clona el repositorio e instala las dependencias:

git clone [https://github.com/josecrc/ONQ.git](https://github.com/josecrc/ONQ.git)
cd ONQ
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install fastapi uvicorn qiskit qiskit-aer numpy pydantic google-genai python-dotenv mcp


Configura tu archivo .env en la raíz del proyecto:

GEMINI_API_KEY="tu-clave-api-de-google-ai-studio"
GEMINI_MODEL="gemini-3.1-flash-lite"
NOTEBOOK_ID="tu-notebook-id-de-notebooklm"  # Opcional para MCP RAG


Arranca el servidor local de FastAPI:

python app.py


El middleware se desplegará localmente en http://localhost:8000. Es compatible al 100% con el estándar de OpenAI para conectarse directamente a interfaces de chat como Open WebUI.

📄 Licencia y Cláusula Defensiva de Patentes

Este proyecto está bajo la Licencia Apache 2.0.

¿Por qué Apache 2.0? A diferencia de las licencias tradicionales (como MIT), Apache 2.0 incluye una concesión explícita y recíproca de derechos de patente entre los contribuyentes y los usuarios del software. Si cualquier entidad o multinacional utiliza este código e intenta demandarte alegando infracción de patente de software, perderá de forma inmediata y automática todas sus licencias, derechos de uso e inmunidad sobre este framework. Esto blinda a los desarrolladores independientes y a la comunidad de código abierto contra trolls de patentes.

Para más detalles, consulte el archivo LICENSE.md.