# Estado del arte: clasificación de intención con modelos de menos de 4B parámetros

**Periodo cubierto:** 2023–2026

## Tabla de contenidos

1. [Introducción](#1-introducción)
2. [Definición del problema y alcance](#2-definición-del-problema-y-alcance)
3. [Revisión por enfoque](#3-revisión-por-enfoque)
4. [Análisis comparativo](#4-análisis-comparativo)
5. [Datasets, métricas y protocolos de evaluación](#5-datasets-métricas-y-protocolos-de-evaluación)
6. [Desafíos abiertos](#6-desafíos-abiertos)
7. [Posicionamiento del proyecto](#7-posicionamiento-del-proyecto)
8. [Conclusiones](#8-conclusiones)
9. [Referencias](#9-referencias)

---

## 1. Introducción

La clasificación de intención consiste en asignar a un enunciado del usuario una etiqueta de un conjunto cerrado de intenciones (por ejemplo, `alarm_set`). Es el primer paso de los asistentes de voz, los agentes de atención al cliente y los sistemas de enrutamiento de modelos de lenguaje (Ganesh et al., 2026; Chen et al., 2026).

Los grandes modelos de lenguaje (LLM) resuelven esta tarea con buena calidad, pero su despliegue es costoso. Wang et al. (2024) señalan cuatro limitaciones: la dependencia de la nube y los problemas de privacidad que implica, la imposibilidad de ejecutarlos en el dispositivo, la latencia inadecuada para aplicaciones en tiempo real y el alto costo de adaptarlos a dominios específicos. Como ejemplo, estiman en unos 202.5 GB la memoria que requiere Llama 3.1 405B incluso con precisión int4.

Este proyecto trabaja con una restricción explícita: **solo se consideran modelos de menos de 4B (4 mil millones) de parámetros**. La pregunta que organiza este documento es la siguiente: **dentro de ese límite, ¿qué estrategias permiten obtener la mejor clasificación de intención posible y qué evidencia las respalda?**

## 2. Definición del problema y alcance

### 2.1 Tarea

Se considera clasificación de intención de una sola etiqueta sobre enunciados individuales, con tres variantes frecuentes en la literatura: zero-shot (solo nombres o descripciones de las intenciones), few-shot (de 1 a unas decenas de ejemplos por intención) y supervisada con datos completos. Se considera además la detección de enunciados fuera de dominio (OOS, *out-of-scope*), necesaria en sistemas reales (Larson et al., 2019).

### 2.2 Criterio de inclusión: menos de 4B parámetros

La revisión conserva únicamente los resultados de modelos con menos de 4B parámetros. Los modelos mayores que aparecen en las fuentes se mencionan solo como referencia y se marcan como tales. La tabla indica qué parte de cada trabajo cae dentro del límite:

| Trabajo | Modelos dentro del límite (< 4B) | Fuera del límite (solo referencia) |
|---|---|---|
| Ganesh et al. (2026) | 24 de los 41 modelos evaluados, de 0.14B a 3.8B (por ejemplo, Qwen2.5-3B, Phi-3.5-mini, Llama-3.2-1B/3B, SmolLM2) | Modelos de 6B a 9B (Mistral-7B, Llama-3.1-8B, Qwen2.5-7B, entre otros) |
| Parikh et al. (2023) | Flan-T5-Large (770M), Flan-T5-XL (3B), codificador de tamaño similar a LaBSE | Flan-T5-XXL (11B), GPT-3 (175B), GTR-3B |
| Alekseev et al. (2025) | Clasificadores sobre embeddings de codificadores de oraciones | Preset zero-shot con un LLM por API |
| Vera et al. (2025) | EmbeddingGemma (308M) | — |
| Zhang et al. (2025) | BERT, RoBERTa, DistilBERT, ALBERT | LLM comparados en zero-shot (Qwen-7B, Llama-3.1-8B, Mistral) |
| ElZemity et al. (2026) | Qwen2.5-0.5B y SmolLM2-135M (estudiantes) | Maestros (LLM de frontera) |
| Chen et al. (2026) | Llama-3.2-1B y Qwen3-1.7B | Qwen3-8B y Qwen3-32B; Qwen3-4B queda en el límite y no se incluye |
| Das et al. (2026) | Qwen3.5 0.8B, Gemma 3 1B, TinyLlama 1.1B | — |
| Wang et al. (2024) | Revisión general de técnicas de compresión y adaptación | — |

*(El conteo de 24 modelos de Ganesh et al. corresponde a los 41 modelos evaluados que tienen menos de 4B parámetros según su tabla de tamaños; el listado completo está en el trabajo original.)*

### 2.3 Corpus de la revisión

La revisión se basa en nueve trabajos publicados entre 2023 y 2026, identificados mediante búsquedas en arXiv y revisados en su texto completo. Se priorizaron los trabajos que evalúan modelos de este rango de tamaño en clasificación de intención o que aportan técnicas directamente aplicables (embeddings compactos, adaptación con pocos datos, destilación y compresión). Los conjuntos de datos y métodos de referencia que estos trabajos citan se mencionan como fuentes secundarias.

## 3. Revisión por enfoque

### 3.1 Zero-shot con modelos de lenguaje

Ganesh et al. (2026) realizan la evaluación más amplia de modelos abiertos de menos de 10B parámetros: 41 modelos de 15 familias, entre 135M y 9B, en ocho datasets zero-shot (CLINC150, Banking77, SNIPS, MASSIVE, MTOP y tres datasets de comercio electrónico de HINT3), más ATIS con cinco demostraciones, que se excluye del agregado. La Tabla 1 recoge los modelos de menos de 4B (agregado de ocho datasets, con cómputo medido en una sola GPU):

**Tabla 1.** Resultados zero-shot de los modelos de menos de 4B parámetros (Ganesh et al., 2026).

| Modelo | Parámetros (B) | VRAM (GB) | Latencia (ms/muestra) | Agregado (8 datasets) |
|---|---|---|---|---|
| Qwen2.5-3B-Instruct | 3.0 | 5.79 | 20.8 | 0.632 |
| Phi-3.5-mini-Instruct | 3.8 | 7.17 | 40.0 | 0.624 |
| Phi-3-mini-4k-Instruct | 3.8 | 7.12 | 31.9 | 0.568 |
| Qwen2.5-3B (base) | 3.0 | 5.79 | 20.6 | 0.538 |
| Qwen2-1.5B-Instruct | 1.5 | 2.89 | 16.0 | 0.512 |
| Llama-3.2-3B-Instruct | 3.0 | 6.02 | 39.6 | 0.508 |
| Llama-3.2-3B (base) | 3.0 | 6.02 | 21.6 | 0.485 |
| MiniCPM-2B-Instruct | 2.0 | 5.10 | 24.4 | 0.425 |
| Llama-3.2-1B-Instruct | 1.0 | 2.32 | 10.4 | 0.279 |
| Qwen2.5-0.5B-Instruct | 0.5 | 0.93 | 9.0 | 0.274 |
| SmolLM2-360M-Instruct | 0.36 | 0.67 | 10.6 | 0.048 |
| SmolLM2-135M-Instruct | 0.14 | 0.26 | 9.7 | 0.043 |

*Nota:* Qwen2.5-1.5B-Instruct (agregado de 0.462) no entra en el ranking del trabajo porque su evaluación en Banking77 usó el conjunto de prueba completo y no el subconjunto de 500 muestras del resto de modelos. La tabla completa incluye además otros modelos de menos de 4B (por ejemplo, phi-2, StableLM-1.6B y TinyLlama-1.1B).

Los hallazgos principales dentro del límite de 4B son:

- **El mejor modelo del rango es Qwen2.5-3B-Instruct** (0.632), que además es Pareto-óptimo en exactitud, latencia y parámetros. Phi-3.5-mini-Instruct le sigue con 0.624, pero con casi el doble de latencia y más memoria, y no es Pareto-óptimo.
- **El límite cuesta poco en zero-shot.** El mejor modelo de todo el estudio, Mistral-7B-Instruct-v0.3 (7B, fuera del límite), obtiene 0.660, apenas 0.028 puntos por encima de Qwen2.5-3B-Instruct. Cinco de los seis mejores modelos son de 7B a 9B.
- **Por debajo de 3B el rendimiento cae.** Qwen2-1.5B-Instruct es el único modelo de menos de 3B que supera 0.5 (0.512). Los modelos de 1B o menos obtienen agregados de 0.28 o inferiores.
- **El instruction tuning pesa más que la escala.** En las parejas base/instruct de este rango, la versión instruct mejora el agregado (por ejemplo, Qwen2.5-3B: 0.632 frente a 0.538; Llama-3.2-3B: 0.508 frente a 0.485; Qwen2.5-0.5B: 0.274 frente a 0.205). Qwen2.5-3B-Instruct supera además a varios modelos de 7B sin instruction tuning.
- **Los modelos SmolLM2 obtienen agregados cercanos a cero** (0.036 a 0.048) bajo este protocolo, que usa generación libre sin decodificación restringida. Los autores señalan esa generación libre como una limitación del estudio.
- **Eficiencia por parámetro.** Qwen2.5-0.5B-Instruct tiene la mayor exactitud por mil millones de parámetros (0.548), útil si el objetivo es minimizar la huella y no la exactitud absoluta.
- **Robustez y calibración.** Los errores de tipeo son la perturbación más dañina y la que más varía entre modelos; escribir todo en minúsculas no tiene efecto medible. El efecto del instruction tuning sobre la calibración es inconsistente, y SmolLM2-1.7B-Instruct presenta el peor error de calibración esperado (ECE = 0.768). Los autores no evaluaron predicción selectiva ni abstención por umbral.
- **Saturación.** SNIPS está saturado (32 de 41 modelos superan 80 %), mientras que Powerplay11 es el más difícil (el mejor modelo alcanza 53.7 %).

Los autores declaran como limitaciones que la evaluación es solo en inglés, usa las primeras 500 muestras de cada conjunto de prueba y emplea una única semilla en el análisis de robustez.

Parikh et al. (2023) aportan una técnica útil para prompts largos: cuando hay muchas intenciones, recuperan con similitud de oraciones las cinco intenciones más probables y solo ellas se incluyen en el prompt.

### 3.2 Embeddings congelados y clasificadores ligeros

Este enfoque separa la representación del texto (un codificador preentrenado, congelado) de la decisión (un clasificador pequeño sobre los vectores). Su principal ventaja es que el costo de inferencia queda dominado por el codificador y que el clasificador puede reentrenarse en segundos.

**EmbeddingGemma** (Vera et al., 2025) es un modelo de embeddings de 308M parámetros, inicializado desde el codificador de un modelo codificador-decodificador basado en Gemma 3 y entrenado con destilación de embeddings desde Gemini Embedding, un regularizador de dispersión y fusión de checkpoints (*model souping*). Es el mejor modelo de menos de 500M en MTEB multilingüe, inglés y código, y esa ventaja se mantiene al truncar los embeddings a 128 dimensiones (gracias a Matryoshka, con 768, 512, 256 y 128 dimensiones disponibles) o al cuantizar los pesos a 4 bits. La Tabla 1 del trabajo muestra una degradación pequeña por cuantización:

**Tabla 2.** EmbeddingGemma bajo cuantización (Vera et al., 2025).

| Precisión | MTEB Multilingüe v2 (media por tarea) | MTEB Inglés v2 (media por tarea) |
|---|---|---|
| bf16 | 61.15 | 69.67 |
| int8 por bloques | 60.93 | 69.49 |
| int4 por bloques | 60.62 | 69.31 |

La evaluación de MTEB abarca más de 250 idiomas y varios tipos de tarea, entre ellos la clasificación, pero el trabajo no evalúa la clasificación de intención de forma específica.

**AutoIntent** (Alekseev et al., 2025) es un marco de AutoML con tres módulos: embedding, puntuación (*scoring*) y decisión, esta última con umbrales para fuera de dominio y multi-etiqueta. Los clasificadores operan sobre embeddings precalculados, lo que permite el despliegue en CPU. En cinco datasets (Banking77, HWU64, MASSIVE, MINDS-14 y SNIPS), los presets «classic» (kNN, regresión logística, bosques aleatorios y CatBoost) obtienen una exactitud media de 93.2 % (classic-light) y 93.5 % (classic-medium), con 136 s y 216 s de entrenamiento respectivamente. Para comparar, AutoGluon en su configuración media alcanza 92.4 % en 461 s. En CLINC150, con detección fuera de dominio, AutoIntent mantiene una exactitud en dominio de 96.13 % y un F1 de la clase OOS de aproximadamente 76, frente a unos 48 de AutoGluon. En escenarios con pocos datos (de 4 a 128 ejemplos por clase), los autores atribuyen su ventaja a los métodos basados en vecinos. Su preset zero-shot basado en un LLM por API (87.0 %) queda fuera del límite de 4B y solo sirve de referencia. Limitaciones declaradas: no se realizaron estudios con usuarios y la evaluación se limita a conjuntos de intención.

**Adaptación de dominio.** Parikh et al. (2023) entrenan un codificador de oraciones de tamaño comparable a LaBSE con tareas de similitud sobre taxonomías de intención y paráfrasis, y lo usan congelado con una capa densa. Con 3 a 5 ejemplos por intención mejora más de 5 puntos a modelos de tamaño similar y queda a 2-3 puntos de GTR-3B (fuera del límite), un modelo unas 30 veces mayor.

### 3.3 Fine-tuning eficiente y aprendizaje por prompts

**Adaptadores sobre modelos de instrucciones.** Parikh et al. (2023) aplican la receta T-Few (Liu et al., 2022), con adaptadores IA3, sobre Flan-T5 (Chung et al., 2022), con 1, 3 o 5 ejemplos por intención. Es el mejor de los cuatro métodos que comparan. Dentro del límite de 4B, Flan-T5-XL (3B) supera a todos los demás métodos con un solo ejemplo por intención, y Flan-T5-Large (770M) supera a todos salvo a Flan-T5-XXL (11B, fuera del límite) en uno de sus datasets. Los autores concluyen que un modelo mucho menor que GPT-3 puede igualarlo o superarlo con descripciones de intención y unos pocos ejemplos.

**Aprendizaje por prompts con estructura conversacional.** Zhang et al. (2025) proponen SAID, que preentrena un modelo con tokens de relación suaves (relaciones consulta-consulta y consulta-respuesta dentro de una sesión) y reformula el problema few-shot como aprendizaje por prompts. Con BERT como backbone y comparado con BERT, RoBERTa, DistilBERT, ALBERT, CPFT, IntentBERT (Zhang et al., 2021), DNNC, SetFit (Tunstall et al., 2022), LLM-Aug y DFT++, obtiene en IntentChat con 3 ejemplos por clase una exactitud de 52.73 %, frente a 40.59 % de RoBERTa y 40.04 % de SetFit. Con 20 ejemplos por clase (74.05 %) supera a Qwen-7B en zero-shot (72.57 %, fuera del límite). El marco funciona también con DistilBERT y ALBERT. Sus limitaciones para otros contextos son que necesita sesiones de conversación para el preentrenamiento y que uno de los dos datasets (IntentChat) es privado; el otro (WildChat anotado) se publica con el trabajo.

### 3.4 Datos sintéticos y destilación

Wang et al. (2024) distinguen la destilación de caja blanca, donde el estudiante accede a distribuciones o estados internos del maestro, de la de caja negra, donde solo accede a sus salidas (por ejemplo, Distilling Step-by-Step y LaMini-LM). Los LLM como generadores de datos de entrenamiento pertenecen a la segunda familia.

**Aumento de datos.** Parikh et al. (2023) parten de cinco ejemplos por intención y generan veinte paráfrasis con GPT-3 o con un modelo T5. La paráfrasis mejora entre 2 y 4 puntos al clasificador de referencia en la mayoría de los casos; la generación a partir de descripciones de intención solo ayuda cuando se combina con los ejemplos reales.

**Destilación agéntica.** ElZemity et al. (2026) proponen que un LLM maestro actúe como ingeniero de ML autónomo: genera datos sintéticos de entrenamiento y validación, ajusta con LoRA a un estudiante pequeño (Qwen2.5-0.5B o SmolLM2-135M, ambos dentro del límite), evalúa con métricas agregadas, formula hipótesis de fallos y genera datos dirigidos hasta que el rendimiento se estabiliza. Se evaluó en detección de SMS spam con datos reales apartados del bucle:

**Tabla 3.** Exactitud de los estudiantes según el maestro (ElZemity et al., 2026).

| Maestro | Estudiante Qwen2.5-0.5B | Estudiante SmolLM2-135M |
|---|---|---|
| Claude Opus 4.5 | 94.31 % | 86.28 % |
| DeepSeek V3.2 | 92.10 % | 86.21 % |
| Gemini 3 Pro | 85.21 % | 80.46 % |
| GPT 5.2 Codex | 71.08 % | 59.71 % |

Sin ajuste, los estudiantes alcanzan 49.80 % y 46.85 % de exactitud. El bucle iterativo supera a una línea base DPO con los mismos datos sintéticos (50-80 %). Cada ejecución requirió entre 14 800 y 33 000 tokens y entre 5 y 9 minutos en un Mac Mini M4 de 16 GB. Las limitaciones son que la tarea es binaria, que se usa un único conjunto de prueba (un subconjunto balanceado de 1494 mensajes) y que la calidad final depende fuertemente del maestro, con sesgos de precisión o recall cuando los datos generados están desbalanceados o son poco diversos.

### 3.5 Métodos sin entrenamiento sobre activaciones

Chen et al. (2026) comparan dos métodos sin gradientes, NormStat y VecStat (estadísticas de las activaciones del modelo durante la fase de *prefill*), con cabezas MLP, sondas lineales, un RoBERTa-base ajustado y llamadas directas al LLM, en modelos Qwen3 y Llama de 1B a 32B. Dentro del límite de 4B están Llama-3.2-1B y Qwen3-1.7B. Sus conclusiones generales son que:

- en tareas gruesas (matemáticas, código y texto general) todos los métodos saturan;
- en tareas finas, los clasificadores entrenados suelen ser mejores;
- con prompts de intención mixta o reformulaciones adversariales, los métodos sin entrenamiento son más estables.

Resultados relevantes para el rango de menos de 4B: con Llama-3.2-1B las llamadas directas al LLM producen salidas inválidas y no se pudieron evaluar, y NormStat rinde mucho peor que VecStat en identificación de lenguajes de programación (49.21 % frente a 98.99 %). Qwen3-1.7B se emplea en los análisis de calibración y en las ablaciones sobre el número de capas y la longitud máxima del prompt. El RoBERTa-base ajustado obtiene 42.33 % en MATH500, fuera de distribución, lo que sugiere menor generalización que las representaciones de los LLM. Un matiz importante es que las clases del estudio corresponden al enrutamiento de LLM y no a intenciones de diálogo orientado a tareas. El código está disponible públicamente.

### 3.6 Compresión y despliegue

Wang et al. (2024) organizan las técnicas para reducir modelos de lenguaje en poda, destilación, cuantización (post-entrenamiento, PTQ, o con entrenamiento consciente, QAT, p. ej. GPTQ a 4 y 2 bits) y aproximaciones de bajo rango.

Das et al. (2026) estudian poda estructurada, cuantización GGUF y recuperación con LoRA en GPU, CPU y Raspberry Pi 4B (8 GB, con llama.cpp), sobre modelos de Qwen3.5 0.8B, Gemma 3 1B y TinyLlama 1.1B, todos dentro del límite. Las tareas son de preguntas y respuestas y de segmentación (no de intención), pero sus resultados son pertinentes para cualquier clasificador comprimido:

- **Ninguna técnica gana siempre.** Qwen3.5 0.8B con cuantización Q5_K_M alcanza 93.85 de F1 en SQuAD, mientras que la poda estructurada al 1 % cuesta 16 puntos de F1.
- **Reducir el modelo no garantiza menor latencia.** La poda puede inflar el artefacto desplegado entre 21 % y 49 % y multiplicar por hasta 3.4 la latencia en Raspberry Pi.
- **La compresión puede ocultar un colapso.** Una variante recuperada con LoRA conservaba 71 % de exactitud estricta en BoolQ enviando 97 de 100 predicciones a una sola clase (exactitud balanceada de 52.6 %).

## 4. Análisis comparativo

**Tabla 4.** Comparación de los trabajos, restringida al rango de menos de 4B.

| Trabajo | Enfoque | Modelos dentro del límite | Resultado principal | Limitación principal |
|---|---|---|---|---|
| Ganesh et al. (2026) | Zero-shot | 0.14B–3.8B | Qwen2.5-3B-Instruct lidera el rango (0.632); por debajo de 1.5B el rendimiento es bajo | Solo inglés; sin fine-tuning |
| Parikh et al. (2023) | Dominio, aumento y PEFT | Flan-T5-Large (770M), Flan-T5-XL (3B), codificador tipo LaBSE | PEFT con 1–5 ejemplos supera al resto | Datasets en parte privados |
| Alekseev et al. (2025) | AutoML sobre embeddings | Codificadores de oraciones | kNN/regresión logística sobre embeddings: ~93 % medio con entrenamiento barato | No fija el tamaño de los codificadores |
| Vera et al. (2025) | Embeddings compactos | 308M (int4, 128 dim.) | Mejor modelo de <500M en MTEB; robusto a cuantización | No es específico de intención |
| Zhang et al. (2025) | Few-shot con relaciones | BERT, DistilBERT, ALBERT | Superior a IntentBERT y SetFit con 3–20 ejemplos | Requiere sesiones; un dataset privado |
| ElZemity et al. (2026) | Destilación agéntica | 135M, 494M | 94.31 % con datos 100 % sintéticos | Tarea binaria; un dataset de prueba |
| Chen et al. (2026) | Sin entrenamiento vs. entrenado | Llama-3.2-1B, Qwen3-1.7B | Llamadas directas inválidas en 1B; VecStat supera a NormStat | Clases de enrutamiento, no de diálogo |
| Das et al. (2026) | Compresión en el borde | 0.8B–1.1B | Ninguna técnica gana siempre; hay que medir en hardware | No evalúa intención |
| Wang et al. (2024) | Revisión de técnicas | General | Taxonomía de compresión y destilación | Cubre hasta finales de 2024 |

*Las cifras de los distintos trabajos no son directamente comparables entre sí: cada uno usa datasets, protocolos y métricas distintos.*

## 5. Datasets, métricas y protocolos de evaluación

### 5.1 Datasets

| Dataset | Dominio | Nº de intenciones | Referencia |
|---|---|---|---|
| CLINC150 | Asistente general (10 dominios), incluye fuera de dominio | 150 | Larson et al. (2019) |
| Banking77 | Banca | 77 | Casanueva et al. (2020) |
| HWU64 | Asistente doméstico | 64 | Liu et al. (2019) |
| MASSIVE | Asistente de voz, 51 idiomas | 60 | FitzGerald et al. (2022) |
| MTOP | Asistente de Facebook | 117 | Citado en Ganesh et al. (2026) |
| MINDS-14 | Banca por voz, multilingüe | 14 | Gerz et al. (2021) |
| SNIPS | Casa inteligente | 7 | Coucke et al. (2018) |
| ATIS | Viajes aéreos | 20 | Hemphill et al. (1990) |
| HINT3 (Curekart, Powerplay11, Sofmattress) | Comercio electrónico en producción | 20, 53 y 20 | Mehta et al. (2020) |

Larson y Leach (2022) ofrecen un panorama de datasets de intención y extracción de ranuras. Ganesh et al. (2026) citan además IntentGrasp (Yin et al., 2026), un benchmark que reformula 49 datasets como preguntas de opción múltiple y evalúa sobre todo modelos de frontera.

### 5.2 Métricas

La literatura del corpus reporta exactitud, F1 y las métricas siguientes:

- **Exactitud balanceada y F1 por clase.** Das et al. (2026) muestran que la exactitud global puede ocultar el colapso hacia una clase.
- **Calibración (ECE y puntuación de Brier).** Ganesh et al. (2026).
- **F1 de la clase fuera de dominio y exactitud en dominio.** Alekseev et al. (2025).
- **Robustez a perturbaciones** (tipeos, intercambio de palabras) y a entradas de intención mixta o adversariales (Ganesh et al., 2026; Chen et al., 2026).
- **Costos de despliegue:** latencia por muestra, memoria máxima, tamaño del artefacto y energía en el hardware objetivo (Ganesh et al., 2026; Das et al., 2026).

### 5.3 Recomendaciones de protocolo derivadas del corpus

1. **Evaluar en varios datasets** y no depender de uno solo: en MASSIVE, los cinco mejores modelos no se distinguen con la prueba de McNemar (Ganesh et al., 2026).
2. **Retirar o relativizar SNIPS** por saturación, y preferir MASSIVE, MTOP y los datasets de HINT3 (Ganesh et al., 2026).
3. **Interpretar los errores con cautela:** la confusión `play_music` → `music_query` aparece como el error principal en 21 de 41 modelos, lo que sugiere ambigüedad en el esquema de etiquetas y no solo debilidad de los modelos.
4. **Usar decodificación restringida o puntuación de etiquetas** cuando se evalúen modelos pequeños, porque la generación libre puede producir salidas inválidas (Ganesh et al., 2026; Chen et al., 2026).
5. **Reportar variación** entre semillas y entre selecciones de ejemplos few-shot (Parikh et al., 2023, informan desviaciones estándar sobre 5 selecciones aleatorias).
6. **Medir en el hardware objetivo** y con el artefacto ya comprimido (Das et al., 2026).

## 6. Desafíos abiertos

1. **Falta de fine-tuning de decodificadores de 1B a 4B en intención.** Ganesh et al. (2026) evalúan los modelos solo en zero-shot, mientras que el ajuste eficiente se ha estudiado sobre Flan-T5 (Parikh et al., 2023) y no sobre los modelos que lideran el zero-shot, como Qwen2.5-3B o Phi-3.5-mini.
2. **Falta de comparaciones homogéneas.** Embeddings con clasificador, fine-tuning eficiente y destilación sintética se han medido en trabajos distintos, con datasets y presupuestos diferentes.
3. **Idioma.** La evaluación más amplia de modelos de este rango es solo en inglés (Ganesh et al., 2026). MASSIVE y MINDS-14 son multilingües, y EmbeddingGemma se evalúa en más de 250 idiomas, pero sin un estudio específico de intención con modelos de menos de 4B en otros idiomas.
4. **Calibración y fuera de dominio.** Solo AutoIntent integra la detección fuera de dominio y el ajuste de umbrales como parte del flujo. Ganesh et al. (2026) no evalúan abstención por umbral y encuentran una calibración inconsistente.
5. **Compresión en clasificación.** Los estudios de cuantización, poda y recuperación se han hecho en preguntas y respuestas y segmentación. No se ha medido su efecto sobre la exactitud por clase y la calibración de un clasificador de intención.
6. **Dependencia del maestro y validación sintética.** En la destilación agéntica, la calidad varía más de 25 puntos según el maestro y la validación es sintética, por lo que su validez para tareas multiclase queda por demostrar.
7. **Saturación de benchmarks.** Los resultados altos en datasets como SNIPS no discriminan entre modelos.

## 7. Posicionamiento del proyecto

A partir de las brechas anteriores, el proyecto puede formularse alrededor de las siguientes preguntas de investigación, siempre dentro del límite de 4B parámetros:

- **RQ1.** ¿Qué exactitud y qué calibración alcanzan, con el mismo presupuesto de datos, los tres enfoques principales: zero-shot con los mejores modelos del rango (Qwen2.5-3B-Instruct, Phi-3.5-mini), embeddings congelados con clasificador ligero y fine-tuning eficiente (por ejemplo, LoRA o IA3) de modelos de 1B a 4B?
- **RQ2.** ¿Puede el fine-tuning eficiente cerrar o superar la distancia con el mejor zero-shot del estudio (0.660 con un modelo de 7B, fuera del límite) usando solo modelos de menos de 4B?
- **RQ3.** ¿Cómo varía el rendimiento al reducir el tamaño del modelo (de 3.8B a cientos de millones de parámetros) y al cuantizarlo a 8 y 4 bits, medido con exactitud balanceada, F1 por clase, ECE y latencia?
- **RQ4.** ¿Cuántos ejemplos reales por intención son necesarios, y cuánto aportan los datos sintéticos generados por un LLM maestro?
- **RQ5.** ¿Es viable combinar el modelo con un umbral de confianza y un modelo de respaldo para los casos fuera de dominio o dudosos?

Estas preguntas se apoyan en los protocolos de la sección 5.3 y en las líneas base descritas en la sección 3 (AutoIntent, EmbeddingGemma, T-Few y destilación agéntica).

## 8. Conclusiones

Dentro del límite de 4B parámetros, la evidencia del corpus apunta a cinco ideas.

1. **El zero-shot es competitivo solo en la parte alta del rango.** Qwen2.5-3B-Instruct (0.632) y Phi-3.5-mini-Instruct (0.624) quedan a menos de 0.04 puntos del mejor modelo de todo el estudio (0.660, con 7B). Por debajo de 3B, solo Qwen2-1.5B-Instruct supera 0.5, y los modelos de 1B o menos rinden por debajo de 0.3.
2. **El instruction tuning y la calidad del preentrenamiento pesan más que el tamaño** dentro de este rango, por lo que la elección del modelo base importa tanto como su escala.
3. **Con pocos datos, los embeddings compactos con clasificadores ligeros** (EmbeddingGemma, presets «classic» de AutoIntent) son baratos de entrenar, funcionan en CPU y logran en torno al 93 % de exactitud en cinco datasets de intención, aunque esa cifra no es comparable con el agregado zero-shot de Ganesh et al.
4. **La adaptación con adaptadores y la destilación con datos sintéticos** son las rutas con mejor respaldo para mejorar modelos de 0.1B a 3B. Flan-T5-XL (3B) con T-Few supera a otros métodos con un solo ejemplo por intención, y la destilación agéntica lleva a estudiantes de 135M y 494M a más de 86 % y 94 % de exactitud en una tarea binaria.
5. **La cuantización posterior al entrenamiento conserva casi toda la calidad** en los casos estudiados (EmbeddingGemma int4: 60.62 frente a 61.15), siempre que se mida en el hardware real y con métricas que detecten el colapso hacia una clase.

La principal oportunidad de investigación es comparar estas rutas, con el mismo protocolo, sobre los modelos de 1B a 4B, donde hoy solo existe evidencia zero-shot en intención.

## 9. Referencias

Alekseev, I., Solomatin, R., Rustamova, D., y Kuznetsov, D. (2025). *AutoIntent: AutoML for Text Classification*. arXiv:2509.21138. Código: <https://github.com/DeepPavlov/AutoIntent>

Casanueva, I., Temčinas, T., Gerz, D., Henderson, M., y Vulić, I. (2020). Efficient intent detection with dual sentence encoders. *Proceedings of the 2nd Workshop on NLP for Conversational AI*. arXiv:2003.04807.

Chen, N., Yang, Z., y Hayou, S. (2026). *Training-Free versus Training-Based Intent Classification in LLMs: Accuracy, Robustness, and Failure Modes*. arXiv:2608.02415. Código: <https://github.com/Zhouhao-Yang/Training-Free-versus-Training-Based-Intent-Classification-in-LLMs>

Chung, H. W., et al. (2022). *Scaling Instruction-Finetuned Language Models*. arXiv:2210.11416.

Coucke, A., et al. (2018). *Snips Voice Platform: an embedded Spoken Language Understanding system for private-by-design voice interfaces*. arXiv:1805.10190.

Das, S., Cheng, J., Kumar, A., Afrose, S., Han, M., Silagy, M., Palande, S., Soni, B., y Ramnath, R. (2026). *Large Models for Small Devices: Recent Advances and Empirical Analysis of Edge AI Deployment*. arXiv:2608.15693. Código: <https://github.com/Arnavvvkumar/deployment>

ElZemity, A., Sylvester, J., Arief, B., y De Lemos, R. (2026). *Agentic Knowledge Distillation: Autonomous Training of Small Language Models for SMS Threat Detection*. arXiv:2602.10869.

FitzGerald, J., et al. (2022). *MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages*. arXiv:2204.08582.

Ganesh, P., Dozier, G., y Seals, C. (2026). *Selecting Open-Weight Language Models for Zero-Shot Intent Classification: A Systematic Evaluation of 41 Models*. arXiv:2607.27421.

Gerz, D., Su, P.-H., Kusztos, R., Mondal, A., Lis, M., Singhal, E., Mrkšić, N., Wen, T.-H., y Vulić, I. (2021). *Multilingual and Cross-Lingual Intent Detection from Spoken Data*. arXiv:2104.08524.

Hemphill, C. T., Godfrey, J. J., y Doddington, G. R. (1990). The ATIS spoken language systems pilot corpus. *Speech and Natural Language: Proceedings of a Workshop*.

Larson, S., y Leach, K. (2022). *A Survey of Intent Classification and Slot-Filling Datasets for Task-Oriented Dialog*. arXiv:2207.13211.

Larson, S., et al. (2019). An evaluation dataset for intent classification and out-of-scope prediction. *Proceedings of EMNLP-IJCNLP*.

Liu, H., Mahabadi, R. K., et al. (2022). Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. *Advances in Neural Information Processing Systems (NeurIPS)*. arXiv:2205.05638.

Liu, X., Eshghi, A., Swietojanski, P., y Rieser, V. (2019). Benchmarking natural language understanding services for building conversational agents. *International Workshop on Spoken Dialogue Systems (IWSDS)*.

Mehta, G., Jain, C., Chaturvedi, M., y Modi, K. (2020). HINT3: Raising the bar for intent detection in the wild. *Proceedings of the First Workshop on Insights from Negative Results in NLP*.

Tunstall, L., et al. (2022). *Efficient Few-Shot Learning Without Prompts* (SetFit). arXiv:2209.11055.

Vera, H. S., Dua, S., Zhang, B., et al. (2025). *EmbeddingGemma: Powerful and Lightweight Text Representations*. arXiv:2509.20354.

Wang, F., Zhang, Z., Zhang, X., Wu, Z., Mo, T., Lu, Q., Wang, W., Li, R., Xu, J., Tang, X., He, Q., Ma, Y., Huang, M., y Wang, S. (2024). *A Comprehensive Survey of Small Language Models in the Era of Large Language Models: Techniques, Enhancements, Applications, Collaboration with LLMs, and Trustworthiness*. arXiv:2411.03350. Código: <https://github.com/FairyFali/SLMs-Survey>

Yin, Y., Li, C., y Carenini, G. (2026). *IntentGrasp: A Comprehensive Benchmark for Intent Understanding*. arXiv preprint.

Zhang, J., et al. (2021). Few-shot intent detection via contrastive pre-training and fine-tuning. *Proceedings of EMNLP 2021*.

Zhang, L., Li, Y., Zhang, S., Zhang, Z., y Li, X. (2025). *Few-Shot Query Intent Detection via Relation-Aware Prompt Learning*. arXiv:2509.05635.
