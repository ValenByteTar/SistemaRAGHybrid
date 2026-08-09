# Suite de evaluacion RAG - Ciberseguridad

## Archivos

| Archivo | Descripcion |
|---------|-------------|
| `cybersec_eval_questions.json` | Dataset curado: 75 preguntas con verdad de terreno a nivel pagina |
| `run_cybersec_eval.py` | Harness que consulta la API y genera reportes JSON + Markdown |
| `build_ground_truth.py` | Utilidad read-only para anclar `(source, page)` en Chroma |
| `reports/` | Directorio de salida de reportes |

## Pre-requisitos

1. La re-ingesta del corpus debe estar completa: la coleccion `cybersec_docs_bge_m3` debe estar poblada.
   ```
   .venv\Scripts\python.exe build_rag_system.py --variant bge --rebuild
   ```
2. El servidor debe estar corriendo:
   ```
   .venv\Scripts\python.exe web_app.py
   ```

## Ejecucion

### Suite completa
```
.venv\Scripts\python.exe tests/eval/run_cybersec_eval.py
```

### Subconjunto por categoria
```
.venv\Scripts\python.exe tests/eval/run_cybersec_eval.py --category no_answer
.venv\Scripts\python.exe tests/eval/run_cybersec_eval.py --category simple
.venv\Scripts\python.exe tests/eval/run_cybersec_eval.py --category multi_document
.venv\Scripts\python.exe tests/eval/run_cybersec_eval.py --category complex
.venv\Scripts\python.exe tests/eval/run_cybersec_eval.py --category ambiguous
```

### Por IDs especificos
```
.venv\Scripts\python.exe tests/eval/run_cybersec_eval.py --ids 1,5,31,41
```

### Primeras N preguntas (modo rapido)
```
.venv\Scripts\python.exe tests/eval/run_cybersec_eval.py --limit 15
```

## Distribucion del dataset

| Categoria | Cantidad | Descripcion |
|-----------|----------|-------------|
| simple | 30 | Preguntas de un solo documento, factual |
| multi_document | 11 | Requieren combinar 2-3 fuentes |
| no_answer | 13 | La respuesta no esta en el corpus (chequeo de alucinacion) |
| ambiguous | 9 | Preguntas vagas o de doble interpretacion |
| complex | 12 | Largas, requieren sintesis profunda |

## Metricas reportadas

| Metrica | Descripcion |
|---------|-------------|
| Recuperacion doc (hit@5) | El documento esperado aparece entre las 5 fuentes devueltas por la API |
| Recuperacion pag (hit@5 +/-tol) | Ademas, la pagina esta dentro de la tolerancia (+/-2 por defecto) |
| Fidelidad de citas | Marcadores `[Doc N - fuente p.X]` en la respuesta corresponden a fuentes recuperadas |
| Alucinaciones | En preguntas `is_answerable=false`, el modelo declino sin inventar |
| Keyword score | Fraccion de `answer_keywords` presentes en la respuesta |

## Interpretacion de resultados

El reporte distingue dos tipos de fallo:

- **Fallo de recuperacion (`retrieval_doc_miss`)**: el retriever no trajo el documento correcto.
  Intervencion: ajustar `top_k`, `score_threshold`, reranker, o metadata del chunk.

- **Fallo del LLM (`missing_keywords`)**: el retriever trajo el documento pero el LLM no uso
  la informacion correctamente.
  Intervencion: revisar el prompt, la evidencia gate, o el contexto construido.

Esto permite separar problemas de recuperacion de problemas del modelo de lenguaje.

## Nucleo determinista vs capa generativa

La evaluacion tiene dos capas claramente separadas:

- **Nucleo determinista (retrieval)**: Recall@1/3/5, MRR, Document Hit Rate, Page Hit Rate y Precision@K se calculan exclusivamente a partir de la lista de fuentes recuperadas contra el ground truth `(source, page)`. No dependen del LLM. Estas metricas son reproducibles y auditable.
- **Capa generativa (end-to-end)**: keyword score, fidelidad de citas, anti-alucinacion y aprobacion final dependen de la respuesta del LLM. Son metricas auxiliares y estan sujetas a la variabilidad del modelo.

El `run_retrieval_only.py` ejecuta unicamente el nucleo determinista (sin LLM), util para iterar en retrieval sin el costo de generacion.

## Integridad de resultados historicos

Los reportes en `reports/` son artefactos historicos inmutables. No se modifican las preguntas, ground truth, logica de evaluacion ni reportes existentes para mejorar los numeros reportados. Cualquier cambio de metodologia debe versionarse como una nueva evaluacion.

## Tolerancia de pagina

Por defecto `page_tolerance=2`. Se puede ajustar por linea de comando:
```
.venv\Scripts\python.exe tests/eval/run_cybersec_eval.py --tolerance 5
```

## Utilidad de ancla de paginas

Para verificar en que pagina esta un fragmento de texto especifico:
```
.venv\Scripts\python.exe tests/eval/build_ground_truth.py --search "never trust always verify"
.venv\Scripts\python.exe tests/eval/build_ground_truth.py --list-sources --limit 30
```
