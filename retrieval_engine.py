"""
Motor de recuperación híbrida (BM25 + semántico + reranking).
Extrae la lógica de búsqueda y filtrado de rag_hybrid.py.
"""
import heapq
import math
import os
import re
import time
from typing import TYPE_CHECKING, List, Optional

from rich.console import Console

if TYPE_CHECKING:
    from rag_hybrid import HybridRAG


console = Console()

# FASE 4.BIS (v4.1): Lexical Query Expansion exclusiva para BM25.
# No modifica la query para el embedding (BGE-m3 ya es multilingue).
# Complementa (no duplica) a EquivalencesManager: ese cubre acronimos y
# nombres propios (NIST, SQL, CISM); este cubre sustantivos comunes del
# dominio que no tienen cobertura alli (confirmado FASE 1.3/3.1-3.3:
# queries como "nube" o "agente" dan BM25 score=0.000 en todo el corpus).
LEXICAL_EXPANSION_MAP = {
    "nube": ["cloud", "cloud computing"],
    "auditoria": ["audit", "auditing", "assessment"],
    "audito": ["audit", "auditing"],
    "vulnerabilidad": ["vulnerability", "vulnerabilities"],
    "vulnerabilidades": ["vulnerability", "vulnerabilities"],
    "amenaza": ["threat", "threats"],
    "amenazas": ["threat", "threats"],
    "ataque": ["attack", "attacks"],
    "ataques": ["attack", "attacks"],
    "agente": ["agent", "endpoint agent"],
    "ingenieria social": ["social engineering"],
    "cifrado": ["encryption", "cryptography"],
    "autenticacion": ["authentication", "authn"],
    "autorizacion": ["authorization", "authz"],
    "pentester": ["penetration tester", "pentester"],
    "pentest": ["penetration test", "pentest"],
    "escaneo": ["scan", "scanning"],
    "comando": ["command", "commands"],
    "comandos": ["command", "commands"],
    "red": ["network", "networking"],
    "riesgo": ["risk"],
    "riesgos": ["risks", "risk"],
    "gobierno": ["governance"],
    "gobernanza": ["governance"],
    "politica": ["policy", "policies"],
    "politicas": ["policy", "policies"],
    "control": ["control", "controls"],
    "controles": ["control", "controls"],
    "marco": ["framework"],
    "cumplimiento": ["compliance"],
    "incidente": ["incident", "incident response"],
    "incidentes": ["incidents", "incident response"],
    "deteccion": ["detection"],
    "prevencion": ["prevention"],
    "respuesta": ["response"],
    "confianza cero": ["zero trust"],
    "responsabilidad compartida": ["shared responsibility"],
    "infraestructura critica": ["critical infrastructure"],
    "modelo": ["model"],
    "compartida": ["shared"],
    "responsabilidad": ["responsibility"],
}


def expand_query_for_bm25(query: str) -> str:
    """Expande la query con equivalentes en ingles, solo para la rama BM25.
    No modifica la query original (esta funcion se aplica sobre una copia,
    y la query normalizada para embedding no se toca)."""
    query_lower = query.lower()
    added = []
    for es_term, en_terms in LEXICAL_EXPANSION_MAP.items():
        if es_term in query_lower:
            added.extend(en_terms)
    if not added:
        return query
    return query + " " + " ".join(added)


class RetrievalEngine:
    """
    Encapsula la búsqueda híbrida, reranking, filtrado y planificación de retrieval.

    Args:
        rag: Back-reference al HybridRAG orquestador para acceder a embedder,
             vector_store, bm25, reranker, flags, config, doc_roles, etc.
    """

    def __init__(self, rag: "HybridRAG"):
        self._rag = rag

    # ------------------------------------------------------------------
    # Búsqueda principal
    # ------------------------------------------------------------------

    def hybrid_search(self, query: str, top_k: int = 20, semantic_weight: float = 0.6,
                      allowed_sources: List[str] = None) -> list:
        """Búsqueda híbrida: semántica (embeddings) + keyword (BM25)."""
        rag = self._rag
        query_normalized = rag._normalize_query(query)
        keyword_weight = 1 - semantic_weight
        _hs_t0 = time.time()
        query_embedding = rag.embedder.generate_embedding(query)
        _t_embed = time.time() - _hs_t0
        where = None
        if allowed_sources:
            try:
                where = {'source': {'$in': allowed_sources}}
            except Exception:
                where = None
        if allowed_sources:
            sem_top = max(8, min(12, top_k))
            bm25_top = max(30, top_k * 2)
        else:
            sem_top = max(20, top_k * 2)
            bm25_top = max(80, top_k * 4)
        _t_sem_start = time.time()
        sem = rag.vector_store.search(query_embedding.tolist(), top_k=sem_top, where=where)
        _t_semantic = time.time() - _t_sem_start
        sem_scores_idx = {}
        if sem.get('ids'):
            for _id, dist in zip(sem.get('ids', []), sem.get('distances', [])):
                try:
                    idx = rag.id_to_index.get(_id)
                    if idx is not None:
                        sem_scores_idx[idx] = 1 - float(dist)
                except Exception:
                    continue
        _t_bm25_start = time.time()
        # FASE 4.BIS (v4.1): expansion lexica solo para BM25. query_normalized
        # (usado para el embedding arriba) no se toca.
        query_for_bm25 = expand_query_for_bm25(query_normalized)
        query_tokens = rag._tokenize_for_bm25(query_for_bm25)
        bm25_arr = rag.bm25.get_scores(query_tokens)
        _t_bm25 = time.time() - _t_bm25_start
        try:
            import numpy as _np
            bm25_list = _np.asarray(bm25_arr).tolist()
        except Exception:
            bm25_list = list(bm25_arr)
        if allowed_sources and getattr(rag, 'source_to_indices', None):
            allowed_idx = set()
            for s in allowed_sources:
                try:
                    arr = rag.source_to_indices.get(s.lower())
                    if arr:
                        for i in arr:
                            allowed_idx.add(i)
                except Exception:
                    continue
            bm25_candidates = [(i, bm25_list[i]) for i in allowed_idx]
        else:
            bm25_candidates = list(enumerate(bm25_list))
        bm25_top_idx = [i for i, _ in heapq.nlargest(bm25_top, bm25_candidates, key=lambda t: t[1])]
        cand_idx = set(sem_scores_idx.keys()) | set(bm25_top_idx)
        max_bm25 = max((bm25_list[i] for i in cand_idx), default=1.0)
        if max_bm25 <= 0:
            max_bm25 = 1.0
        results = []
        for i in sorted(cand_idx):
            try:
                semantic_score = float(sem_scores_idx.get(i, 0.0))
                keyword_score = float(bm25_list[i]) / max_bm25
                hybrid_score = semantic_weight * semantic_score + keyword_weight * keyword_score
                results.append({
                    'text': rag.all_docs[i],
                    'metadata': rag.all_metadatas[i],
                    'hybrid_score': hybrid_score,
                    'semantic_score': semantic_score,
                    'keyword_score': keyword_score,
                })
            except Exception:
                continue
        _t_fusion = time.time() - _hs_t0 - _t_embed - _t_semantic - _t_bm25
        _hs_timing = {
            't_embed_ms': round(_t_embed * 1000, 1),
            't_semantic_ms': round(_t_semantic * 1000, 1),
            't_bm25_ms': round(_t_bm25 * 1000, 1),
            't_fusion_ms': round(_t_fusion * 1000, 1),
            't_hybrid_total_ms': round((time.time() - _hs_t0) * 1000, 1),
        }
        top = heapq.nlargest(top_k, results, key=lambda x: x['hybrid_score'])
        if top:
            top[0]['_hs_timing'] = _hs_timing
        return top

    def search_in_specific_doc(self, doc_name: str, page: int = None, top_k: int = 10) -> list:
        """Busca contenido de un documento específico por nombre (y opcionalmente página)."""
        doc_name_clean = doc_name.lower().replace('.pdf', '').replace(' ', '')
        all_results = self.hybrid_search(doc_name, top_k=100, semantic_weight=0.1)
        doc_results = []
        for r in all_results:
            source = r['metadata']['source'].lower().replace('.pdf', '').replace(' ', '')
            if doc_name_clean in source or source in doc_name_clean:
                if page is not None:
                    if r['metadata']['page'] == page:
                        doc_results.append(r)
                else:
                    doc_results.append(r)
        if page is not None and len(doc_results) == 0:
            if not hasattr(self._rag, '_page_warning_shown'):
                console.print(f"[dim]Buscando páginas cercanas a la solicitada...[/dim]")
                self._rag._page_warning_shown = True
            for r in all_results:
                source = r['metadata']['source'].lower().replace('.pdf', '').replace(' ', '')
                if doc_name_clean in source or source in doc_name_clean:
                    if abs(r['metadata']['page'] - page) <= 3:
                        doc_results.append(r)
        if page is None and doc_results:
            for r in doc_results:
                if r['metadata']['page'] > 1:
                    r['hybrid_score'] = r.get('hybrid_score', 0.5) * 1.3
        doc_results.sort(key=lambda x: x['metadata']['page'])
        return doc_results[:top_k]

    def search_for_comparison(self, entities: list, top_k: int = 20) -> list:
        """Búsqueda balanceada para comparaciones: garantiza docs de cada entidad."""
        if len(entities) < 2:
            return []
        docs_per_entity = top_k // len(entities)
        combined_results = []
        for entity in entities:
            entity_results = self.hybrid_search(entity, top_k=docs_per_entity * 2, semantic_weight=0.3)
            filtered = [r for r in entity_results if entity.lower() in r['text'].lower()]
            combined_results.extend(filtered[:docs_per_entity])
        combined_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return combined_results[:top_k]

    # ------------------------------------------------------------------
    # Filtrado y boosting
    # ------------------------------------------------------------------

    def filter_by_entity(self, results: list, entities: list, min_matches: int = 1,
                         strict: bool = False) -> list:
        """Filtra resultados para asegurar que contengan las entidades buscadas."""
        if not entities:
            return results
        compound_entities = [e.lower() for e in entities if len(e.split()) >= 3]
        simple_entities = [e.lower() for e in entities if len(e.split()) < 3]
        filtered = []
        for r in results:
            text_lower = r['text'].lower()
            source_lower = r['metadata']['source'].lower()
            matches = 0
            entity_boost = 1.0
            have_rf = False
            try:
                from rapidfuzz import fuzz
                have_rf = True
            except Exception:
                have_rf = False
            for compound in compound_entities:
                comp_vars = {compound, compound.replace(' del ', ' de '), compound.replace(' de ', ' del ')}
                if any((cv in text_lower or cv in source_lower) for cv in comp_vars):
                    matches += 3
                    entity_boost += 1.0
                elif have_rf:
                    try:
                        if max(fuzz.partial_ratio(compound, text_lower),
                               fuzz.partial_ratio(compound, source_lower)) >= 85:
                            matches += 2
                            entity_boost += 0.7
                    except Exception:
                        pass
            for entity in simple_entities:
                ent_vars = {entity, entity.replace(' del ', ' de '), entity.replace(' de ', ' del ')}
                if any((ev in text_lower or ev in source_lower) for ev in ent_vars):
                    matches += 1
                    entity_boost += 0.3
                elif have_rf:
                    try:
                        if max(fuzz.partial_ratio(entity, text_lower),
                               fuzz.partial_ratio(entity, source_lower)) >= 90:
                            matches += 1
                            entity_boost += 0.2
                    except Exception:
                        pass
            r['entity_boost'] = entity_boost
            r['hybrid_score'] = r['hybrid_score'] * r['entity_boost']
            r['entity_matches'] = matches
            if matches >= min_matches:
                filtered.append(r)
        if compound_entities and not filtered:
            console.print(f"[yellow]ADVERTENCIA: No se encontro la entidad especifica: '{compound_entities[0]}'[/yellow]")
            console.print(f"[dim]Verifica el nombre exacto en los documentos[/dim]")
            if strict:
                return []
            console.print("[dim yellow]Filtro de entidad no arrojó resultados. Usando resultados originales sin filtro.[/dim yellow]")
            return results
        filtered.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return filtered

    def filter_to_candidates(self, results: list, allowed_sources: list) -> list:
        """Filtra resultados a un conjunto de fuentes permitidas. Si filtra todo, devuelve original."""
        if not results or not allowed_sources:
            return results
        allowed = set(s.lower() for s in allowed_sources)
        filtered = [r for r in results
                    if (r.get('metadata', {}) or {}).get('source', '').lower() in allowed]
        return filtered if filtered else results

    def deduplicate_results(self, results: list, similarity_threshold: float = 0.85) -> list:
        """Elimina resultados duplicados basándose en similitud de texto."""
        if not results or len(results) < 2:
            return results
        deduplicated = []
        skipped_indices = set()
        for i, r1 in enumerate(results):
            if i in skipped_indices:
                continue
            text1 = r1.get('text', '')
            best_score = r1.get('final_score', r1.get('rerank_score', r1.get('hybrid_score', 0)))
            for j, r2 in enumerate(results[i + 1:], i + 1):
                if j in skipped_indices:
                    continue
                text2 = r2.get('text', '')
                similarity = self._calculate_text_similarity(text1, text2)
                if similarity >= similarity_threshold:
                    score2 = r2.get('final_score', r2.get('rerank_score', r2.get('hybrid_score', 0)))
                    if score2 > best_score:
                        skipped_indices.add(i)
                        break
                    else:
                        skipped_indices.add(j)
            if i not in skipped_indices:
                deduplicated.append(r1)
        if len(deduplicated) < len(results):
            console.print(f"[dim]Deduplicación: {len(results)} → {len(deduplicated)} documentos[/dim]")
        return deduplicated

    def adaptive_quality_filter(self, results: list, question: str) -> list:
        """Aplica filtro de calidad adaptativo basado en la distribución de scores."""
        if not results:
            return results
        scores = []
        for r in results:
            score = r.get('final_score', r.get('rerank_score', r.get('hybrid_score', 0)))
            try:
                scores.append(float(score))
            except Exception:
                pass
        if not scores:
            return results
        max_score = max(scores)
        high_quality_count = len([s for s in scores if s > 0.50])
        if high_quality_count >= 5:
            threshold = 0.40
            console.print(f"[dim]Filtro adaptativo: estricto (≥5 docs >0.50, threshold={threshold})[/dim]")
        elif high_quality_count < 3 and max_score < 0.30:
            threshold = 0.15
            console.print(f"[dim]Filtro adaptativo: relajado (max={max_score:.2f}, threshold={threshold})[/dim]")
        else:
            threshold = 0.25
            console.print(f"[dim]Filtro adaptativo: estándar (threshold={threshold})[/dim]")
        filtered = [r for r in results if r.get('final_score', r.get('rerank_score', 0)) >= threshold]
        if len(filtered) < 3 and len(results) >= 3:
            sorted_results = sorted(results, key=lambda x: x.get('final_score', x.get('rerank_score', 0)), reverse=True)
            filtered = sorted_results[:3]
            console.print(f"[dim]Filtro adaptativo: preservando top 3 documentos[/dim]")
        return filtered if filtered else results

    def limit_results_per_source(self, results: list, max_per_source: int = 2) -> list:
        """Limita el número de resultados por fuente documental."""
        if not results:
            return results
        source_counts = {}
        limited = []
        for r in results:
            source = r.get('metadata', {}).get('source', 'Unknown')
            count = source_counts.get(source, 0)
            if count < max_per_source:
                limited.append(r)
                source_counts[source] = count + 1
        if len(limited) < len(results):
            console.print(f"[dim]Limitación por fuente: {len(results)} → {len(limited)} (máx {max_per_source} por doc)[/dim]")
        return limited

    def diversify_by_source(self, results: list, per_source_limit: int = 1,
                             max_results: int = 50) -> list:
        """Intercala resultados para cubrir múltiples fuentes en top-N."""
        if not results:
            return results
        counts = {}
        diversified = []
        for r in results:
            src = r.get('metadata', {}).get('source', '')
            c = counts.get(src, 0)
            if c < per_source_limit:
                diversified.append(r)
                counts[src] = c + 1
            if len(diversified) >= max_results:
                break
        if len(diversified) < min(len(results), max_results):
            for r in results:
                if r not in diversified:
                    diversified.append(r)
                if len(diversified) >= max_results:
                    break
        return diversified

    def categorize_results(self, results: list) -> list:
        """Categoriza los resultados por tipo de contenido."""
        if not results:
            return results
        definition_patterns = [
            r'\bes\s+(?:un|una|el|la)\s+', r'\bse\s+define\s+como\b', r'\bsignifica\b',
            r'\brefiere\s+a\b', r'\btérmino\b', r'\bconcepto\b', r'\bmarco\s+de\b',
            r'\bestándar\b', r'\bframework\b', r'\bcertificación\b', r'\bprotocolo\b',
        ]
        procedure_patterns = [
            r'\bprocedimiento\b', r'\bpaso\s+\d+\b', r'\bprimero\b', r'\bsegundo\b',
            r'\btercero\b', r'\bluego\b', r'\bdespués\b', r'\bfinalmente\b',
            r'\bimplementar\b', r'\bconfigurar\b', r'\binstalar\b', r'\bejecutar\b',
            r'\bmejores\s+prácticas\b', r'\brecomendado\b', r'\bdebe\s+', r'\bdebería\s+',
        ]
        example_patterns = [
            r'\bejemplo\b', r'\bcaso\s+de\s+uso\b', r'\bpor\s+ejemplo\b',
            r'\btal\s+como\b', r'\bcomo\s+se\s+muestra\b', r'\bilustra\b',
        ]
        for r in results:
            text = r.get('text', '').lower()
            def_score = sum(1 for p in definition_patterns if re.search(p, text))
            proc_score = sum(1 for p in procedure_patterns if re.search(p, text))
            ex_score = sum(1 for p in example_patterns if re.search(p, text))
            if def_score > 0 and def_score >= proc_score:
                category = 'definition'
            elif proc_score > 0:
                category = 'procedure'
            elif ex_score > 0:
                category = 'example'
            else:
                category = 'mention'
            r['content_category'] = category
        category_order = {'definition': 0, 'procedure': 1, 'example': 2, 'mention': 3}
        results.sort(key=lambda x: category_order.get(x.get('content_category', 'mention'), 3))
        return results

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def rerank_results(self, query: str, results: list, top_k: int = 10) -> list:
        """Re-rankea resultados usando CrossEncoder. Combina hybrid_score con rerank_score."""
        rag = self._rag
        if not results:
            return results
        try:
            if rag.flags.get('disable_reranker', False) or not rag.config.get('reranker', {}).get('enabled', True):
                return sorted(results, key=lambda x: x.get('hybrid_score', 0), reverse=True)[:top_k]
        except Exception:
            pass
        if getattr(rag, 'reranker', None) is None:
            try:
                return sorted(results, key=lambda x: x.get('hybrid_score', 0), reverse=True)[:top_k]
            except Exception:
                return results[:top_k]
        try:
            reranker_pool = int(rag.config.get('reranker', {}).get('candidate_pool', 20))
            # v4.1: se removio el cap oculto min(reranker_pool, top_k+10) que ignoraba
            # el candidate_pool de config.yaml cuando top_k era pequeno (ej. top_k=10
            # capaba el pool a 20 sin importar el valor configurado). Ahora respeta
            # candidate_pool directamente, con piso de top_k para nunca rankear menos
            # candidatos de los que se van a devolver.
            candidate_count = min(len(results), max(reranker_pool, top_k))
            candidates = results[:candidate_count]
            console.print(f"[dim]Reranker pool: {len(candidates)} candidatos[/dim]")
        except Exception:
            candidates = results[:min(len(results), 20)]
        try:
            pairs = [[query, (doc.get('text', '') or '')[:1024]] for doc in candidates]
        except Exception:
            return results[:top_k]
        try:
            console.print(f"[dim]Re-rankeando top-{len(candidates)} resultados...[/dim]")
            bs = 12
            try:
                bs = int(rag.config.get('reranker', {}).get('batch_size', bs))
            except Exception:
                pass
            bs = min(bs, 16)
            console.print(f"[dim]Reranker batch_size={bs} (ejecución directa)[/dim]")
            _t_rerank_start = time.time()
            rerank_scores_raw = rag.reranker.predict(
                pairs, batch_size=bs, show_progress_bar=False, convert_to_numpy=True
            )
            rerank_scores = [float(s) for s in rerank_scores_raw]
            _t_rerank_ms = round((time.time() - _t_rerank_start) * 1000, 1)
            console.print(f"[dim]Re-ranking completado: {len(rerank_scores)} scores ({_t_rerank_ms}ms)[/dim]")
            if candidates:
                candidates[0]['_t_rerank_ms'] = _t_rerank_ms
        except Exception:
            try:
                return sorted(results, key=lambda x: x.get('hybrid_score', 0), reverse=True)[:top_k]
            except Exception:
                return results[:top_k]
        try:
            s_min = min(rerank_scores)
            s_max = max(rerank_scores)
            s_avg = sum(rerank_scores) / len(rerank_scores)
            if s_max < 0.1:
                rerank_norm = [max(0.0, min(1.0, s / 0.1)) for s in rerank_scores]
                console.print(f"[dim]Reranker: max score bajo ({s_max:.3f}), posible consulta fuera de dominio[/dim]")
            else:
                rerank_norm = [1.0 / (1.0 + math.exp(-5 * s)) for s in rerank_scores]
            console.print(f"[dim]Reranker stats: min={s_min:.3f}, max={s_max:.3f}, avg={s_avg:.3f}[/dim]")
        except Exception:
            rerank_norm = [0.5] * len(rerank_scores)
        try:
            mix = rag.config.get('reranker', {}).get('mix', {}) or {}
            hybrid_w = float(mix.get('hybrid_weight', 0.3))
            rerank_w = float(mix.get('rerank_weight', 0.7))
            if hybrid_w + rerank_w <= 0:
                hybrid_w, rerank_w = 0.3, 0.7
        except Exception:
            hybrid_w, rerank_w = 0.3, 0.7
        ranking_strategy = os.environ.get('RERANKER_RANKING_STRATEGY', str(rag.config.get('reranker', {}).get('ranking_strategy', 'blend'))).lower()
        rrf_k = float(os.environ.get('RERANKER_RRF_K', rag.config.get('reranker', {}).get('rrf_k', 60)))
        if rrf_k <= 0:
            rrf_k = 60.0
        rerank_order = sorted(range(len(candidates)), key=lambda idx: rerank_scores[idx], reverse=True)
        rerank_ranks = {idx: rank + 1 for rank, idx in enumerate(rerank_order)}
        for i, doc in enumerate(candidates):
            try:
                doc['rerank_score'] = rerank_scores[i]
                doc['rerank_norm'] = float(rerank_norm[i])
                if ranking_strategy == 'rrf':
                    doc['final_score'] = (1.0 / (rrf_k + i + 1)) + (1.0 / (rrf_k + rerank_ranks[i]))
                else:
                    doc['final_score'] = (doc.get('hybrid_score', 0.0) * hybrid_w) + (doc['rerank_norm'] * rerank_w)
            except Exception:
                doc['final_score'] = doc.get('hybrid_score', 0.0)
        try:
            ranked = sorted(candidates, key=lambda x: x.get('final_score', 0), reverse=True)[:top_k]
            preserve_enabled = bool(rag.config.get('reranker', {}).get('preserve_prerank_top_k', True))
            if not preserve_enabled:
                return ranked
            prerank_keys = {
                ((doc.get('metadata', {}) or {}).get('source', '').lower(),
                 (doc.get('metadata', {}) or {}).get('page', 0))
                for doc in results[:top_k]
            }
            preserved = [
                doc for doc in ranked
                if ((doc.get('metadata', {}) or {}).get('source', '').lower(),
                    (doc.get('metadata', {}) or {}).get('page', 0)) in prerank_keys
            ]
            preserved_keys = {
                ((doc.get('metadata', {}) or {}).get('source', '').lower(),
                 (doc.get('metadata', {}) or {}).get('page', 0))
                for doc in preserved
            }
            for original in results[:top_k]:
                metadata = original.get('metadata', {}) or {}
                key = (metadata.get('source', '').lower(), metadata.get('page', 0))
                if key not in preserved_keys:
                    preserved.append(original)
                    preserved_keys.add(key)
            return preserved[:top_k]
        except Exception:
            return results[:top_k]

    # ------------------------------------------------------------------
    # Cobertura de fuentes
    # ------------------------------------------------------------------

    def ensure_source_for_entity(self, results: list, source_substr: str,
                                  entity_substr: str, limit: int = 1) -> list:
        """Garantiza al menos un resultado de source_substr que mencione entity_substr."""
        rag = self._rag
        if not results:
            return results
        source_substr = source_substr.lower()
        entity_substr = entity_substr.lower()
        for r in results[:10]:
            src = r.get('metadata', {}).get('source', '').lower()
            if source_substr in src and entity_substr in r.get('text', '').lower():
                return results
        try:
            col = rag.vector_store.collection.get()
            present_ids = set(r.get('id') for r in results)
            added = []
            for i, md in enumerate(col['metadatas']):
                src = md.get('source', '').lower()
                if source_substr in src:
                    txt = col['documents'][i]
                    if entity_substr in txt.lower():
                        rid = col['ids'][i]
                        if rid in present_ids:
                            continue
                        added.append({
                            'text': txt, 'metadata': md,
                            'hybrid_score': 0.9, 'rerank_score': 0.0,
                            'final_score': 0.85, 'id': rid,
                        })
                        if len(added) >= limit:
                            break
            if added:
                return added + results
        except Exception:
            pass
        return results

    def ensure_sources(self, results: list, source_substrings: list,
                       per_source_limit: int = 1) -> list:
        """Incluye fragmentos de fuentes clave aunque no mencionen la entidad."""
        rag = self._rag
        if not source_substrings:
            return results
        try:
            col = rag.vector_store.collection.get()
            present_ids = set((r.get('id') or '') for r in (results or []))
            added = []
            for src_sub in source_substrings:
                count = 0
                for i, md in enumerate(col.get('metadatas', [])):
                    src = (md or {}).get('source', '')
                    if src and (src_sub.lower() in src.lower()):
                        rid = col['ids'][i]
                        if rid in present_ids:
                            continue
                        txt = col['documents'][i]
                        added.append({
                            'text': txt, 'metadata': md,
                            'hybrid_score': 0.75, 'rerank_score': 0.0,
                            'final_score': 0.72, 'id': rid, 'priority_boost': 0.3,
                        })
                        present_ids.add(rid)
                        count += 1
                        if count >= per_source_limit:
                            break
            if added:
                results = added + (results or [])
        except Exception:
            pass
        return results

    # ------------------------------------------------------------------
    # Planificación de retrieval
    # ------------------------------------------------------------------

    def plan_retrieval(self, question: str, entities: list, is_conceptual: bool,
                       is_procedural: bool, is_direct_comparison: bool = False,
                       is_simple_numeric: bool = False,
                       is_troubleshooting: bool = False) -> dict:
        """Planner: selecciona roles preferidos, atributo y candidatos por DocCards."""
        rag = self._rag
        try:
            attribute = None
            try:
                attribute = rag.conceptual_map._extract_attribute(question)
            except Exception:
                pass
            is_protection_query = False
            try:
                ql = question.lower()
                protection_keywords = [
                    'proteccion', 'protección', 'protecciones', 'ansi',
                    'relé', 'rele', 'relay', 'sobrecorriente', 'sobre-corriente',
                ]
                is_protection_query = any(k in ql for k in protection_keywords)
            except Exception:
                pass
            preferred = []
            if is_protection_query and entities:
                preferred = ['grid_ops', 'entity_profile', 'procedure']
            elif is_conceptual:
                preferred = ['analysis_report', 'entity_profile', 'manual_scada']
            elif is_procedural or is_troubleshooting:
                preferred = ['procedure', 'manual_scada', 'analysis_report']
            elif rag._is_cells_query(question):
                preferred = ['manual_scada', 'entity_profile']
            elif is_direct_comparison:
                preferred = ['entity_profile', 'analysis_report']
            elif is_simple_numeric:
                preferred = ['analysis_report', 'entity_profile']
            elif entities:
                preferred = ['entity_profile', 'entity_list']
            else:
                preferred = ['entity_list', 'analysis_report', 'other']
            candidates = []
            if rag.config.get('use_doc_roles', True) and isinstance(rag.doc_roles, dict) and rag.doc_roles.get('docs'):
                try:
                    from doc_cards import select_docs_by_roles
                    candidates = select_docs_by_roles(
                        rag.doc_roles, preferred_roles=preferred,
                        entities=entities or [], attribute=attribute, limit=60,
                    )
                except Exception:
                    candidates = []
            return {'doc_roles_preferred': preferred, 'attribute': attribute, 'candidate_docs': candidates}
        except Exception:
            return {'doc_roles_preferred': [], 'attribute': None, 'candidate_docs': []}

    # ------------------------------------------------------------------
    # Utilidad interna
    # ------------------------------------------------------------------

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Similitud de Jaccard entre dos textos (fallback sin embeddings)."""
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            return len(words1 & words2) / len(words1 | words2)
        except Exception:
            return 0.0
