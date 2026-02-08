import os
import json
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from ddgs import DDGS

from src.config import (
    RAG_INDEX_DIR, 
    RAG_INDEX_PATH, 
    RAG_METADATA_PATH, 
    RAG_INFO_PATH, 
    SENTENCE_TRANSFORMER_NAME,
    RAG_LLM_NAME,
    RAG_TOP_K,
    RAG_GEN_PARAMS,
    RAG_PROMPT_TEMPLATE,
    TOKENIZER_MAX_LENGTH,
    RAG_CONTEXT_MAX_LENGTH,
    RAG_CLAIM_MAX_LENGTH,
    RAG_VERDICT_REAL_KEYWORDS,
    RAG_VERDICT_FAKE_KEYWORDS
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

class WebRetriever:
    """
    Recuperador de información basado en búsqueda web (DuckDuckGo).
    Reemplaza al RagIndex local cuando se requiere acceso a internet.
    """
    def __init__(self):
        self.ddgs = DDGS()

    def query(self, text: str, top_k: int = RAG_TOP_K) -> List[Dict]:
        results = []
        try:
            logger.info(f"Buscando en web: {text[:50]}...")
            # duckduckgo-search devuelve una lista de diccionarios con claves: title, href, body
            # Nota: DDGS.text devuelve un generador o lista dependiendo de la versión, forzamos lista
            gen = self.ddgs.text(text, max_results=top_k)
            search_results = list(gen)
            
            if search_results:
                for i, r in enumerate(search_results):
                    # Combinar título y cuerpo para dar más contexto al LLM
                    content = f"{r.get('title', '')}: {r.get('body', '')}"
                    results.append({
                        "text": content,
                        "source": r.get('href', ''),
                        "score": 1.0 / (i + 1)  # Score heurístico basado en rango (1.0, 0.5, 0.33...)
                    })
            else:
                logger.warning("DuckDuckGo no devolvió resultados.")
        except Exception as e:
            logger.error(f"Error en búsqueda web DuckDuckGo: {e}")
            raise e
            
        return results

class RagIndex:
    def __init__(
        self,
        index_dir: Optional[str] = None,
        model_name: str = SENTENCE_TRANSFORMER_NAME
    ):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict] = []

        # Si se pasa index_dir, se construyen rutas dinámicas
        if index_dir:
            self.index_dir = Path(index_dir)
            self.index_path = self.index_dir / "faiss.index"
            self.meta_path = self.index_dir / "metadata.json"
            self.info_path = self.index_dir / "index_info.json"
        else:
            # Si no, usamos las rutas definidas en config
            self.index_dir = RAG_INDEX_DIR
            self.index_path = RAG_INDEX_PATH
            self.meta_path = RAG_METADATA_PATH
            self.info_path = RAG_INFO_PATH

    def _ensure_model(self):
        if self.model is None:
            logger.info(f"Cargando modelo de embeddings: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms

    def build_from_csvs(self, csv_paths: List[str], text_col: str = "text", min_len: int = 50) -> Dict:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_model()
        assert self.model is not None

        texts: List[str] = []
        sources: List[str] = []
        for p in csv_paths:
            logger.info(f"Procesando CSV: {p}")
            df = pd.read_csv(p)
            if text_col not in df.columns:
                logger.warning(f"Columna {text_col} no encontrada en {p}")
                continue
            col = df[text_col].astype(str).str.strip()
            col = col[col.str.len() >= min_len]
            texts.extend(col.tolist())
            sources.extend([Path(p).name] * len(col))


        unique = {}
        for t, s in zip(texts, sources):
            if t not in unique:
                unique[t] = s
        texts = list(unique.keys())
        sources = [unique[t] for t in texts]

        if len(texts) == 0:
            raise ValueError("No hay textos válidos para indexar")

        logger.info(f"Generando embeddings para {len(texts)} textos...")
        emb = self.model.encode(texts, batch_size=64, show_progress_bar=True)
        emb = np.asarray(emb, dtype=np.float32)
        emb = self._normalize(emb)

        dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb)

        self.metadata = [{"text": t, "source": s} for t, s in zip(texts, sources)]
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.info_path, "w", encoding="utf-8") as f:
            json.dump({"model_name": self.model_name, "dim": dim, "count": len(texts)}, f, indent=2)
        
        logger.info(f"Índice construido y guardado en {self.index_dir}")
        return {"count": len(texts), "dim": dim, "index_path": str(self.index_path)}

    def load(self):
        self._ensure_model()
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("Índice RAG no encontrado. Construye primero el índice.")
        
        logger.info(f"Cargando índice RAG desde {self.index_dir}")
        self.index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        return self

    def query(self, text: str, top_k: int = 5) -> List[Dict]:
        if self.index is None or self.model is None or not self.metadata:
            self.load()
        assert self.model is not None, "Model not loaded"
        
        q = self.model.encode([text])
        q = np.asarray(q, dtype=np.float32)
        q = self._normalize(q)
        
        assert self.index is not None, "Index not loaded"
        scores, idxs = self.index.search(q, top_k)
        
        result: List[Dict] = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0 or idx >= len(self.metadata):
                continue
            m = self.metadata[idx]
            result.append({"text": m.get("text", ""), "source": m.get("source", ""), "score": float(score)})
        return result


class FactChecker:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FactChecker, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            logger.info("Inicializando FactChecker (Singleton)...")
            self.rag = WebRetriever()
            self.llm_pipeline = None
            self.tokenizer = None
            self.model = None
            self._initialized = True

    def _ensure_llm(self):
        if self.llm_pipeline is None:
            model_name = RAG_LLM_NAME
            logger.info(f"Cargando LLM para FactChecker: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.llm_pipeline = pipeline(
                "text2text-generation", 
                model=self.model, 
                tokenizer=self.tokenizer,
                max_length=TOKENIZER_MAX_LENGTH
            )

    def check(self, claim: str) -> Dict:
        """
        1. Busca en Internet (WebRetriever).
        2. Si no hay contexto relevante o vacío, retorna 'unverified'.
        3. Si hay contexto, usa LLM para verificar.
        """
        # 1. Retrieval
        # Usamos una versión truncada y limpia del claim para la búsqueda web
        # DuckDuckGo funciona mejor con queries cortas (<200 chars)
        search_query = claim[:RAG_CLAIM_MAX_LENGTH].replace("\n", " ").strip()
        retrieved = self.rag.query(search_query, top_k=RAG_TOP_K)
        if not retrieved:
            return {
                "analysis": "No se encontró información relevante en internet.",
                "sources": [],
                "analysis_type": "heuristic"
            }
            
        # 2. Context building
        # Aumentamos el límite de caracteres para el contexto ya que los snippets web pueden ser densos
        context_texts = [r["text"][:RAG_CONTEXT_MAX_LENGTH] for r in retrieved] 
        context_block = "\n".join(f"- {t}" for t in context_texts)
        
        # 3. LLM Generation
        self._ensure_llm()
        assert self.llm_pipeline is not None
        
        # Truncamos el claim para evitar overflow de tokens en FLAN-T5 (512 max)
        prompt = RAG_PROMPT_TEMPLATE.format(context=context_block, claim=claim[:RAG_CLAIM_MAX_LENGTH])
        
        # Parámetros para reducir repetición
        out = self.llm_pipeline(prompt, **RAG_GEN_PARAMS)
        analysis = out[0]["generated_text"]
        
        # Parsear veredicto explícito basado en el prompt
        analysis_lower = analysis.lower()
        
        # Verificar keywords
        is_real = any(k in analysis_lower for k in RAG_VERDICT_REAL_KEYWORDS)
        is_fake = any(k in analysis_lower for k in RAG_VERDICT_FAKE_KEYWORDS)
        
        # Prioridad: si dice "yes" o "supported", es REAL. Si dice "no" o "contradicted", es FAKE.
        # Cuidado con falsos positivos tipo "not supported" (contiene "supported").
        # Pero con "yes"/"no" es más limpio.
        
        if "yes" in analysis_lower.split() or "supported" in analysis_lower:
             verdict = "REAL"
        elif "no" in analysis_lower.split() or "contradicted" in analysis_lower:
             verdict = "FAKE"
        else:
             verdict = "UNCERTAIN"
        
        return {
            "analysis": analysis,
            "verdict": verdict,
            "sources": retrieved,
            "analysis_type": "llm_rag_web"
        }

def rag_fact_check(text: str) -> Dict:
    """
    Función helper para usar el FactChecker singleton.
    Compatible con llamadas anteriores que esperaban una función.
    """
    checker = FactChecker()
    return checker.check(text)
