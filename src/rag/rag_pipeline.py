import os
import json
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import (
    RAG_INDEX_DIR, 
    RAG_INDEX_PATH, 
    RAG_METADATA_PATH, 
    RAG_INFO_PATH, 
    SENTENCE_TRANSFORMER_NAME,
    RAW_DATA_DIR
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

class RagIndex:
    def __init__(
        self,
        index_dir: Optional[str] = None,
        model_name: str = SENTENCE_TRANSFORMER_NAME
    ):
        # Si se pasa index_dir, se usa; si no, se usa el default de config
        self.index_dir = Path(index_dir) if index_dir else RAG_INDEX_DIR
        self.index_path = self.index_dir / "faiss.index"
        self.meta_path = self.index_dir / "metadata.json"
        self.info_path = self.index_dir / "index_info.json"
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict] = []

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

        # dedupe
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

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FactChecker, cls).__new__(cls)
            cls._instance.rag = RagIndex()
            cls._instance.llm_pipeline = None
            cls._instance.tokenizer = None
            cls._instance.model = None
        return cls._instance

    def _ensure_llm(self):
        if self.llm_pipeline is None:
            model_name = "google/flan-t5-base"
            logger.info(f"Cargando LLM para FactChecker: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.llm_pipeline = pipeline(
                "text2text-generation", 
                model=self.model, 
                tokenizer=self.tokenizer,
                max_length=512
            )

    def check(self, claim: str) -> Dict:
        """
        1. Busca en RAG (retrieval).
        2. Si no hay contexto relevante o vacío, retorna 'unverified'.
        3. Si hay contexto, usa LLM para verificar.
        """
        # 1. Retrieval
        retrieved = self.rag.query(claim, top_k=3)
        if not retrieved:
            return {
                "analysis": "No se encontró información relevante en la base de conocimientos.",
                "sources": [],
                "analysis_type": "heuristic"
            }
            
        # 2. Context building
        context_texts = [r["text"][:300] for r in retrieved] # Truncar cada fuente
        context_block = "\n".join(f"- {t}" for t in context_texts)
        
        # 3. LLM Generation
        self._ensure_llm()
        assert self.llm_pipeline is not None
        
        prompt = (
            f"Context:\n{context_block}\n\n"
            f"Claim: {claim[:600]}\n\n"
            "Task: Based ONLY on the context above, explain if the claim is supported or contradicted. "
            "If the context is irrelevant, say 'Not enough information'."
        )
        
        # Parámetros para reducir repetición
        out = self.llm_pipeline(
            prompt, 
            max_length=200, 
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2
        )
        analysis = out[0]["generated_text"]
        
        return {
            "analysis": analysis,
            "sources": retrieved,
            "analysis_type": "llm_rag"
        }

def rag_fact_check(text: str) -> Dict:
    """
    Función helper para usar el FactChecker singleton.
    Compatible con llamadas anteriores que esperaban una función.
    """
    checker = FactChecker()
    return checker.check(text)
