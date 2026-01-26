import os
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import json
import torch
from preprocessing.clean_text import clean_text
from explainability.explainers import (
    explain_with_lime,
    explain_with_shap,
    explain_with_attention,
)

_CACHED_DIR: Optional[str] = None
_CACHED_CLF = None
_CACHED_TOKENIZER = None
_CACHED_MODEL = None

class ArticleExtractionError(Exception):
    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage

def _label_names_from_config(model) -> list[str]:
    id2label = getattr(model.config, "id2label", None)
    if isinstance(id2label, dict) and len(id2label) >= 2:
        return [id2label.get(0, "FAKE"), id2label.get(1, "REAL")]
    return ["FAKE", "REAL"]


def _default_model_dir() -> str:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    candidate = os.path.join(base_dir, "models", "baseline")
    return os.environ.get("MODEL_DIR", candidate)


def _load_pipeline(model_dir: Optional[str] = None):
    model_dir = model_dir or _default_model_dir()
    global _CACHED_DIR, _CACHED_CLF, _CACHED_TOKENIZER, _CACHED_MODEL
    if _CACHED_CLF is not None and _CACHED_DIR == model_dir:
        return _CACHED_CLF, _CACHED_TOKENIZER, _CACHED_MODEL
    adapter_cfg = os.path.join(model_dir, "adapter_config.json")
    if os.path.isfile(adapter_cfg):
        try:
            with open(adapter_cfg, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
        base_name = cfg.get("base_model_name_or_path", os.environ.get("MODEL_NAME", "microsoft/deberta-v3-base"))
        tokenizer = AutoTokenizer.from_pretrained(base_name)
        model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, model_dir)
        except Exception:
            pass
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
        except Exception:
            base_name = os.environ.get("MODEL_NAME", "distilbert-base-uncased")
            tokenizer = AutoTokenizer.from_pretrained(base_name)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        except Exception:
            base_name = os.environ.get("MODEL_NAME", "distilbert-base-uncased")
            model = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
    clf = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        truncation=True
    )
    _CACHED_DIR = model_dir
    _CACHED_CLF = clf
    _CACHED_TOKENIZER = tokenizer
    _CACHED_MODEL = model
    return _CACHED_CLF, _CACHED_TOKENIZER, _CACHED_MODEL


def extract_article_from_url(url: str) -> Dict[str, str]:
    """
    Extrae título y texto de un artículo dado un URL.
    Lanza ValueError si el texto es demasiado corto.
    """
    from newspaper import Article
    article = Article(url, language="es")
    try:
        article.download()
    except Exception as e:
        raise ArticleExtractionError("download", str(e))
    try:
        article.parse()
    except Exception as e:
        raise ArticleExtractionError("parse", str(e))
    if not article.text or len(article.text) < 500:
        raise ArticleExtractionError("too_short", "Artículo demasiado corto")
    return {"title": article.title or "", "text": article.text}


def _label_map(idx: int) -> str:
    return "FAKE" if idx == 0 else "REAL"


def model_predict(text: str, model_dir: Optional[str] = None) -> Dict:
    clf, tokenizer, model = _load_pipeline(model_dir)
    scores = clf(text)[0]
    idx = int(max(range(len(scores)), key=lambda i: scores[i]["score"]))
    conf = float(scores[idx]["score"])
    class_names = _label_names_from_config(model)
    label = class_names[idx] if 0 <= idx < len(class_names) else _label_map(idx)
    return {"label": label, "confidence": round(conf, 4)}


def generate_explanation(text: str, method: str = "lime", model_dir: Optional[str] = None) -> Dict:
    """
    Genera una explicación para el texto usando el método indicado.
    Soporta: "lime", "shap", "attention".
    Aplica truncado básico para textos muy largos.
    """
    clf, tokenizer, model = _load_pipeline(model_dir)
    enc = tokenizer(text, truncation=True, max_length=512, return_offsets_mapping=True)
    offsets = enc.get("offset_mapping", None)
    if offsets:
        last_end = 0
        for off in offsets:
            if off is not None:
                last_end = max(last_end, off[1])
        text = text[:last_end] if last_end > 0 else text
    if method == "lime":
        return explain_with_lime(text, model_dir=model_dir, clf=clf, tokenizer=tokenizer, model=model)
    if method == "shap":
        return explain_with_shap(text, model_dir=model_dir, clf=clf, tokenizer=tokenizer, model=model)
    if method == "attention":
        return explain_with_attention(text, model_dir=model_dir, clf=clf, tokenizer=tokenizer, model=model)
    return explain_with_lime(text, model_dir=model_dir, clf=clf, tokenizer=tokenizer, model=model)


def predict(input_data: Dict, method: str = "lime", model_dir: Optional[str] = None) -> Dict:
    """
    Orquesta la predicción y la explicación.
    Entradas:
      - input_data: {"type": "text"|"url", "content": "..."}
      - method: "lime" | "shap" | "attention"
    Salida:
      - {"label", "confidence", "explanation": {"top_words","top_word_scores","sentence_contributions"}, "extracted_title"?}
    """
    content_type = str(input_data.get("type", "text")).lower()
    content = str(input_data.get("content", "")).strip()
    extracted_title = ""

    if content_type == "url":
        try:
            extracted = extract_article_from_url(content)
        except ArticleExtractionError as e:
            return {"type": content_type, "content": content, "error_stage": e.stage, "error": str(e)}
        except Exception as e:
            return {"type": content_type, "content": content, "error_stage": "unknown", "error": str(e)}
        extracted_title = extracted.get("title", "")
        text = extracted.get("text", "")
    else:
        text = content

    text = clean_text(text)
    result = model_predict(text, model_dir=model_dir)
    explanation = generate_explanation(text, method=method, model_dir=model_dir)
    out = {
        "label": result["label"],
        "confidence": result["confidence"],
        "explanation": {
            "top_words": explanation.get("top_words", []),
            "top_word_scores": explanation.get("top_word_scores", []),
            "sentence_contributions": explanation.get("sentence_contributions", [])
        }
    }
    if extracted_title:
        out["extracted_title"] = extracted_title
    return out
