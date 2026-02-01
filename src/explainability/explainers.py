import math
from typing import List, Dict, Tuple, Optional
import numpy as np
import shap
from lime.lime_text import LimeTextExplainer
import torch
from src.inference.predict import load_text_clf_pipeline, label_names_from_config


def explain_with_lime(text: str, top_k: int = 10, model_dir: Optional[str] = None, clf=None, tokenizer=None, model=None) -> Dict:
    if clf is None or tokenizer is None or model is None:
        clf, tokenizer, model = load_text_clf_pipeline(model_dir)
    class_names = label_names_from_config(model)

    def predict_proba(texts: List[str]) -> np.ndarray:
        out = clf(texts)
        arr = []
        for row in out:
            arr.append([c["score"] for c in row])
        return np.array(arr, dtype=float)

    enc = tokenizer(text, truncation=True, max_length=512, return_offsets_mapping=True)
    offsets = enc.get("offset_mapping", None)
    if offsets:
        last_end = 0
        for off in offsets:
            if off is not None:
                last_end = max(last_end, off[1])
        text = text[:last_end] if last_end > 0 else text
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, predict_proba, num_features=top_k)

    words = text.split()
    weights = np.zeros(len(words), dtype=float)
    class_idx = next((i for i, n in enumerate(class_names) if str(n).upper() == "REAL"), 1 if len(class_names) > 1 else 0)
    for idx, w in exp.as_map()[class_idx]:
        if 0 <= idx < len(weights):
            weights[idx] += w

    top_indices = np.argsort(-np.abs(weights))[:top_k]
    top_words = [words[i] for i in top_indices]
    top_values = [float(weights[i]) for i in top_indices]

    sentences = _split_sentences(text)
    sent_scores = []
    for s in sentences:
        tokens = s.split()
        score = float(sum(weights[_safe_index(words, t)] for t in tokens if _safe_index(words, t) is not None))
        sent_scores.append(score)
    sent_scores = _normalize_list(sent_scores)

    return {
        "top_words": top_words,
        "top_word_scores": top_values,
        "sentence_contributions": sent_scores
    }


def explain_with_shap(text: str, top_k: int = 10, model_dir: Optional[str] = None, clf=None, tokenizer=None, model=None) -> Dict:
    if clf is None or tokenizer is None or model is None:
        clf, tokenizer, model = load_text_clf_pipeline(model_dir)
    enc = tokenizer(text, truncation=True, max_length=512, return_offsets_mapping=True)
    offsets = enc.get("offset_mapping", None)
    if offsets:
        last_end = 0
        for off in offsets:
            if off is not None:
                last_end = max(last_end, off[1])
        text = text[:last_end] if last_end > 0 else text
    masker = shap.maskers.Text(tokenizer)
    def predict_proba(texts: List[str]) -> np.ndarray:
        enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs
    explainer = shap.Explainer(predict_proba, masker)
    try:
        sv = explainer([text])
    except Exception:
        sentences = _split_sentences(text)
        return {"top_words": [], "top_word_scores": [], "sentence_contributions": _normalize_list([0.0 for _ in sentences])}
    data_tokens = sv.data[0]
    values = sv.values[0]
    if values.ndim == 2:
        # Seleccionar la clase con mayor impacto total
        class_idx = int(np.argmax(np.sum(np.abs(values), axis=0)))
        token_imp = values[:, class_idx]
    else:
        token_imp = values
    # Alinear impactos de tokens a palabras
    token_imp = np.array(token_imp, dtype=float)
    top_indices = np.argsort(-np.abs(token_imp))[:top_k]
    top_words = [data_tokens[i] for i in top_indices]
    top_values = [float(token_imp[i]) for i in top_indices]

    sentences = _split_sentences(text)
    sent_scores = []
    for s in sentences:
        score = 0.0
        for i, tok in enumerate(data_tokens):
            if tok in s:
                score += float(token_imp[i])
        sent_scores.append(score)
    sent_scores = _normalize_list(sent_scores)

    return {
        "top_words": top_words,
        "top_word_scores": top_values,
        "sentence_contributions": sent_scores
    }


def explain_with_attention(text: str, top_k: int = 10, model_dir: Optional[str] = None, clf=None, tokenizer=None, model=None) -> Dict:
    if tokenizer is None or model is None:
        _, tokenizer, model = load_text_clf_pipeline(model_dir)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**enc, output_attentions=True)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    if not outputs.attentions or len(outputs.attentions) == 0:
        cls_weights = np.zeros(len(tokens), dtype=float)
    else:
        attn_stack = torch.stack(outputs.attentions, dim=0)
        attn_mean = attn_stack.mean(dim=(0, 2))
        cls_weights = attn_mean[0, 0, :].detach().cpu().numpy()

    top_indices = np.argsort(-cls_weights)[:top_k]
    top_words = [tokens[i] for i in top_indices]
    top_values = [float(cls_weights[i]) for i in top_indices]

    sentences = _split_sentences(text)
    sent_scores = []
    for s in sentences:
        score = 0.0
        for i, tok in enumerate(tokens):
            if tok in s:
                score += float(cls_weights[i])
        sent_scores.append(score)
    sent_scores = _normalize_list(sent_scores)

    return {
        "top_words": top_words,
        "top_word_scores": top_values,
        "sentence_contributions": sent_scores
    }


def _split_sentences(text: str) -> List[str]:
    parts = []
    buff = []
    for ch in text:
        buff.append(ch)
        if ch in ".!?":
            parts.append("".join(buff).strip())
            buff = []
    if buff:
        parts.append("".join(buff).strip())
    return [p for p in parts if p]


def _normalize_list(vals: List[float]) -> List[float]:
    total = float(sum(abs(v) for v in vals)) or 1.0
    return [round(abs(v) / total, 4) for v in vals]


def _safe_index(words: List[str], token: str) -> Optional[int]:
    try:
        return words.index(token)
    except ValueError:
        return None
