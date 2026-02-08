import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# Imports
from src.inference.predict import load_text_clf_pipeline

def calibrate_threshold():
    print("CALIBRACIÓN DEL THRESHOLD PARA DETECCIÓN DE FAKE NEWS")
    print("=" * 100)
    
    examples = [
        # FAKES (Deben ser FAKE)
        { "text": "URGENTE: La NASA confirma que el sol se apagará en 2025 para mantenimiento.", "label": "FAKE" },
        { "text": "Elon Musk compra Uruguay y lo renombra como 'X-Land'.", "label": "FAKE" },
        { "text": "Científicos descubren dinosaurio vivo en el Amazonas.", "label": "FAKE" },
        
        # REALS (Deben ser REAL)
        { "text": "El Ministerio de Economía anunció las nuevas tasas de interés para plazos fijos.", "label": "REAL" },
        { "text": "La selección argentina disputará un partido amistoso el próximo mes.", "label": "REAL" },
        { "text": "El presidente se reunió hoy con líderes del G20 para discutir el cambio climático.", "label": "REAL" },
        { "text": "Aumentó el precio de la nafta un 4% en todas las estaciones de servicio del país.", "label": "REAL" }
    ]
    
    # Cargar modelo una vez
    _, tokenizer, model = load_text_clf_pipeline()
    
    results = []
    
    print(f"{'TEXTO (Resumen)':<40} | {'Prob. FAKE':<10} | {'Prob. REAL':<10} | {'Etiqueta Real':<15}")
    print("-" * 100)
    
    for item in examples:
        text = item["text"]
        true_label = item["label"]
        
        # Inferencia manual para obtener probabilidades raw
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model = model.cuda()
            
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
            
        prob_fake = probs[0] # Label 0 = FAKE
        prob_real = probs[1] # Label 1 = REAL
        
        results.append({
            "text": text,
            "true_label": true_label,
            "prob_fake": prob_fake
        })
        
        print(f"{text[:37]+'...':<40} | {prob_fake:.4f}     | {prob_real:.4f}     | {true_label:<15}")

    print("\n" + "=" * 100)
    print("ANÁLISIS DE UMBRALES")
    print("=" * 100)
    
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    
    for t in thresholds:
        correct = 0
        fp = 0 # False Positives (Real clasificado como Fake)
        fn = 0 # False Negatives (Fake clasificado como Real)
        
        for res in results:
            pred_label = "FAKE" if res["prob_fake"] >= t else "REAL"
            if pred_label == res["true_label"]:
                correct += 1
            else:
                if res["true_label"] == "REAL" and pred_label == "FAKE":
                    fp += 1
                elif res["true_label"] == "FAKE" and pred_label == "REAL":
                    fn += 1
        
        acc = correct / len(results)
        print(f"Umbral > {t:.2f} | Accuracy: {acc:.2f} ({correct}/{len(results)}) | FP (Alarmismo): {fp} | FN (Fuga): {fn}")
        if fp == 0 and fn == 0:
            print("UMBRAL IDEAL ENCONTRADO")

if __name__ == "__main__":
    calibrate_threshold()
