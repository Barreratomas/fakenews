import os
from src.rag.rag_pipeline import RagIndex, rag_fact_check

if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), "data", "raw")
    csv_path = os.path.join(data_dir, "onlytrue1000.csv")
    idx = RagIndex()
    if not os.path.exists(os.path.join(os.getcwd(), "models", "rag_index", "faiss.index")):
        print("Building RAG index from onlytrue1000.csv ...")
        out = idx.build_from_csvs([csv_path])
        print("Index built:", out)
    else:
        print("Loading existing RAG index ...")
        idx.load()
    sample_text = (
        "El gobierno anunció nuevas medidas económicas mientras la oposición critica el plan. "
        "La noticia describe cambios en políticas fiscales y reacciones de distintos sectores."
    )
    print("\nQuerying similar real news ...")
    retrieved = idx.query(sample_text, top_k=3)
    for r in retrieved:
        print(f"- score={r['score']:.3f} source={r['source']}")
        print(r["text"][:180], "...\n")
    print("\nLLM comparative analysis (with fallback if needed) ...")
    result = rag_fact_check(sample_text, top_k=3)
    print(result["llm_analysis"])
