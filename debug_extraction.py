from src.extraction.article_extractor import extract_article_from_url
import sys

url = "https://tn.com.ar/#:~:text=El%20riesgo%20pa%C3%ADs%20se%20ubica,las%20acciones%20en%20el%20exterior."
try:
    # Set min_length to 0 to see what we get
    article = extract_article_from_url(url, min_length=10)
    print("--- Extracted Text ---")
    print(article["text"])
    print("--- End ---")
    print(f"Length: {len(article['text'])}")
except Exception as e:
    print(f"Error: {e}")
