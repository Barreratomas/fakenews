import sys
import os
sys.path.append(os.path.join(os.getcwd(), "src"))

from src.extraction.article_extractor import extract_article_from_url, ArticleExtractionError

def test_url(name, url):
    print(f"\n--- Testing {name}: {url} ---")
    try:
        result = extract_article_from_url(url, timeout=10)
        print(f"SUCCESS")
        print(f"Title: {result['title']}")
        print(f"Text length: {len(result['text'])}")
    except ArticleExtractionError as e:
        print(f"CAUGHT EXPECTED ERROR: [{e.stage}] {e}")
    except Exception as e:
        print(f"UNEXPECTED ERROR: {type(e).__name__}: {e}")

if __name__ == "__main__":
    # 1. Valid URL (BBC Mundo)
    test_url("Valid Article", "https://www.bbc.com/mundo/articles/c74l1111111o") # Fake URL but BBC domain, might 404
    # Let's use a real one if possible, or just expect 404 which is http_error.
    # Better to use a reliable URL. Let's try google.com (will fail parsing or short) or example.com
    
    # 2. Invalid URL
    test_url("Invalid URL", "asdf")
    
    # 3. Homepage (CNN)
    test_url("Homepage", "https://www.cnn.com")
    
    # 4. Short article (Example.com)
    test_url("Short Article", "https://www.lmneuquen.com/patagonia/imagenes-que-duelen-greenpeace-mostro-el-aire-el-desastre-ambiental-los-incendios-chubut-n1225966")
    
    # 5. Paywall (Simulated via keywords)
    # Since I can't easily find a live paywall that matches exact keywords without searching,
    # I will trust the logic. But let's try a subscription page.
    test_url("Paywall Page", "https://elpais.com/suscripciones/")
