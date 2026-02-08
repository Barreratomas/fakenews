import pytest
from src.preprocessing.clean_text import clean_text

# TEST UNITARIO: Prueba una función aislada sin dependencias externas.
# Objetivo: Verificar que la limpieza de texto funciona en todos los casos posibles.

class TestCleanText:
    
    # Caso 1: Limpieza básica de espacios
    def test_basic_cleaning(self):
        text = "  Hola   Mundo  "
        # Verificamos que quite espacios al inicio/final y reduzca espacios múltiples a uno solo
        assert clean_text(text) == "Hola Mundo"

    # Caso 2: Eliminación de HTML
    def test_html_tags(self):
        text = "<p>Hola</p> <div>Mundo</div>"
        # La función debe usar expresiones regulares para borrar todo lo que esté entre < >
        assert clean_text(text) == "Hola Mundo"

    # Caso 3: Decodificación de entidades HTML
    def test_html_entities(self):
        # &amp; es el código HTML para el símbolo &
        text = "Hola &amp; Mundo"
        # html.unescape() debe convertirlo a su caracter real
        assert clean_text(text) == "Hola & Mundo"

    # Caso 4: Eliminación de URLs
    def test_urls(self):
        # Las URLs ensucian el modelo de IA, así que deben eliminarse
        text = "Visita https://google.com para más info"
        assert clean_text(text) == "Visita para más info"
        
        text = "Visita www.google.com"
        assert clean_text(text) == "Visita"

    # Caso 5: Manejo de inputs inválidos (Robustez)
    def test_non_string_input(self):
        # Si por error llega un None o un número, no debe explotar, sino devolver string vacío
        assert clean_text(None) == ""
        assert clean_text(123) == ""

    # Caso 6: Prueba compleja (Todo junto)
    def test_complex_mix(self):
        text = """
        <div>
            <h1>Título &amp; Algo más</h1>
            <a href="http://example.com">Link</a>
        </div>
        """
        # Debe: quitar divs/h1/a, decodificar &amp;, quitar la URL, y normalizar saltos de línea
        expected = "Título & Algo más Link"
        assert clean_text(text) == expected
