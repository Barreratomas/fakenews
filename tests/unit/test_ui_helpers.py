import pytest
from src.ui.html_generators import get_label_styles, generate_conflict_html, generate_label_html

class TestHtmlGenerators:
    
    def test_get_label_styles(self):
        # REAL -> Green
        styles = get_label_styles("REAL")
        assert styles["color"] == "green"
        
        # FAKE -> Red
        styles = get_label_styles("FAKE")
        assert styles["color"] == "red"
        
        # WARNING -> Yellow
        styles = get_label_styles("ADVERTENCIA_CONTROVERTIDA")
        assert styles["color"] == "#d69e2e"
        
        # UNKNOWN -> Gray
        styles = get_label_styles("UNKNOWN")
        assert styles["color"] == "gray"

    def test_generate_conflict_html(self):
        # Empty message -> Empty string
        assert generate_conflict_html("", "red") == ""
        
        # Message present -> HTML block
        html = generate_conflict_html("Conflicto detectado", "red")
        assert "Conflicto detectado" in html
        assert "background-color: #fffaf0" in html
        assert "border-left: 5px solid red" in html

    def test_generate_label_html(self):
        # Standard case
        html = generate_label_html("REAL", 0.95)
        assert "REAL" in html
        assert "95.00%" in html
        assert "background-color: #e6fffa" in html
        
        # With Conflict
        html_conflict = generate_label_html("FAKE", 0.80, conflict_msg="RAG says otherwise")
        assert "FAKE" in html_conflict
        assert "RAG says otherwise" in html_conflict
