import pytest
from unittest.mock import patch, MagicMock
from src.rag.rag_pipeline import WebRetriever, FactChecker

class TestWebRetriever:
    
    @patch("src.rag.rag_pipeline.DDGS")
    def test_query_success(self, mock_ddgs_cls):
        # Setup mock
        mock_ddgs_instance = MagicMock()
        mock_ddgs_cls.return_value = mock_ddgs_instance
        
        # Mock search results
        mock_results = [
            {"title": "Result 1", "body": "Body 1", "href": "http://1.com"},
            {"title": "Result 2", "body": "Body 2", "href": "http://2.com"}
        ]
        mock_ddgs_instance.text.return_value = mock_results
        
        # Execute
        retriever = WebRetriever()
        results = retriever.query("fake news query")
        
        # Assert
        assert len(results) == 2
        assert results[0]["text"] == "Result 1: Body 1"
        assert results[0]["source"] == "http://1.com"
        assert results[0]["score"] == 1.0 # 1/(0+1)
        assert results[1]["score"] == 0.5 # 1/(1+1)

    @patch("src.rag.rag_pipeline.DDGS")
    def test_query_empty(self, mock_ddgs_cls):
        mock_ddgs_instance = MagicMock()
        mock_ddgs_cls.return_value = mock_ddgs_instance
        mock_ddgs_instance.text.return_value = []
        
        retriever = WebRetriever()
        results = retriever.query("query")
        
        assert len(results) == 0

class TestFactChecker:
    
    @patch("src.rag.rag_pipeline.WebRetriever")
    @patch("src.rag.rag_pipeline.pipeline") # Mocking transformers pipeline
    @patch("src.rag.rag_pipeline.AutoTokenizer") # Mocking Tokenizer
    @patch("src.rag.rag_pipeline.AutoModelForSeq2SeqLM") # Mocking Model
    def test_check_fake_contradicted(self, mock_model, mock_tokenizer, mock_pipeline, mock_retriever_cls):
        # Setup Retriever Mock
        mock_retriever = MagicMock()
        mock_retriever_cls.return_value = mock_retriever
        mock_retriever.query.return_value = [
            {"text": "Fact 1", "source": "http://source.com", "score": 1.0}
        ]
        
        # Setup LLM Mock
        mock_llm = MagicMock()
        # The pipeline returns a list of dicts [{'generated_text': '...'}]
        mock_llm.return_value = [{"generated_text": "The claim is CONTRADICTED by sources."}]
        mock_pipeline.return_value = mock_llm
        
        # Initialize singleton (reset if needed or just rely on new mock injection)
        # Note: FactChecker is a singleton. We might need to reset its instance or 
        # ensure _initialized is handled. For unit tests, it's safer to patch the class attributes 
        # or manually reset the singleton.
        FactChecker._instance = None 
        
        checker = FactChecker()
        result = checker.check("Some fake claim")
        
        assert result["verdict"] == "FAKE"
        assert "CONTRADICTED" in result["analysis"]
        assert len(result["sources"]) == 1

    @patch("src.rag.rag_pipeline.WebRetriever")
    @patch("src.rag.rag_pipeline.pipeline") 
    @patch("src.rag.rag_pipeline.AutoTokenizer")
    @patch("src.rag.rag_pipeline.AutoModelForSeq2SeqLM")
    def test_check_real_supported(self, mock_model, mock_tokenizer, mock_pipeline, mock_retriever_cls):
        mock_retriever = MagicMock()
        mock_retriever_cls.return_value = mock_retriever
        mock_retriever.query.return_value = [{"text": "Fact 1", "source": "s", "score": 1}]
        
        mock_llm = MagicMock()
        mock_llm.return_value = [{"generated_text": "The claim is SUPPORTED by evidence."}]
        mock_pipeline.return_value = mock_llm
        
        FactChecker._instance = None
        checker = FactChecker()
        result = checker.check("Some real claim")
        
        assert result["verdict"] == "REAL"

    @patch("src.rag.rag_pipeline.WebRetriever")
    def test_check_no_info(self, mock_retriever_cls):
        mock_retriever = MagicMock()
        mock_retriever_cls.return_value = mock_retriever
        mock_retriever.query.return_value = [] # Empty results
        
        FactChecker._instance = None
        checker = FactChecker()
        result = checker.check("Unknown claim")
        
        assert result["analysis_type"] == "heuristic"
        assert "No se encontró información" in result["analysis"]
