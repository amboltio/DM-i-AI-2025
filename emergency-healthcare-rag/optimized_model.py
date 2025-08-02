import json
import torch
import torch.nn as nn
from typing import Tuple, List, Dict
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
import os
import pickle
from tqdm import tqdm
import re

class OptimizedMedicalClassifier(nn.Module):
    """Optimized medical statement classifier with enhanced architecture"""
    
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", num_topics: int = 115):
        super().__init__()
        
        # Use PubMedBERT - specifically trained on biomedical literature
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Enhanced classification heads with better architecture
        hidden_size = self.bert.config.hidden_size
        
        # Multi-layer classification heads
        self.truth_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 2)  # True/False
        )
        
        self.topic_classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_topics)  # 115 topics
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        truth_logits = self.truth_classifier(pooled_output)
        topic_logits = self.topic_classifier(pooled_output)
        
        return truth_logits, topic_logits

class EnhancedRAGSystem:
    """Enhanced RAG system with better retrieval and context processing"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use better embedding model for medical text
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
        # Load and preprocess reference articles
        self.reference_articles = self._load_and_preprocess_articles()
        self.article_embeddings = self._compute_article_embeddings()
        
        # Load topic mappings
        self.topics_mapping = self._load_topics_mapping()
        
    def _load_and_preprocess_articles(self) -> Dict[str, str]:
        """Load and preprocess medical reference articles"""
        articles = {}
        topics_dir = 'data/topics'
        
        print("Loading medical reference articles...")
        for topic_dir in tqdm(os.listdir(topics_dir)):
            topic_path = os.path.join(topics_dir, topic_dir)
            if os.path.isdir(topic_path):
                topic_content = []
                for file in os.listdir(topic_path):
                    if file.endswith('.md'):
                        file_path = os.path.join(topic_path, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Clean and preprocess content
                            content = self._preprocess_medical_text(content)
                            topic_content.append(content)
                
                if topic_content:
                    articles[topic_dir] = ' '.join(topic_content)
        
        return articles
    
    def _preprocess_medical_text(self, text: str) -> str:
        """Clean and preprocess medical text"""
        # Remove markdown formatting
        text = re.sub(r'#+\s*', '', text)  # Remove headers
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # Remove italic
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Remove links
        text = re.sub(r'---.*?---', '', text, flags=re.DOTALL)  # Remove frontmatter
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _compute_article_embeddings(self) -> Dict[str, np.ndarray]:
        """Compute embeddings for all reference articles"""
        print("Computing article embeddings...")
        embeddings = {}
        
        for topic_name, content in tqdm(self.reference_articles.items()):
            # Split content into chunks for better retrieval
            chunks = self._split_into_chunks(content, max_length=1000)
            chunk_embeddings = self.embedding_model.encode(chunks)
            embeddings[topic_name] = chunk_embeddings
        
        return embeddings
    
    def _split_into_chunks(self, text: str, max_length: int = 1000) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_length // 2):  # 50% overlap
            chunk = ' '.join(words[i:i + max_length])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def _load_topics_mapping(self) -> Dict[str, int]:
        """Load topic name to ID mapping"""
        with open('data/topics.json', 'r') as f:
            return json.load(f)
    
    def retrieve_relevant_context(self, statement: str, top_k: int = 3) -> Tuple[List[str], int]:
        """Enhanced context retrieval with better similarity matching"""
        # Encode the statement
        statement_embedding = self.embedding_model.encode([statement])
        
        # Find most similar articles
        similarities = {}
        for topic_name, chunk_embeddings in self.article_embeddings.items():
            # Compute similarity with all chunks
            chunk_similarities = cosine_similarity(statement_embedding, chunk_embeddings)[0]
            # Use max similarity for this topic
            max_similarity = np.max(chunk_similarities)
            similarities[topic_name] = max_similarity
        
        # Get top-k most similar topics
        top_topics = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Retrieve relevant context
        relevant_context = []
        best_topic_id = None
        
        for topic_name, similarity in top_topics:
            if topic_name in self.reference_articles:
                relevant_context.append(self.reference_articles[topic_name])
                if best_topic_id is None:
                    best_topic_id = self.topics_mapping.get(topic_name, 0)
        
        return relevant_context, best_topic_id or 0

class EnsembleMedicalPredictor:
    """Ensemble predictor combining multiple approaches for better accuracy"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load fine-tuned model
        self.classifier = self._load_fine_tuned_model()
        
        # Enhanced RAG system
        self.rag_system = EnhancedRAGSystem()
        
        # Rule-based fallback
        self.rule_based_classifier = self._create_rule_based_classifier()
        
        # Load topic keywords for keyword matching
        self.topic_keywords = self._load_topic_keywords()
        
    def _load_fine_tuned_model(self):
        """Load the fine-tuned medical classifier"""
        model_path = "fine_tuned_medical_model"
        
        if os.path.exists(model_path):
            print("Loading fine-tuned model...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = OptimizedMedicalClassifier()
            model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=self.device))
            model.to(self.device)
            model.eval()
            return {"model": model, "tokenizer": tokenizer}
        else:
            print("Fine-tuned model not found. Using fallback approach.")
            return None
    
    def _create_rule_based_classifier(self):
        """Create rule-based classifier as fallback"""
        # Simple rules based on medical keywords
        medical_truth_keywords = [
            'treatment', 'diagnosis', 'symptoms', 'management', 'therapy', 'medication',
            'procedure', 'test', 'examination', 'assessment', 'evaluation', 'monitoring',
            'prevention', 'risk factors', 'complications', 'outcomes', 'guidelines'
        ]
        
        medical_false_keywords = [
            'contraindicated', 'not recommended', 'avoid', 'never', 'incorrect',
            'wrong', 'false', 'myth', 'misconception', 'debunked'
        ]
        
        return {
            'truth_keywords': medical_truth_keywords,
            'false_keywords': medical_false_keywords
        }
    
    def _load_topic_keywords(self) -> Dict[str, List[str]]:
        """Load keywords for each topic"""
        keywords = {}
        topics_mapping = self._load_topics_mapping()
        
        for topic_name in topics_mapping.keys():
            # Extract keywords from topic name
            topic_words = re.findall(r'\b\w+\b', topic_name.lower())
            # Add medical variations
            medical_variations = []
            for word in topic_words:
                if word in ['acute', 'chronic']:
                    medical_variations.extend(['acute', 'chronic'])
                elif word in ['failure', 'dysfunction']:
                    medical_variations.extend(['failure', 'dysfunction', 'insufficiency'])
                elif word in ['syndrome', 'disease']:
                    medical_variations.extend(['syndrome', 'disease', 'disorder'])
                else:
                    medical_variations.append(word)
            
            keywords[topic_name] = list(set(medical_variations))
        
        return keywords
    
    def _load_topics_mapping(self) -> Dict[str, int]:
        """Load topic name to ID mapping"""
        with open('data/topics.json', 'r') as f:
            return json.load(f)
    
    def predict(self, statement: str) -> Tuple[int, int]:
        """
        Enhanced prediction combining multiple approaches
        Returns: (statement_is_true, statement_topic)
        """
        # Approach 1: Fine-tuned model prediction
        if self.classifier:
            try:
                truth_pred, topic_pred = self._predict_with_fine_tuned_model(statement)
                if truth_pred is not None and topic_pred is not None:
                    return truth_pred, topic_pred
            except Exception as e:
                print(f"Fine-tuned model failed: {e}")
        
        # Approach 2: Enhanced RAG + rule-based
        truth_pred, topic_pred = self._predict_with_rag_and_rules(statement)
        
        return truth_pred, topic_pred
    
    def _predict_with_fine_tuned_model(self, statement: str) -> Tuple[int, int]:
        """Predict using fine-tuned model"""
        model = self.classifier["model"]
        tokenizer = self.classifier["tokenizer"]
        
        # Tokenize
        inputs = tokenizer(
            statement,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            truth_logits, topic_logits = model(**inputs)
            
            truth_pred = torch.argmax(truth_logits, dim=1).item()
            topic_pred = torch.argmax(topic_logits, dim=1).item()
        
        return truth_pred, topic_pred
    
    def _predict_with_rag_and_rules(self, statement: str) -> Tuple[int, int]:
        """Predict using RAG + rule-based approach"""
        # Get relevant context
        context, rag_topic = self.rag_system.retrieve_relevant_context(statement)
        
        # Rule-based truth classification
        statement_lower = statement.lower()
        
        # Count positive and negative keywords
        truth_score = 0
        false_score = 0
        
        for keyword in self.rule_based_classifier['truth_keywords']:
            if keyword in statement_lower:
                truth_score += 1
        
        for keyword in self.rule_based_classifier['false_keywords']:
            if keyword in statement_lower:
                false_score += 1
        
        # Determine truth prediction
        if false_score > truth_score:
            truth_pred = 0
        else:
            truth_pred = 1
        
        # Enhanced topic classification
        topic_pred = self._enhanced_topic_classification(statement, rag_topic)
        
        return truth_pred, topic_pred
    
    def _enhanced_topic_classification(self, statement: str, rag_topic: int) -> int:
        """Enhanced topic classification combining RAG and keyword matching"""
        statement_lower = statement.lower()
        best_topic = rag_topic
        best_score = 0
        
        # Keyword matching with RAG as prior
        for topic_name, keywords in self.topic_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in statement_lower:
                    score += 1
            
            # Boost score if RAG suggested this topic
            if self.topics_mapping.get(topic_name, -1) == rag_topic:
                score += 2
            
            if score > best_score:
                best_score = score
                best_topic = self.topics_mapping.get(topic_name, 0)
        
        return best_topic

# Global instance
predictor = None

def predict(statement: str) -> Tuple[int, int]:
    """
    Main prediction function - optimized for competition
    Returns: (statement_is_true, statement_topic)
    """
    global predictor
    
    if predictor is None:
        predictor = EnsembleMedicalPredictor()
    
    return predictor.predict(statement) 