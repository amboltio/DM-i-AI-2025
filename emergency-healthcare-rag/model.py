import json
import torch
from typing import Tuple, List
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

class EmergencyHealthcareRAG:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the embedding model for RAG first
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load topic embeddings
        self.topics_embeddings = self._load_topic_embeddings()
        
        # Load the classification model (you can replace with your fine-tuned model)
        self.classifier = self._load_classifier()
        
        # Load reference articles
        self.reference_articles = self._load_reference_articles()
        
    def _load_topic_embeddings(self):
        """Load pre-computed topic embeddings"""
        topics_file = 'data/topics.json'
        with open(topics_file, 'r') as f:
            topics = json.load(f)
        
        # Create embeddings for topic names
        topic_names = list(topics.keys())
        embeddings = self.embedding_model.encode(topic_names)
        
        return {topic_id: embedding for topic_id, embedding in zip(topics.values(), embeddings)}
    
    def _load_classifier(self):
        """Load the classification model - can be fine-tuned on your dataset"""
        # Option 1: Use a pre-trained model for classification
        model_name = "microsoft/DialoGPT-medium"  # You can replace with better medical models
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Option 2: Load your fine-tuned model
        # model_path = "path/to/your/fine_tuned_model"
        # tokenizer = AutoTokenizer.from_pretrained(model_path)
        # model = AutoModel.from_pretrained(model_path)
        
        model.to(self.device)
        return {"tokenizer": tokenizer, "model": model}
    
    def _load_reference_articles(self):
        """Load reference articles for RAG"""
        articles = {}
        topics_dir = 'data/topics'
        
        for topic_dir in os.listdir(topics_dir):
            topic_path = os.path.join(topics_dir, topic_dir)
            if os.path.isdir(topic_path):
                for file in os.listdir(topic_path):
                    if file.endswith('.md'):
                        file_path = os.path.join(topic_path, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            articles[topic_dir] = content
        
        return articles
    
    def retrieve_relevant_context(self, statement: str, top_k: int = 3):
        """Retrieve relevant medical context using RAG"""
        # Encode the statement
        statement_embedding = self.embedding_model.encode([statement])
        
        # Find most similar topics
        similarities = {}
        for topic_id, topic_embedding in self.topics_embeddings.items():
            similarity = cosine_similarity(statement_embedding, [topic_embedding])[0][0]
            similarities[topic_id] = similarity
        
        # Get top-k most similar topics
        top_topics = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Retrieve relevant articles
        relevant_context = []
        for topic_id, similarity in top_topics:
            # Find topic name from ID
            topic_name = None
            for name, tid in json.load(open('data/topics.json')).items():
                if tid == topic_id:
                    topic_name = name
                    break
            
            if topic_name and topic_name in self.reference_articles:
                relevant_context.append(self.reference_articles[topic_name])
        
        return relevant_context, top_topics[0][0]  # Return context and best topic
    
    def classify_statement(self, statement: str, context: List[str]) -> Tuple[int, int]:
        """Classify the statement as true/false and determine topic"""
        # Combine statement with context
        full_context = f"Statement: {statement}\n\nMedical Context: {' '.join(context[:2])}"
        
        # For now, using a simple rule-based approach
        # You should replace this with your fine-tuned model
        
        # Simple keyword-based truth classification
        medical_keywords = ['treatment', 'diagnosis', 'symptoms', 'management', 'therapy']
        statement_lower = statement.lower()
        
        # Count medical keywords to estimate truthfulness
        keyword_count = sum(1 for keyword in medical_keywords if keyword in statement_lower)
        statement_is_true = 1 if keyword_count > 2 else 0
        
        # Topic classification using RAG
        _, best_topic = self.retrieve_relevant_context(statement)
        
        return statement_is_true, best_topic

# Global instance
rag_system = None

def predict(statement: str) -> Tuple[int, int]:
    """
    Predict both binary classification (true/false) and topic classification for a medical statement.
    
    Args:
        statement (str): The medical statement to classify
        
    Returns:
        Tuple[int, int]: (statement_is_true, statement_topic)
            - statement_is_true: 1 if true, 0 if false
            - statement_topic: topic ID from 0-114
    """
    global rag_system
    
    # Initialize RAG system if not already done
    if rag_system is None:
        rag_system = EmergencyHealthcareRAG()
    
    # Retrieve relevant context
    context, _ = rag_system.retrieve_relevant_context(statement)
    
    # Classify the statement
    statement_is_true, statement_topic = rag_system.classify_statement(statement, context)
    
    return statement_is_true, statement_topic

# Legacy function for backward compatibility
def match_topic(statement: str) -> int:
    """
    Simple keyword matching to find the best topic match.
    """
    # Load topics mapping
    with open('data/topics.json', 'r') as f:
        topics = json.load(f)
    
    statement_lower = statement.lower()
    best_topic = 0
    max_matches = 0
    
    for topic_name, topic_id in topics.items():
        # Extract keywords from topic name
        keywords = topic_name.lower().replace('_', ' ').replace('(', '').replace(')', '').split()
        
        # Count keyword matches in statement
        matches = sum(1 for keyword in keywords if keyword in statement_lower)
        
        if matches > max_matches:
            max_matches = matches
            best_topic = topic_id
    
    return best_topic
