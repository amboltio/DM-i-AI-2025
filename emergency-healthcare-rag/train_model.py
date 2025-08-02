import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import json
import os
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

class MedicalStatementDataset(Dataset):
    def __init__(self, statements: List[str], truth_labels: List[int], topic_labels: List[int], tokenizer, max_length=512):
        self.statements = statements
        self.truth_labels = truth_labels
        self.topic_labels = topic_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.statements)
    
    def __getitem__(self, idx):
        statement = self.statements[idx]
        truth_label = self.truth_labels[idx]
        topic_label = self.topic_labels[idx]
        
        # Tokenize the statement
        encoding = self.tokenizer(
            statement,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'truth_label': torch.tensor(truth_label, dtype=torch.long),
            'topic_label': torch.tensor(topic_label, dtype=torch.long)
        }

class MedicalClassifier(nn.Module):
    def __init__(self, model_name: str, num_topics: int = 115):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        
        # Classification heads
        self.truth_classifier = nn.Linear(self.bert.config.hidden_size, 2)  # True/False
        self.topic_classifier = nn.Linear(self.bert.config.hidden_size, num_topics)  # 115 topics
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        truth_logits = self.truth_classifier(pooled_output)
        topic_logits = self.topic_classifier(pooled_output)
        
        return truth_logits, topic_logits

def load_training_data():
    """Load training data from the data/train/ directory"""
    statements = []
    truth_labels = []
    topic_labels = []
    
    # Load statements
    statements_dir = 'data/train/statements'
    answers_dir = 'data/train/answers'
    
    for filename in os.listdir(statements_dir):
        if filename.endswith('.txt'):
            # Load statement
            with open(os.path.join(statements_dir, filename), 'r', encoding='utf-8') as f:
                statement = f.read().strip()
            
            # Load corresponding answer
            answer_file = filename.replace('.txt', '.json')
            with open(os.path.join(answers_dir, answer_file), 'r') as f:
                answer = json.load(f)
            
            statements.append(statement)
            truth_labels.append(answer['statement_is_true'])
            topic_labels.append(answer['statement_topic'])
    
    return statements, truth_labels, topic_labels

def train_model(model_name: str = "microsoft/DialoGPT-medium", 
                batch_size: int = 8, 
                epochs: int = 3, 
                learning_rate: float = 2e-5,
                save_path: str = "fine_tuned_medical_model"):
    """Train the medical statement classifier"""
    
    # Load data
    print("Loading training data...")
    statements, truth_labels, topic_labels = load_training_data()
    
    # Split data
    train_statements, val_statements, train_truth, val_truth, train_topic, val_topic = train_test_split(
        statements, truth_labels, topic_labels, test_size=0.2, random_state=42
    )
    
    # Initialize tokenizer and model
    print("Initializing model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MedicalClassifier(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    train_dataset = MedicalStatementDataset(train_statements, train_truth, train_topic, tokenizer)
    val_dataset = MedicalStatementDataset(val_statements, val_truth, val_topic, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # Loss functions
    truth_criterion = nn.CrossEntropyLoss()
    topic_criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            truth_labels = batch['truth_label'].to(device)
            topic_labels = batch['topic_label'].to(device)
            
            optimizer.zero_grad()
            
            truth_logits, topic_logits = model(input_ids, attention_mask)
            
            # Calculate losses
            truth_loss = truth_criterion(truth_logits, truth_labels)
            topic_loss = topic_criterion(topic_logits, topic_labels)
            
            # Combined loss (you can adjust weights)
            loss = truth_loss + topic_loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        val_loss = 0
        truth_correct = 0
        topic_correct = 0
        total_val = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                truth_labels = batch['truth_label'].to(device)
                topic_labels = batch['topic_label'].to(device)
                
                truth_logits, topic_logits = model(input_ids, attention_mask)
                
                # Calculate validation loss
                truth_loss = truth_criterion(truth_logits, truth_labels)
                topic_loss = topic_criterion(topic_logits, topic_labels)
                loss = truth_loss + topic_loss
                val_loss += loss.item()
                
                # Calculate accuracy
                truth_preds = torch.argmax(truth_logits, dim=1)
                topic_preds = torch.argmax(topic_logits, dim=1)
                
                truth_correct += (truth_preds == truth_labels).sum().item()
                topic_correct += (topic_preds == topic_labels).sum().item()
                total_val += truth_labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        truth_accuracy = truth_correct / total_val
        topic_accuracy = topic_correct / total_val
        
        print(f"Epoch {epoch + 1}: Val Loss: {avg_val_loss:.4f}, "
              f"Truth Acc: {truth_accuracy:.4f}, Topic Acc: {topic_accuracy:.4f}")
    
    # Save the model
    print(f"Saving model to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print("Training completed!")
    return model, tokenizer

if __name__ == "__main__":
    # Train the model
    model, tokenizer = train_model(
        model_name="microsoft/DialoGPT-medium",
        batch_size=4,  # Adjust based on your GPU memory
        epochs=3,
        learning_rate=2e-5,
        save_path="fine_tuned_medical_model"
    ) 