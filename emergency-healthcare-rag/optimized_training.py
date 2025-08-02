import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
import json
import os
import re
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import time
from optimized_model import OptimizedMedicalClassifier
import warnings
warnings.filterwarnings('ignore')

class OptimizedMedicalDataset(Dataset):
    """Optimized dataset with better preprocessing and augmentation"""
    
    def __init__(self, statements: List[str], truth_labels: List[int], topic_labels: List[int], 
                 tokenizer, max_length=512, augment=True):
        self.statements = statements
        self.truth_labels = truth_labels
        self.topic_labels = topic_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        
        # Medical-specific preprocessing
        self.medical_terms = self._load_medical_terms()
    
    def _load_medical_terms(self):
        """Load medical terminology for better preprocessing"""
        return {
            'abbreviations': {
                'mi': 'myocardial infarction',
                'acs': 'acute coronary syndrome',
                'pe': 'pulmonary embolism',
                'dvt': 'deep vein thrombosis',
                'copd': 'chronic obstructive pulmonary disease',
                'chf': 'congestive heart failure',
                'cva': 'cerebrovascular accident',
                'tia': 'transient ischemic attack'
            }
        }
    
    def _preprocess_statement(self, statement: str) -> str:
        """Enhanced preprocessing for medical statements"""
        # Expand medical abbreviations
        for abbr, full in self.medical_terms['abbreviations'].items():
            statement = re.sub(rf'\b{abbr}\b', full, statement, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        statement = ' '.join(statement.split())
        
        return statement
    
    def __len__(self):
        return len(self.statements)
    
    def __getitem__(self, idx):
        statement = self.statements[idx]
        truth_label = self.truth_labels[idx]
        topic_label = self.topic_labels[idx]
        
        # Preprocess statement
        statement = self._preprocess_statement(statement)
        
        # Tokenize with better settings
        encoding = self.tokenizer(
            statement,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=True
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'truth_label': torch.tensor(truth_label, dtype=torch.long),
            'topic_label': torch.tensor(topic_label, dtype=torch.long)
        }

def load_training_data():
    """Load and preprocess training data"""
    statements = []
    truth_labels = []
    topic_labels = []
    
    # Load statements
    statements_dir = 'data/train/statements'
    answers_dir = 'data/train/answers'
    
    print("Loading training data...")
    for filename in tqdm(os.listdir(statements_dir)):
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
    
    print(f"Loaded {len(statements)} training examples")
    return statements, truth_labels, topic_labels

def calculate_class_weights(truth_labels: List[int], topic_labels: List[int]):
    """Calculate class weights for imbalanced data"""
    # Truth class weights
    truth_counts = np.bincount(truth_labels)
    truth_weights = 1.0 / truth_counts
    truth_weights = truth_weights / np.sum(truth_weights)
    
    # Topic class weights
    topic_counts = np.bincount(topic_labels, minlength=115)
    topic_weights = 1.0 / (topic_counts + 1)  # Add 1 to avoid division by zero
    topic_weights = topic_weights / np.sum(topic_weights)
    
    return torch.FloatTensor(truth_weights), torch.FloatTensor(topic_weights)

def train_optimized_model(
    model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    batch_size: int = 8,
    epochs: int = 5,
    learning_rate: float = 3e-5,
    warmup_steps: int = 100,
    save_path: str = "fine_tuned_medical_model",
    use_class_weights: bool = True
):
    """Train the optimized medical statement classifier"""
    
    # Load data
    print("Loading training data...")
    statements, truth_labels, topic_labels = load_training_data()
    
    # Check data distribution
    unique_topics, topic_counts = np.unique(topic_labels, return_counts=True)
    print(f"Data distribution: {len(statements)} total examples")
    print(f"Topic distribution: {len(unique_topics)} unique topics")
    print(f"Topics with 1 example: {np.sum(topic_counts == 1)}")
    print(f"Topics with 2+ examples: {np.sum(topic_counts >= 2)}")
    
    # Try stratified split first, fall back to regular split if it fails
    try:
        print("Attempting stratified train-test split...")
        train_statements, val_statements, train_truth, val_truth, train_topic, val_topic = train_test_split(
            statements, truth_labels, topic_labels, test_size=0.2, random_state=42, stratify=topic_labels
        )
        print("Stratified split successful!")
    except ValueError as e:
        print(f"Stratified split failed: {e}")
        print("Using regular train-test split...")
        train_statements, val_statements, train_truth, val_truth, train_topic, val_topic = train_test_split(
            statements, truth_labels, topic_labels, test_size=0.2, random_state=42
        )
        print("Regular split successful!")
    
    print(f"Training set: {len(train_statements)} examples")
    print(f"Validation set: {len(val_statements)} examples")
    
    # Initialize tokenizer and model
    print("Initializing optimized model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = OptimizedMedicalClassifier(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    train_dataset = OptimizedMedicalDataset(train_statements, train_truth, train_topic, tokenizer)
    val_dataset = OptimizedMedicalDataset(val_statements, val_truth, val_topic, tokenizer, augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Calculate class weights
    if use_class_weights:
        truth_weights, topic_weights = calculate_class_weights(truth_labels, topic_labels)
        truth_weights = truth_weights.to(device)
        topic_weights = topic_weights.to(device)
    else:
        truth_weights = topic_weights = None
    
    # Optimizer with weight decay
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    # Loss functions with class weights
    if truth_weights is not None:
        truth_criterion = nn.CrossEntropyLoss(weight=truth_weights)
        topic_criterion = nn.CrossEntropyLoss(weight=topic_weights)
    else:
        truth_criterion = nn.CrossEntropyLoss()
        topic_criterion = nn.CrossEntropyLoss()
    
    # Training metrics
    best_val_accuracy = 0.0
    patience = 3
    patience_counter = 0
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        total_loss = 0
        truth_correct = 0
        topic_correct = 0
        total_train = 0
        
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
            
            # Weighted loss combination
            loss = 0.4 * truth_loss + 0.6 * topic_loss  # Give more weight to topic classification
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Calculate training accuracy
            truth_preds = torch.argmax(truth_logits, dim=1)
            topic_preds = torch.argmax(topic_logits, dim=1)
            
            truth_correct += (truth_preds == truth_labels).sum().item()
            topic_correct += (topic_preds == topic_labels).sum().item()
            total_train += truth_labels.size(0)
            
            total_loss += loss.item()
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'truth_acc': f'{truth_correct/total_train:.3f}',
                'topic_acc': f'{topic_correct/total_train:.3f}'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_truth_correct = 0
        val_topic_correct = 0
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
                loss = 0.4 * truth_loss + 0.6 * topic_loss
                val_loss += loss.item()
                
                # Calculate validation accuracy
                truth_preds = torch.argmax(truth_logits, dim=1)
                topic_preds = torch.argmax(topic_logits, dim=1)
                
                val_truth_correct += (truth_preds == truth_labels).sum().item()
                val_topic_correct += (topic_preds == topic_labels).sum().item()
                total_val += truth_labels.size(0)
        
        # Calculate metrics
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_truth_accuracy = truth_correct / total_train
        train_topic_accuracy = topic_correct / total_train
        val_truth_accuracy = val_truth_correct / total_val
        val_topic_accuracy = val_topic_correct / total_val
        
        # Combined accuracy (weighted)
        val_combined_accuracy = 0.4 * val_truth_accuracy + 0.6 * val_topic_accuracy
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch + 1}/{epochs} ({epoch_time:.1f}s):")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Truth Acc: {train_truth_accuracy:.4f}, Topic Acc: {train_topic_accuracy:.4f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, Truth Acc: {val_truth_accuracy:.4f}, Topic Acc: {val_topic_accuracy:.4f}")
        print(f"  Combined Val Accuracy: {val_combined_accuracy:.4f}")
        
        # Save best model
        if val_combined_accuracy > best_val_accuracy:
            best_val_accuracy = val_combined_accuracy
            patience_counter = 0
            
            print(f"  New best model! Saving to {save_path}...")
            
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            # Save training config
            config = {
                'model_name': model_name,
                'best_val_accuracy': best_val_accuracy,
                'epoch': epoch + 1,
                'training_params': {
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'epochs': epochs,
                    'warmup_steps': warmup_steps
                }
            }
            
            with open(os.path.join(save_path, 'training_config.json'), 'w') as f:
                json.dump(config, f, indent=2)
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {patience} epochs without improvement")
            break
    
    print(f"Training completed! Best validation accuracy: {best_val_accuracy:.4f}")
    return model, tokenizer

def evaluate_model(model, tokenizer, test_statements, test_truth, test_topic, device):
    """Evaluate the trained model"""
    model.eval()
    
    truth_correct = 0
    topic_correct = 0
    total = 0
    
    with torch.no_grad():
        for statement, truth_label, topic_label in zip(test_statements, test_truth, test_topic):
            # Tokenize
            inputs = tokenizer(
                statement,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)
            
            # Predict
            truth_logits, topic_logits = model(**inputs)
            
            truth_pred = torch.argmax(truth_logits, dim=1).item()
            topic_pred = torch.argmax(topic_logits, dim=1).item()
            
            if truth_pred == truth_label:
                truth_correct += 1
            if topic_pred == topic_label:
                topic_correct += 1
            total += 1
    
    truth_accuracy = truth_correct / total
    topic_accuracy = topic_correct / total
    combined_accuracy = 0.4 * truth_accuracy + 0.6 * topic_accuracy
    
    print(f"Test Results:")
    print(f"  Truth Accuracy: {truth_accuracy:.4f}")
    print(f"  Topic Accuracy: {topic_accuracy:.4f}")
    print(f"  Combined Accuracy: {combined_accuracy:.4f}")
    
    return truth_accuracy, topic_accuracy, combined_accuracy

if __name__ == "__main__":
    import re
    
    # Train the optimized model
    print("Starting optimized medical model training...")
    
    model, tokenizer = train_optimized_model(
        model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        batch_size=8,  # Adjust based on your GPU memory
        epochs=5,
        learning_rate=3e-5,
        warmup_steps=100,
        save_path="fine_tuned_medical_model",
        use_class_weights=True
    )
    
    print("Training completed successfully!")
    print("Model saved to 'fine_tuned_medical_model' directory")
    print("You can now use the optimized model in your API!") 