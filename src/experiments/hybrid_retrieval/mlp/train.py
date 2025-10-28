import argparse
import numpy as np
import boto3
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .config import PARAM_GRID, INPUT_DIM, MAX_EPOCHS, EARLY_STOPPING_PATIENCE


class FusionMLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[512, 256], dropout=0.2, use_batch_norm=False):
        super(FusionMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()


def get_query_embedding(query):
    """Get Titan v2 embedding for query"""
    bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-west-2')
    
    request_body = {
        "inputText": query,
        "dimensions": 1024,
        "normalize": True
    }
    
    try:
        response = bedrock_runtime.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            contentType='application/json',
            accept='application/json',
            body=str(request_body)
        )
        response_body = response['body'].read().decode('utf-8')
        return np.array(eval(response_body)['embedding'])
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return np.zeros(1024)


def compute_optimal_alpha(union_chunks):
    """Find optimal alpha by minimizing MSE between fused scores and LSS"""
    if not union_chunks:
        return 0.5
    
    s = np.array([c['normalized_semantic_score'] for c in union_chunks])
    b = np.array([c['normalized_bm25_score'] for c in union_chunks])
    L = np.array([c['lss'] for c in union_chunks])
    
    diff = s - b
    numerator = np.sum((L - b) * diff)
    denominator = np.sum(diff ** 2)
    
    if denominator < 1e-10:
        return 0.5
    
    alpha = numerator / denominator
    return float(np.clip(alpha, 0.0, 1.0))


def load_data(data_dir):
    """Load all datasets and prepare training data"""
    print("Loading data...")
    
    all_data = []
    datasets = ['single_file', 'single_file_multihop', 'multi_file_multihop']
    
    for dataset in datasets:
        file_path = Path(data_dir) / f"{dataset}_merged.jsonl"
        if not file_path.exists():
            print(f"  Skipping {dataset} - file not found")
            continue
        
        print(f"  Loading {dataset}...")
        
        with open(file_path, 'r') as f:
            for line in f:
                item = eval(line.strip())
                optimal_alpha = compute_optimal_alpha(item['union_chunks'])
                
                all_data.append({
                    'question': item['question'],
                    'optimal_alpha': optimal_alpha,
                    'dataset': dataset
                })
    
    print(f"Loaded {len(all_data)} samples")
    return all_data


def get_embeddings(questions):
    """Get embeddings for all questions"""
    print(f"Getting embeddings for {len(questions)} questions...")
    embeddings = []
    
    for i, question in enumerate(questions):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(questions)}")
        embeddings.append(get_query_embedding(question))
    
    return np.array(embeddings)


def main():
    parser = argparse.ArgumentParser(description='Train MLP model')
    parser.add_argument('--input-dir', default='results/hybrid_retrieval', help='Input directory with merged.jsonl files')
    parser.add_argument('--output-dir', default='results/hybrid_retrieval/models', help='Output directory')
    args = parser.parse_args()
    
    print("=== MLP TRAINING ===")
    
    data = load_data(args.input_dir)
    
    train_data, temp_data = train_test_split(
        data, test_size=0.3, random_state=42,
        stratify=[d['dataset'] for d in data]
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=0.5, random_state=42,
        stratify=[d['dataset'] for d in temp_data]
    )
    
    print(f"\nData split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    print("\nGetting embeddings...")
    X_train = get_embeddings([d['question'] for d in train_data])
    X_val = get_embeddings([d['question'] for d in val_data])
    X_test = get_embeddings([d['question'] for d in test_data])
    
    y_train = np.array([d['optimal_alpha'] for d in train_data])
    y_val = np.array([d['optimal_alpha'] for d in val_data])
    y_test = np.array([d['optimal_alpha'] for d in test_data])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    print("\nHyperparameter search...")
    
    best_val_mse = float('inf')
    best_model = None
    best_params = None
    
    total_configs = (len(PARAM_GRID['hidden_dims']) * len(PARAM_GRID['learning_rate']) * 
                    len(PARAM_GRID['batch_size']) * len(PARAM_GRID['dropout']) * 
                    len(PARAM_GRID['weight_decay']) * len(PARAM_GRID['use_batch_norm']))
    
    print(f"Testing {total_configs} configurations...")
    
    config_num = 0
    for hidden_dims in PARAM_GRID['hidden_dims']:
        for lr in PARAM_GRID['learning_rate']:
            for batch_size in PARAM_GRID['batch_size']:
                for dropout in PARAM_GRID['dropout']:
                    for weight_decay in PARAM_GRID['weight_decay']:
                        for use_batch_norm in PARAM_GRID['use_batch_norm']:
                            config_num += 1
                            
                            print(f"\n[{config_num}/{total_configs}] hidden={hidden_dims}, lr={lr}, batch={batch_size}, dropout={dropout}, wd={weight_decay}, bn={use_batch_norm}")
                            
                            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                            
                            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                            
                            model = FusionMLP(
                                input_dim=INPUT_DIM,
                                hidden_dims=hidden_dims,
                                dropout=dropout,
                                use_batch_norm=use_batch_norm
                            ).to(device)
                            
                            criterion = nn.MSELoss()
                            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                            
                            best_epoch_val_mse = float('inf')
                            best_model_state = None
                            patience_counter = 0
                            train_losses = []
                            val_losses = []
                            
                            for epoch in range(MAX_EPOCHS):
                                model.train()
                                train_loss = 0.0
                                for batch_X, batch_y in train_loader:
                                    optimizer.zero_grad()
                                    outputs = model(batch_X)
                                    loss = criterion(outputs, batch_y)
                                    loss.backward()
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                    optimizer.step()
                                    train_loss += loss.item()
                                
                                train_loss /= len(train_loader)
                                train_losses.append(train_loss)
                                
                                model.eval()
                                val_loss = 0.0
                                with torch.no_grad():
                                    for batch_X, batch_y in val_loader:
                                        outputs = model(batch_X)
                                        loss = criterion(outputs, batch_y)
                                        val_loss += loss.item()
                                
                                val_loss /= len(val_loader)
                                val_losses.append(val_loss)
                                
                                if val_loss < best_epoch_val_mse:
                                    best_epoch_val_mse = val_loss
                                    best_model_state = model.state_dict()
                                    patience_counter = 0
                                else:
                                    patience_counter += 1
                                
                                if patience_counter >= EARLY_STOPPING_PATIENCE:
                                    break
                            
                            print(f"  Best val MSE: {best_epoch_val_mse:.6f}")
                            
                            if best_epoch_val_mse < best_val_mse:
                                best_val_mse = best_epoch_val_mse
                                model.load_state_dict(best_model_state)
                                best_model = model
                                best_params = {
                                    'hidden_dims': hidden_dims,
                                    'learning_rate': lr,
                                    'batch_size': batch_size,
                                    'dropout': dropout,
                                    'weight_decay': weight_decay,
                                    'use_batch_norm': use_batch_norm
                                }
                                best_train_losses = train_losses
                                best_val_losses = val_losses
                                print(f"  *** NEW BEST CONFIG! ***")
    
    print(f"\n=== BEST CONFIGURATION ===")
    print(f"Best validation MSE: {best_val_mse:.6f}")
    print(f"Best parameters: {best_params}")
    
    print("\n=== EVALUATION ===")
    
    best_model.eval()
    with torch.no_grad():
        for split_name, X_split, y_split in [
            ('Train', X_train_tensor, y_train),
            ('Val', X_val_tensor, y_val),
            ('Test', X_test_tensor, y_test)
        ]:
            y_pred = best_model(X_split).cpu().numpy()
            y_pred = np.clip(y_pred, 0.0, 1.0)
            
            mse = mean_squared_error(y_split, y_pred)
            r2 = r2_score(y_split, y_pred)
            
            print(f"\n{split_name} Set:")
            print(f"  MSE: {mse:.6f}")
            print(f"  R²: {r2:.4f}")
            print(f"  Mean α (true): {y_split.mean():.4f}")
            print(f"  Mean α (pred): {y_pred.mean():.4f}")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "mlp_model.pt"
    
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'best_params': best_params,
        'train_losses': best_train_losses,
        'val_losses': best_val_losses
    }, model_path)
    
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
