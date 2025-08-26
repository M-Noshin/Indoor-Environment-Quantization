#!/usr/bin/env python
# coding: utf-8
"""
Indoor Environment Classification — PyTorch/ai8x
- Trains 20 times with different seeds
- First run (seed=42) matches model_max.py exactly
- Collects per-run results and saves summaries
"""

import os
import json
import pandas as pd

# Configuration
NUM_REPEATS = 20
SEEDS = [42 + i for i in range(NUM_REPEATS)]
OUTPUT_DIR = "seed_runs_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Storage for results across all runs
all_results = []

# Function to format the DataFrame with four decimal places for float values
def format_with_precision(df, precision=4):
    formatted_df = df.copy()
    for column in formatted_df.select_dtypes(include=['float']):
        formatted_df[column] = formatted_df[column].apply(lambda x: f"{x:.{precision}f}")
    return formatted_df

# ===== MAIN LOOP - Each iteration is like running model_max.py with different seed =====
for run_idx, seed in enumerate(SEEDS):
    print(f"\n{'='*80}")
    print(f"STARTING RUN {run_idx+1}/{NUM_REPEATS} with SEED {seed}")
    print(f"{'='*80}")
    
    # ===== SEED SETUP (MUST BE FIRST) =====
    import random
    import numpy as np
    
    # Set random seeds for reproducibility
    random.seed(seed)           # Python random module
    np.random.seed(seed)        # NumPy random
    
    # Set environment variable for complete reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # ===== IMPORTS (after basic seeding) =====
    import sys
    sys.path.append('/Users/hamza/Desktop/testMax/ai8x-training/')
    sys.path.append('/Users/hamza/Desktop/testMax/ai8x-training/')
    
    import torch
    torch.manual_seed(seed)     # PyTorch CPU random
    
    # Additional settings for complete determinism
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)           # PyTorch GPU random (current GPU)
        torch.cuda.manual_seed_all(seed)       # PyTorch GPU random (all GPUs)
    
    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    from torch.utils.data import TensorDataset, DataLoader
    from torchinfo import summary  # For model summary
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
    import torch.optim as optim
    from torch import nn
    import torch.nn.functional as F
    
    
    print(f"All random seeds set to {seed} for reproducible results")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")


    # ===== DATA LOADING (exactly like model_max.py) =====
    import scipy.io
    
    CTF_class_mov = scipy.io.loadmat('data/indoor_environment/CTF_Class_mov_final.mat')
    CTF_class_static = scipy.io.loadmat('data/indoor_environment/CTF_Class_static_final.mat')
    
    CTF_corridor_mov = scipy.io.loadmat('data/indoor_environment/CTF_Corridor_mov_final.mat')
    CTF_corridor_static = scipy.io.loadmat('data/indoor_environment/CTF_Corridor_static_final.mat')
    
    CTF_lab_mov = scipy.io.loadmat('data/indoor_environment/CTF_Lab_mov_final.mat')
    CTF_lab_static = scipy.io.loadmat('data/indoor_environment/CTF_Lab_static_final.mat')
    
    CTF_SC_mov = scipy.io.loadmat('data/indoor_environment/CTF_SC_mov_final.mat')
    CTF_SC_static = scipy.io.loadmat('data/indoor_environment/CTF_SC_static_final.mat')
    
    # Access the 4th item in each loaded variable
    #static
    CTF_class_static_array = CTF_class_static[list(CTF_class_static.keys())[3]].T
    CTF_corridor_static_array = CTF_corridor_static[list(CTF_corridor_static.keys())[3]].T
    CTF_lab_static_array = CTF_lab_static[list(CTF_lab_static.keys())[3]].T
    CTF_SC_static_array = CTF_SC_static[list(CTF_SC_static.keys())[3]].T
    
    # mov
    CTF_class_mov_array = CTF_class_mov[list(CTF_class_mov.keys())[3]].T
    CTF_corridor_mov_array = CTF_corridor_mov[list(CTF_corridor_mov.keys())[3]].T
    CTF_lab_mov_array = CTF_lab_mov[list(CTF_lab_mov.keys())[3]].T
    CTF_SC_mov_array = CTF_SC_mov[list(CTF_SC_mov.keys())[3]].T
    
    # Combine real and imaginary parts for the 4th item in each loaded variable using stack
    CTF_class = np.concatenate((CTF_class_static_array, CTF_class_mov_array), axis=0)
    CTF_corridor = np.concatenate((CTF_corridor_static_array, CTF_corridor_mov_array), axis=0)
    CTF_lab = np.concatenate((CTF_lab_static_array, CTF_lab_mov_array), axis=0)
    CTF_SC = np.concatenate((CTF_SC_static_array, CTF_SC_mov_array), axis=0)
    
    # Combine 145 training points randomly selected from each env dataset into a single training set array, 
    # and combine 49 testing points randomly selected from each dataset into a single testing set array.
    
    # Number of unique grid points
    num_grid_points = 194*2 #388
    
    # Number of measurements per grid point
    num_measurements = 200
    
    # Number of grid points to select for training
    num_train_points = int(0.75 * num_grid_points)  # 291
    
    # Initialize empty lists for training and test sets
    train_set = []
    test_set = []
    
    # For each array (representing a different environment)
    for array in [CTF_class, CTF_corridor, CTF_lab, CTF_SC]:
        # Reshape the array to separate the measurements for each grid point
        reshaped_array = array.reshape(num_grid_points, num_measurements, -1, 2)
    
        # Randomly select grid points for training
        train_points = random.sample(range(num_grid_points), num_train_points)
    
        # Get boolean array for test points
        test_points_bool = ~np.isin(range(num_grid_points), train_points)
    
        # Add the selected grid points to the training set and the rest to the test set
        train_set.append(reshaped_array[train_points])
        test_set.append(reshaped_array[test_points_bool])
    
    # Concatenate the data from all environments
    train_set = np.concatenate(train_set, axis=0)
    test_set = np.concatenate(test_set, axis=0)
    
    # print("Training set:",train_set.shape)
    # print("Testing set:",test_set.shape)
    
    # Reshaping the arrays
    large_X_train = train_set.reshape([-1,101,2])
    large_X_test = test_set.reshape([-1,101,2])
    # print("large_X_train after reshape:", large_X_train.shape)
    # print("large_X_test after reshape:", large_X_test.shape)
    
    # Create Labels for training data
    #static
    label1 = np.ones([291*200]) * 0  # class
    label2 = np.ones([291*200]) * 1  # corridor
    label3 = np.ones([291*200]) * 2  # lab
    label4 = np.ones([291*200]) * 3  # SC
    
    large_Y_train = np.concatenate([label1, label2, label3, label4])
    
    label5 = np.ones([97*200]) * 0  # class
    label6 = np.ones([97*200]) * 1  # corridor
    label7 = np.ones([97*200]) * 2  # lab
    label8 = np.ones([97*200]) * 3  # SC
    
    large_Y_test = np.concatenate([label5, label6, label7, label8])
    
    # print("Training labels:", large_Y_train.shape)
    # print("Testing labels:", large_Y_test.shape)
    
    # Shuffle Data
    shuffle_index1 = random.sample(range(0,232800), 232800)
    
    # Shuffle data using the shuffled indices
    large_X_train_new = large_X_train[shuffle_index1, :, :]
    # print(shuffle_index1[232797:])
    
    large_Y_train = large_Y_train[shuffle_index1]
    
    # Shuffle testing data
    shuffle_index2 = random.sample(range(0,77600), 77600)
    
    # Shuffle testing data using the shuffled indices
    large_X_test_new = large_X_test[shuffle_index2, :, :]
    # print(shuffle_index2[77597:])
    
    large_Y_test = large_Y_test[shuffle_index2]
    
    # Converting labels to categorical one-hot encoding
    y_train = F.one_hot(torch.tensor(large_Y_train, dtype=torch.long), num_classes=4).float()  # (232000, 4)
    y_test = F.one_hot(torch.tensor(large_Y_test, dtype=torch.long), num_classes=4).float()   # (78400, 4)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(large_X_train_new, dtype=torch.float32)  # (232000, 101, 2)
    X_test = torch.tensor(large_X_test_new, dtype=torch.float32)    # (77600, 101, 2)
    
    # print("Training labels:", y_train.shape)
    # print("Testing labels:", y_test.shape)
    # print("Training data:", X_train.shape)
    # print("Testing data:", X_test.shape)
    
    # ===== MODEL DEFINITION (exactly like model_max.py) =====
    class CNNModel(nn.Module):
        def __init__(self, p_dropout=0.4):
            super().__init__()
            # Keras convs used VALID padding with (3,1) → H: 101→99→97, W: stays 2
            self.conv1 = nn.Conv2d(1, 5, kernel_size=(3, 3), stride=1, padding=1, bias=True)
            self.bn1   = nn.BatchNorm2d(5, affine=True)
    
            self.conv2 = nn.Conv2d(5, 5, kernel_size=(3, 3), stride=1, padding=1, bias=True)
            self.bn2   = nn.BatchNorm2d(5, affine=True)
    
            # After conv2: (B, 32, 97, 2) → 32*97*2 = 6208
            self.fc1 = nn.Linear(5 * 101 * 2, 50, bias=True)
            self.fc2 = nn.Linear(50, 4, bias=True)
    
            self.relu = nn.ReLU()
            # self.relu = nn.Hardtanh(min_val=-128/128, max_val=127/128)
            # self.dropout = nn.Dropout(p=p_dropout)
    
        def forward(self, x):
            # x: (B, 101, 2) → (B, 1, 101, 2)
            x = x.unsqueeze(1)
    
            x = self.relu(self.bn1(self.conv1(x)))   # (B,32,99,2)
            x = self.relu(self.bn2(self.conv2(x)))   # (B,32,97,2)
    
            x = x.view(x.size(0), -1)               # (B, 6208)
            x = self.relu(self.fc1(x))               # (B, 200)
            # x = self.dropout(x)
            x = self.fc2(x)                         # (B, 4)
            return x
        
    
    # Quick shape & parameter check
    model = CNNModel()
    dummy = torch.randn(1, 101, 2)
    with torch.no_grad():
        out = model(dummy)
    # print("Output shape:", out.shape)  # torch.Size([1, 4])
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params:", total_params)
    print("Trainable params:", trainable_params)
    
    # print(model)
    
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # print(f"Total parameters: {total_params}")
    # print(f"Trainable parameters: {trainable_params}")
    
    # Quick shape verification
    model = CNNModel()
    dummy_input = torch.randn(1, 101, 2)  # Single sample
    
    # print("=== SHAPE VERIFICATION ===")
    # print(f"Input to model: {dummy_input.shape}")
    
    # # Run one forward pass to see shapes
    # with torch.no_grad():
    #     output = model(dummy_input)
    
    # print(f"Output from model: {output.shape}")
    # print("Expected: torch.Size([1, 4])")  # 1 sample, 
    
    def lr_schedule(epoch):
        lr = 0.001
        if epoch > 4:        # After epoch 4 (epoch 5+)
            lr *= 1e-3       # lr = 0.001 * 0.001 = 0.000001
        elif epoch > 2:      # After epoch 2 (epoch 3-4)
            lr *= 0.5e-3     # lr = 0.001 * 0.0005 = 0.0000005
        return lr
    
    # Set up optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr_schedule(0))
    criterion = nn.CrossEntropyLoss()
    
    # Split training data into train (90%) and validation (10%)
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.1, 
        random_state=42, 
        stratify=torch.argmax(y_train, dim=1)  # Stratify by class to maintain class balance
    )
    
    print(f"Original training set: {X_train.shape}")
    print(f"New training set: {X_train_split.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Create data loaders
    batch_size = 256
    train_dataset = TensorDataset(X_train_split, y_train_split)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # ===== TRAINING FUNCTION (exactly like model_max.py) =====
    def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=50, 
                    patience=10, min_delta=0.001, monitor='val_loss', restore_best_weights=True):
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
    
        # Early stopping variables
        best_metric = float('inf') if monitor == 'val_loss' else 0.0
        epochs_without_improvement = 0
        best_model_state = None
    
        print(f"Early stopping enabled: patience={patience}, min_delta={min_delta}, monitor='{monitor}'")
    
        for epoch in range(num_epochs):
            # Update learning rate
            new_lr = lr_schedule(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
    
            print(f'\nEpoch [{epoch+1}/{num_epochs}], Learning Rate: {new_lr:.6f}')
            print('-' * 60)
    
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
    
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
    
                # Convert one-hot to class indices for CrossEntropyLoss
                target_indices = torch.argmax(target, dim=1)
                loss = criterion(output, target_indices)
    
                loss.backward()
                optimizer.step()
    
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target_indices).sum().item()
    
                # Print progress every 1000 batches
                if batch_idx % 1000 == 0:
                    current_acc = 100. * train_correct / train_total if train_total > 0 else 0
                    print(f'  Batch [{batch_idx:4d}/{len(train_loader)}] | '
                          f'Loss: {loss.item():.4f} | '
                          f'Acc: {current_acc:.2f}% | '
                          f'Processed: {train_total}/{len(train_loader.dataset)}')
    
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
    
            print(f'  Running validation...')
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(val_loader):
                    output = model(data)
                    target_indices = torch.argmax(target, dim=1)
                    loss = criterion(output, target_indices)
    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target_indices).sum().item()
    
                    # Print validation progress every 500 batches
                    if batch_idx % 500 == 0:
                        current_val_acc = 100. * val_correct / val_total if val_total > 0 else 0
                        print(f'    Val Batch [{batch_idx:3d}/{len(val_loader)}] | '
                              f'Loss: {loss.item():.4f} | '
                              f'Acc: {current_val_acc:.2f}%')
    
            # Calculate averages
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
    
            # Store metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
    
            # Early stopping logic
            current_metric = val_loss if monitor == 'val_loss' else val_acc
    
            if monitor == 'val_loss':
                # For validation loss, lower is better
                improvement = best_metric - current_metric > min_delta
            else:
                # For validation accuracy, higher is better
                improvement = current_metric - best_metric > min_delta
    
            if improvement:
                best_metric = current_metric
                epochs_without_improvement = 0
                if restore_best_weights:
                    best_model_state = model.state_dict().copy()
                print(f'  *** New best {monitor}: {current_metric:.4f} ***')
            else:
                epochs_without_improvement += 1
    
            print(f'\n  EPOCH {epoch+1} SUMMARY:')
            print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%')
            print(f'  Best {monitor}: {best_metric:.4f} | Epochs w/o improvement: {epochs_without_improvement}/{patience}')
            print('=' * 60)
    
            # Check for early stopping
            if epochs_without_improvement >= patience:
                print(f'\n*** EARLY STOPPING ***')
                print(f'No improvement in {monitor} for {patience} epochs.')
                print(f'Best {monitor} was: {best_metric:.4f}')
                break
    
        # Restore best weights if requested
        if restore_best_weights and best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f'\nRestored model weights from best epoch (best {monitor}: {best_metric:.4f})')
    
        return train_losses, train_accuracies, val_losses, val_accuracies
    
    # Train the model with early stopping
    history = train_model(
        model, train_loader, val_loader, optimizer, criterion, 
        num_epochs=10,           # Maximum epochs
        patience=10,             # Stop if no improvement for 10 epochs
        min_delta=0.001,         # Minimum improvement threshold
        monitor='val_loss',      # Monitor validation loss (or 'val_acc' for accuracy)
        restore_best_weights=True # Restore best weights at the end
    )
    
    # ===== EVALUATION (exactly like model_max.py) =====
    def evaluate_model(model, test_loader, criterion):
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_targets = []
    
        print("Evaluating model on test set...")
        print("-" * 50)
    
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                output = model(data)
                target_indices = torch.argmax(target, dim=1)
                loss = criterion(output, target_indices)
    
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target_indices).sum().item()
    
                # Store predictions and targets for detailed analysis
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target_indices.cpu().numpy())
    
                # Print progress every 100 batches
                if batch_idx % 100 == 0:
                    current_acc = 100. * test_correct / test_total if test_total > 0 else 0
                    print(f'  Test Batch [{batch_idx:3d}/{len(test_loader)}] | '
                          f'Loss: {loss.item():.4f} | '
                          f'Acc: {current_acc:.2f}%')
    
        # Calculate final metrics
        test_loss /= len(test_loader)
        test_acc = 100. * test_correct / test_total
    
        print("\n" + "="*50)
        print("FINAL TEST RESULTS:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Correct Predictions: {test_correct}/{test_total}")
        print("="*50)
    
        return test_loss, test_acc, all_predictions, all_targets
    
    # Run evaluation
    test_loss, test_acc, predictions, targets = evaluate_model(model, test_loader, criterion)
    
    # Calculate basic metrics for summary
    class_names = ['Classroom', 'Corridor', 'Lab', 'Sports-Complex']
    report_dict = classification_report(targets, predictions, target_names=class_names, digits=4, output_dict=True)
    
    # ===== STORE RESULTS FOR THIS RUN =====
    run_result = {
        'run': run_idx + 1,
        'seed': seed,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'macro_f1': report_dict['macro avg']['f1-score'],
        'weighted_f1': report_dict['weighted avg']['f1-score']
    }
    
    all_results.append(run_result)
    
    # Update Excel file after each run
    current_df = pd.DataFrame(all_results)
    formatted_df = format_with_precision(current_df, precision=4)
    
    # Initialize Excel writer
    excel_path = os.path.join(OUTPUT_DIR, 'experiment_results_20_seeds.xlsx')
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as excel_writer:
        formatted_df.to_excel(excel_writer, sheet_name='Results', index=False)
        
        # Get workbook and worksheet objects
        workbook = excel_writer.book
        worksheet = excel_writer.sheets['Results']
        
        # Add some formatting
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Write headers with formatting
        for col_num, value in enumerate(formatted_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
    
    print(f"\nCompleted run {run_idx+1}/{NUM_REPEATS} (seed={seed})")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Results updated in: {excel_path}")

# ===== AGGREGATE RESULTS ACROSS ALL RUNS =====
print(f"\n{'='*80}")
print("AGGREGATE RESULTS ACROSS ALL RUNS")
print(f"{'='*80}")

# Convert to DataFrame
results_df = pd.DataFrame(all_results)

# Save detailed results as CSV backup
results_df.to_csv(os.path.join(OUTPUT_DIR, "all_runs_results.csv"), index=False)

# Print summary statistics
print("\nSUMMARY STATISTICS:")
print(results_df[['test_accuracy', 'macro_f1', 'weighted_f1']].describe())

print(f"\nBest accuracy: {results_df['test_accuracy'].max():.2f}% (seed {results_df.loc[results_df['test_accuracy'].idxmax(), 'seed']})")
print(f"Worst accuracy: {results_df['test_accuracy'].min():.2f}% (seed {results_df.loc[results_df['test_accuracy'].idxmin(), 'seed']})")
print(f"Mean accuracy: {results_df['test_accuracy'].mean():.2f}% ± {results_df['test_accuracy'].std():.2f}%")

print(f"\nResults saved to {OUTPUT_DIR}/")
print("Files created:")
print("- experiment_results_20_seeds.xlsx: Main results file (updated after each run)")
print("- all_runs_results.csv: CSV backup of all results")

print(f"\n{'='*80}")
print("ALL RUNS COMPLETED!")
print(f"{'='*80}")