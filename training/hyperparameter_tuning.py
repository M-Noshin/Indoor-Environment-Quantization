#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for QAT Policy Parameters
Automatically tunes QAT policy parameters to find the best combination for indoor environment model.
"""

import os
import sys
import json
import yaml
import re
import subprocess
import time
import shutil
import argparse
from datetime import datetime
from itertools import product
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Note: pandas not available. CSV export will be basic.")


class QATHyperparameterTuner:
    """Hyperparameter tuning for QAT policy parameters."""
    
    def __init__(self, base_policy_file, training_script, results_dir="tuning_results"):
        self.base_policy_file = base_policy_file
        self.training_script = training_script
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Results tracking
        self.results = []
        self.best_result = None
        self.results_file = self.results_dir / "tuning_results.json"
        self.csv_file = self.results_dir / "tuning_results.csv"
        
        # Load base policy
        with open(base_policy_file, 'r') as f:
            self.base_policy = yaml.safe_load(f)
            
    def define_parameter_grid(self):
        """Define the parameter search space."""
        # Based on research and common practices for QAT
        param_grid = {
            'start_epoch': [8],  # When to start QAT
            'weight_bits': [8],  # Keep at 8 for MAX78002
            # 'shift_quantile': [0.95, 0.985, 0.99, 0.995, 0.999, 1.0],  # Parameter distribution quantile
            # 'outlier_removal_z_score': [2.0, 4.0, 6.0, 8.0, 10.0]  # Outlier removal threshold
            'shift_quantile': [0.999, 1.0],  # Parameter distribution quantile
            'outlier_removal_z_score': [2.0, 4.0, 6.0, 8.0, 10.0]  # Outlier removal threshold
        }
        return param_grid
    
    def generate_policy_file(self, params, run_id):
        """Generate a QAT policy file with specific parameters."""
        policy = self.base_policy.copy()
        policy.update(params)
        
        policy_file = self.results_dir / f"qat_policy_run_{run_id}.yaml"
        
        with open(policy_file, 'w') as f:
            f.write("---\n")
            f.write(f"# QAT Policy for Run {run_id}\n")
            f.write(f"# start_epoch: {params['start_epoch']}, shift_quantile: {params['shift_quantile']}, z_score: {params['outlier_removal_z_score']}\n")
            f.write("\n")
            yaml.dump(policy, f, default_flow_style=False)
            
        return str(policy_file)
    
    def run_training(self, policy_file, run_id):
        """Run training with specified policy file."""
        print(f"\n{'='*60}")
        print(f"Starting Training Run {run_id}")
        print(f"Policy file: {policy_file}")
        print(f"{'='*60}")
        
        # Modify training command to use our policy file
        cmd = [
            'python', 'train.py',
            '--epochs', '10',
            '--batch-size', '256',
            '--optimizer', 'Adam',
            '--lr', '0.001',
            '--weight-decay', '0.0002',
            '--use-bias',
            '--deterministic',
            '--model', 'ai85indoorenvnetv1',
            '--dataset', 'IndoorEnvironment',
            '--data', 'data/indoor_environment',
            '--compress', 'policies/schedule-indoor-env.yaml',
            '--qat-policy', policy_file,
            '--device', 'MAX78002',
            '--name', f'tuning_run_{run_id}'
        ]
        
        print(f"🏃 Running: {' '.join(cmd[:3])} ... (full command with {len(cmd)} args)")
        start_time = time.time()
        
        try:
            # Run training with real-time output
            process = subprocess.Popen(
                cmd,
                cwd='.',  # Run in current directory where train.py is located
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            output_lines = []
            print("📊 Training output (showing key lines):")
            
            # Read output line by line
            for line in iter(process.stdout.readline, ''):
                output_lines.append(line)
                # Print important lines to show progress
                if any(keyword in line for keyword in ['Epoch:', 'Top1:', 'Loss:', 'Best', 'test', 'QAT', 'validate']):
                    print(f"   {line.strip()}")
            
            process.wait()
            training_time = time.time() - start_time
            full_output = ''.join(output_lines)
            
            if process.returncode == 0:
                # Parse results from output
                metrics = self.parse_training_output(full_output)
                metrics['training_time'] = training_time
                metrics['success'] = True
                print(f"✅ Training completed successfully in {training_time:.1f}s")
                print(f"📊 Results: Top1={metrics.get('test_top1', 'N/A')}, Loss={metrics.get('test_loss', 'N/A')}")
            else:
                print(f"❌ Training failed with return code: {process.returncode}")
                # Show last few lines of output for debugging
                if output_lines:
                    print("Last few lines of output:")
                    for line in output_lines[-5:]:
                        print(f"   {line.strip()}")
                metrics = {
                    'training_time': training_time,
                    'success': False,
                    'error': f'Return code: {process.returncode}'
                }
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Training timed out after 1 hour")
            process.kill()
            metrics = {
                'training_time': 3600,
                'success': False,
                'error': 'Training timeout'
            }
        except Exception as e:
            print(f"💥 Training failed with exception: {e}")
            metrics = {
                'training_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }
            
        return metrics
    
    def parse_training_output(self, output):
        """Parse training output to extract final metrics."""
        metrics = {}
        
        # Look for final test results
        test_patterns = [
            r'==> Top1: ([\d.]+)\s+Top5: ([\d.]+)\s+Loss: ([\d.]+)',
            r'==> Top1: ([\d.]+)\s+Loss: ([\d.]+)',
        ]
        
        # Look for best validation results
        best_patterns = [
            r'==> Best \[Top1: ([\d.]+)\s+Top5: ([\d.]+).*?on epoch: (\d+)\]',
            r'==> Best \[Top1: ([\d.]+).*?on epoch: (\d+)\]'
        ]
        
        # Extract test results (most recent)
        for pattern in test_patterns:
            matches = re.findall(pattern, output)
            if matches:
                last_match = matches[-1]
                if len(last_match) == 3:
                    metrics['test_top1'] = float(last_match[0])
                    metrics['test_top5'] = float(last_match[1])
                    metrics['test_loss'] = float(last_match[2])
                elif len(last_match) == 2:
                    metrics['test_top1'] = float(last_match[0])
                    metrics['test_loss'] = float(last_match[1])
                break
                
        # Extract best validation results
        for pattern in best_patterns:
            matches = re.findall(pattern, output)
            if matches:
                last_match = matches[-1]
                if len(last_match) == 3:
                    metrics['best_val_top1'] = float(last_match[0])
                    metrics['best_val_top5'] = float(last_match[1])
                    metrics['best_epoch'] = int(last_match[2])
                elif len(last_match) == 2:
                    metrics['best_val_top1'] = float(last_match[0])
                    metrics['best_epoch'] = int(last_match[1])
                break
        
        return metrics
    
    def save_results(self):
        """Save results to JSON and CSV files."""
        # Save to JSON
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        # Save to CSV
        if HAS_PANDAS and self.results:
            try:
                df = pd.DataFrame(self.results)
                df.to_csv(self.csv_file, index=False)
            except Exception as e:
                print(f"Warning: Could not save CSV with pandas: {e}")
                self._save_basic_csv()
        elif self.results:
            self._save_basic_csv()
            
        print(f"📄 Results saved to {self.results_file}")
        if self.csv_file.exists():
            print(f"📄 CSV results saved to {self.csv_file}")
    
    def _save_basic_csv(self):
        """Save basic CSV without pandas."""
        if not self.results:
            return
            
        try:
            with open(self.csv_file, 'w') as f:
                # Header
                first_result = self.results[0]
                headers = ['run_id', 'timestamp', 'success', 'training_time']
                
                # Add parameter headers
                if 'parameters' in first_result:
                    for key in first_result['parameters'].keys():
                        headers.append(f'param_{key}')
                
                # Add metric headers
                metric_keys = ['test_top1', 'test_top5', 'test_loss', 'best_val_top1', 'best_val_top5', 'best_epoch']
                headers.extend(metric_keys)
                
                f.write(','.join(headers) + '\n')
                
                # Data rows
                for result in self.results:
                    row = []
                    row.append(str(result.get('run_id', '')))
                    row.append(str(result.get('timestamp', '')))
                    row.append(str(result.get('success', False)))
                    row.append(str(result.get('training_time', '')))
                    
                    # Parameters
                    if 'parameters' in result:
                        for key in first_result['parameters'].keys():
                            row.append(str(result['parameters'].get(key, '')))
                    
                    # Metrics
                    for key in metric_keys:
                        row.append(str(result.get(key, '')))
                    
                    f.write(','.join(row) + '\n')
        except Exception as e:
            print(f"Warning: Could not save basic CSV: {e}")
    
    def print_summary(self):
        """Print summary of results."""
        if not self.results:
            print("No results to summarize.")
            return
            
        successful_runs = [r for r in self.results if r.get('success', False)]
        
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER TUNING SUMMARY")
        print(f"{'='*60}")
        print(f"Total runs: {len(self.results)}")
        print(f"Successful runs: {len(successful_runs)}")
        print(f"Failed runs: {len(self.results) - len(successful_runs)}")
        
        if successful_runs:
            # Sort by test accuracy (or validation accuracy if test not available)
            successful_runs.sort(
                key=lambda x: x.get('test_top1', x.get('best_val_top1', 0)), 
                reverse=True
            )
            
            print(f"\n🏆 TOP 5 RESULTS:")
            print("-" * 60)
            
            for i, result in enumerate(successful_runs[:5], 1):
                params = result['parameters']
                test_acc = result.get('test_top1', result.get('best_val_top1', 'N/A'))
                test_loss = result.get('test_loss', 'N/A')
                
                print(f"{i}. Run {result['run_id']}")
                print(f"   Test Accuracy: {test_acc}")
                print(f"   Test Loss: {test_loss}")
                print(f"   Parameters: start_epoch={params['start_epoch']}, "
                      f"shift_quantile={params['shift_quantile']}, "
                      f"z_score={params['outlier_removal_z_score']}")
                print(f"   Training time: {result.get('training_time', 'N/A'):.1f}s")
                print()
                
            # Best result
            best = successful_runs[0]
            self.best_result = best
            
            print(f"🥇 BEST CONFIGURATION:")
            print(f"   Parameters: {json.dumps(best['parameters'], indent=4)}")
            print(f"   Test Accuracy: {best.get('test_top1', best.get('best_val_top1', 'N/A'))}")
            print(f"   Test Loss: {best.get('test_loss', 'N/A')}")
            
            # Save best configuration
            best_policy_file = self.results_dir / "best_qat_policy.yaml"
            with open(best_policy_file, 'w') as f:
                f.write("---\n")
                f.write("# Best QAT Policy Configuration\n")
                f.write(f"# Found through hyperparameter tuning\n")
                f.write(f"# Test Accuracy: {best.get('test_top1', best.get('best_val_top1', 'N/A'))}\n")
                f.write(f"# Test Loss: {best.get('test_loss', 'N/A')}\n\n")
                yaml.dump(best['parameters'], f, default_flow_style=False)
                
            print(f"💾 Best configuration saved to {best_policy_file}")
    
    def run_tuning(self, max_runs=None):
        """Run the hyperparameter tuning process."""
        param_grid = self.define_parameter_grid()
        
        # Generate all parameter combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        
        if max_runs:
            combinations = combinations[:max_runs]
            
        total_runs = len(combinations)
        print(f"🚀 Starting hyperparameter tuning with {total_runs} combinations...")
        print(f"📁 Results will be saved to: {self.results_dir}")
        
        for run_id, combination in enumerate(combinations, 1):
            params = dict(zip(keys, combination))
            
            print(f"\n{'='*60}")
            print(f"Run {run_id}/{total_runs}")
            print(f"Parameters: {params}")
            
            # Generate policy file
            policy_file = self.generate_policy_file(params, run_id)
            
            # Run training
            metrics = self.run_training(policy_file, run_id)
            
            # Store results
            result = {
                'run_id': run_id,
                'timestamp': datetime.now().isoformat(),
                'parameters': params,
                **metrics
            }
            
            self.results.append(result)
            
            # Save intermediate results
            self.save_results()
            
            # Clean up policy file
            try:
                os.remove(policy_file)
            except:
                pass
                
        # Final summary
        self.print_summary()


def main():
    parser = argparse.ArgumentParser(description='QAT Hyperparameter Tuning')
    parser.add_argument('--policy-file', default='policies/qat_policy_indoor.yaml',
                       help='Base QAT policy file')
    parser.add_argument('--training-script', default='scripts/train_indoor.sh',
                       help='Training script path')
    parser.add_argument('--results-dir', default='tuning_results',
                       help='Directory to save results')
    parser.add_argument('--max-runs', type=int,
                       help='Maximum number of runs (for testing)')
    
    args = parser.parse_args()
    
    # Change to training directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Initialize tuner
    tuner = QATHyperparameterTuner(
        base_policy_file=args.policy_file,
        training_script=args.training_script,
        results_dir=args.results_dir
    )
    
    # Run tuning
    try:
        tuner.run_tuning(max_runs=args.max_runs)
    except KeyboardInterrupt:
        print("\n⚠️  Tuning interrupted by user")
        tuner.print_summary()
    except Exception as e:
        print(f"💥 Tuning failed with error: {e}")
        tuner.print_summary()
        raise


if __name__ == '__main__':
    main() 