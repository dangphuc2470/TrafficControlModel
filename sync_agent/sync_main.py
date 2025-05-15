from sync_model import SyncDRLModel
from sync_environment import IntersectionSyncEnv
from sync_trainer import SyncTrainer
from utils import ReplayBuffer
import os
import time
import logging
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run the traffic synchronization DRL agent')
    parser.add_argument('--model-dir', type=str, default='sync_models',
                        help='Directory to save/load models')
    parser.add_argument('--data-path', type=str, 
                        default='../central_server_old/server_data/agent_data.json',
                        help='Path to agent data JSON from central server')
    parser.add_argument('--output-path', type=str,
                        default='../central_server_old/server_data/sync_times.json',
                        help='Path to save synchronization data')
    parser.add_argument('--update-interval', type=int, default=60,
                        help='Seconds between updates from agent data')
    parser.add_argument('--train-interval', type=int, default=30,
                        help='Seconds between training batches')
    parser.add_argument('--save-interval', type=int, default=3600,
                        help='Seconds between model saves')
    parser.add_argument('--batch-size', type=int, default=2, help='Training batch size')
    parser.add_argument('--min-buffer-size', type=int, default=2, help='Minimum buffer size before training')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = SyncTrainer(
        model_dir=args.model_dir,
        data_path=args.data_path,
        output_path=args.output_path,
        update_interval=args.update_interval,
        train_interval=args.train_interval,
        save_interval=args.save_interval,
        batch_size=args.batch_size,
        min_buffer_size=args.min_buffer_size
    )
    
    # Start training
    try:
        trainer.start()
        print(f"Sync trainer started. Press Ctrl+C to stop.")
        
        # Keep process running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Stopping trainer...")
        trainer.stop()
        print("Trainer stopped.")

if __name__ == "__main__":
    main()