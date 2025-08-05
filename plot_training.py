import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Path to the log file
log_path = "DnCNN-keras/snapshot/save_DnCNN_sigma25_2025-08-02-21-12-53/log.csv"

# Check if file exists
if not os.path.exists(log_path):
    print(f"Error: File not found at {log_path}")
    exit(1)

try:
    # Read the CSV file
    df = pd.read_csv(log_path)
    
    # Print first few rows to debug
    print("First few rows of the log file:")
    print(df.head())
    
    # Convert 'NA' strings to NaN
    df = df.replace('NA', np.nan)
    
    # Ensure we have training loss data to plot
    if 'loss' not in df.columns or df['loss'].isna().all():
        print("No valid training loss data found in the log file.")
        exit(1)
        
    # Fill NaN values in loss with previous value (if any)
    df['loss'] = pd.to_numeric(df['loss'], errors='coerce').fillna(method='ffill')
    
    # Create the plot with larger figure size
    plt.figure(figsize=(14, 7))
    
    # Plot training loss
    plt.plot(df.index, df['loss'], 'b-', label='Training Loss', linewidth=2, marker='o', markersize=5)
    
    # Add labels and title with larger font sizes
    plt.title('DnCNN Training Loss Over Epochs (σ=25)', fontsize=16, pad=20)
    plt.xlabel('Epoch', fontsize=14, labelpad=10)
    plt.ylabel('Loss (MSE)', fontsize=14, labelpad=10)
    
    # Customize ticks
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Use log scale for y-axis to better visualize the loss
    plt.yscale('log')
    
    # Add legend
    plt.legend(fontsize=12, frameon=True, fancybox=True, framealpha=0.9)
    
    # Add some padding around the plot
    plt.tight_layout(pad=2.0)
    
    # Save the figure with high DPI
    output_path = 'DnCNN_training_loss.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training plot saved as '{os.path.abspath(output_path)}'")
    
    # Show the plot
    plt.show()
    
except Exception as e:
    print(f"Error: {str(e)}")
    print("\nMake sure the log file has the correct format with 'epoch' and 'loss' columns.")
