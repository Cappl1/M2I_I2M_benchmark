import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle

# Load the data
with open('/home/brothen/M2I_I2M_benchmark/hyperparam_results/hyperparam_results.json', 'r') as f:
    experiments = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(experiments)

# Expand config dictionary into separate columns
config_df = pd.json_normalize(df['config'])
df = pd.concat([df.drop('config', axis=1), config_df], axis=1)

print(f"Total experiments: {len(df)}")
print(f"SSL Methods: {df['ssl_method'].unique()}")
print(f"Best overall accuracy: {df['accuracy'].max():.2f}%")

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")
fig = plt.figure(figsize=(20, 24))

# 1. Overall performance comparison
ax1 = plt.subplot(4, 3, 1)
sns.boxplot(data=df, x='ssl_method', y='accuracy', ax=ax1)
plt.title('Overall Accuracy Distribution by SSL Method', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)')

# Add mean values as text
for i, ssl_method in enumerate(df['ssl_method'].unique()):
    mean_acc = df[df['ssl_method'] == ssl_method]['accuracy'].mean()
    plt.text(i, mean_acc + 1, f'μ={mean_acc:.1f}%', ha='center', fontweight='bold')

# 2. Performance by optimizer
ax2 = plt.subplot(4, 3, 2)
sns.boxplot(data=df, x='optimizer', y='accuracy', hue='ssl_method', ax=ax2)
plt.title('Accuracy by Optimizer and SSL Method', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)')

# 3. Performance by learning rate
ax3 = plt.subplot(4, 3, 3)
sns.boxplot(data=df, x='lr', y='accuracy', hue='ssl_method', ax=ax3)
plt.title('Accuracy by Learning Rate and SSL Method', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.xticks(rotation=45)

# 4. SimCLR Heatmap - Accuracy by hyperparameters
simclr_df = df[df['ssl_method'] == 'simclr'].copy()

# Create pivot table for SimCLR heatmap
simclr_pivot = simclr_df.groupby(['lr', 'optimizer', 'ssl_temperature', 'kd_classifier_weight'])['accuracy'].mean().reset_index()

ax4 = plt.subplot(4, 3, 4)
# Create a more detailed heatmap for SimCLR
simclr_heatmap_data = simclr_df.pivot_table(
    values='accuracy', 
    index=['optimizer', 'lr'], 
    columns=['ssl_temperature', 'kd_classifier_weight'],
    aggfunc='mean'
)

sns.heatmap(simclr_heatmap_data, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax4, cbar_kws={'label': 'Accuracy (%)'})
plt.title('SimCLR: Accuracy Heatmap\n(Optimizer+LR vs Temperature+KD_Weight)', fontsize=12, fontweight='bold')
plt.xlabel('SSL Temperature, KD Weight')
plt.ylabel('Optimizer, Learning Rate')

# 5. MoCo Heatmap - Accuracy by hyperparameters
moco_df = df[df['ssl_method'] == 'moco'].copy()

ax5 = plt.subplot(4, 3, 5)
moco_heatmap_data = moco_df.pivot_table(
    values='accuracy', 
    index=['optimizer', 'lr'], 
    columns=['kd_classifier_weight', 'minibatch_size'],
    aggfunc='mean'
)

sns.heatmap(moco_heatmap_data, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax5, cbar_kws={'label': 'Accuracy (%)'})
plt.title('MoCo: Accuracy Heatmap\n(Optimizer+LR vs KD_Weight+Batch_Size)', fontsize=12, fontweight='bold')
plt.xlabel('KD Weight, Batch Size')
plt.ylabel('Optimizer, Learning Rate')

# 6. Training time comparison
ax6 = plt.subplot(4, 3, 6)
sns.scatterplot(data=df, x='train_time', y='accuracy', hue='ssl_method', style='optimizer', s=60, ax=ax6)
plt.title('Accuracy vs Training Time', fontsize=14, fontweight='bold')
plt.xlabel('Training Time (seconds)')
plt.ylabel('Accuracy (%)')

# 7. Loss components analysis
ax7 = plt.subplot(4, 3, 7)
loss_data = df[['ssl_method', 'ssl_loss', 'ce_loss']].melt(
    id_vars=['ssl_method'], 
    value_vars=['ssl_loss', 'ce_loss'],
    var_name='loss_type', 
    value_name='loss_value'
)
sns.boxplot(data=loss_data, x='loss_type', y='loss_value', hue='ssl_method', ax=ax7)
plt.title('Loss Components by SSL Method', fontsize=14, fontweight='bold')
plt.ylabel('Loss Value')
plt.yscale('log')

# 8. Top performers table
ax8 = plt.subplot(4, 3, 8)
ax8.axis('off')

# Get top 10 experiments
top_experiments = df.nlargest(10, 'accuracy')[['ssl_method', 'optimizer', 'lr', 'accuracy', 'kd_classifier_weight', 'minibatch_size']]

# For SimCLR, add temperature
simclr_top = top_experiments[top_experiments['ssl_method'] == 'simclr'].copy()
if len(simclr_top) > 0:
    simclr_details = df[df['ssl_method'] == 'simclr'].nlargest(5, 'accuracy')[['ssl_temperature']]
    
table_text = "🏆 TOP 10 CONFIGURATIONS\n\n"
for idx, (_, row) in enumerate(top_experiments.iterrows(), 1):
    ssl_method = row['ssl_method']
    if ssl_method == 'simclr':
        # Get temperature for this specific experiment
        temp = df.iloc[_]['ssl_temperature'] if 'ssl_temperature' in df.columns else 'N/A'
        table_text += f"{idx:2d}. {ssl_method.upper()}: {row['accuracy']:.2f}% | {row['optimizer']} | lr={row['lr']} | T={temp} | KD={row['kd_classifier_weight']}\n"
    else:
        table_text += f"{idx:2d}. {ssl_method.upper()}: {row['accuracy']:.2f}% | {row['optimizer']} | lr={row['lr']} | KD={row['kd_classifier_weight']}\n"

plt.text(0.05, 0.95, table_text, transform=ax8.transAxes, fontsize=10, 
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

# 9. Epochs effect
ax9 = plt.subplot(4, 3, 9)
sns.boxplot(data=df, x='epochs', y='accuracy', hue='ssl_method', ax=ax9)
plt.title('Accuracy by Training Epochs', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)')

# 10. SimCLR Temperature Analysis (detailed)
ax10 = plt.subplot(4, 3, 10)
if 'ssl_temperature' in df.columns:
    simclr_temp_data = df[df['ssl_method'] == 'simclr']
    sns.boxplot(data=simclr_temp_data, x='ssl_temperature', y='accuracy', ax=ax10)
    plt.title('SimCLR: Accuracy by SSL Temperature', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('SSL Temperature')

# 11. Batch size effect
ax11 = plt.subplot(4, 3, 11)
sns.boxplot(data=df, x='minibatch_size', y='accuracy', hue='ssl_method', ax=ax11)
plt.title('Accuracy by Batch Size', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)')
plt.xlabel('Batch Size')

# 12. Summary statistics
ax12 = plt.subplot(4, 3, 12)
ax12.axis('off')

# Calculate summary statistics
simclr_stats = df[df['ssl_method'] == 'simclr']['accuracy']
moco_stats = df[df['ssl_method'] == 'moco']['accuracy']

summary_text = f"""
📊 SUMMARY STATISTICS

SimCLR:
  • Best: {simclr_stats.max():.2f}%
  • Mean: {simclr_stats.mean():.2f}%
  • Std:  {simclr_stats.std():.2f}%
  • Experiments: {len(simclr_stats)}

MoCo:
  • Best: {moco_stats.max():.2f}%
  • Mean: {moco_stats.mean():.2f}%
  • Std:  {moco_stats.std():.2f}%
  • Experiments: {len(moco_stats)}

🎯 BEST OVERALL:
  • Method: {df.loc[df['accuracy'].idxmax(), 'ssl_method'].upper()}
  • Accuracy: {df['accuracy'].max():.2f}%
  • Config: {df.loc[df['accuracy'].idxmax(), 'optimizer']} + lr={df.loc[df['accuracy'].idxmax(), 'lr']}
"""

plt.text(0.05, 0.95, summary_text, transform=ax12.transAxes, fontsize=11, 
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

plt.tight_layout(pad=3.0)
plt.savefig('ssl_experiment_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional detailed analysis
print("\n" + "="*50)
print("DETAILED ANALYSIS")
print("="*50)

# Best configuration for each SSL method
print(f"\n🏆 BEST SimCLR Configuration:")
best_simclr = df[df['ssl_method'] == 'simclr'].loc[df[df['ssl_method'] == 'simclr']['accuracy'].idxmax()]
print(f"   Accuracy: {best_simclr['accuracy']:.2f}%")
print(f"   Optimizer: {best_simclr['optimizer']}")
print(f"   Learning Rate: {best_simclr['lr']}")
if 'ssl_temperature' in best_simclr:
    print(f"   SSL Temperature: {best_simclr['ssl_temperature']}")
print(f"   KD Weight: {best_simclr['kd_classifier_weight']}")
print(f"   Batch Size: {best_simclr['minibatch_size']}")
print(f"   Epochs: {best_simclr['epochs']}")

print(f"\n🏆 BEST MoCo Configuration:")
best_moco = df[df['ssl_method'] == 'moco'].loc[df[df['ssl_method'] == 'moco']['accuracy'].idxmax()]
print(f"   Accuracy: {best_moco['accuracy']:.2f}%")
print(f"   Optimizer: {best_moco['optimizer']}")
print(f"   Learning Rate: {best_moco['lr']}")
print(f"   KD Weight: {best_moco['kd_classifier_weight']}")
print(f"   Batch Size: {best_moco['minibatch_size']}")
print(f"   Epochs: {best_moco['epochs']}")

# Performance by key factors
print(f"\n📈 PERFORMANCE BY OPTIMIZER:")
perf_by_opt = df.groupby(['ssl_method', 'optimizer'])['accuracy'].agg(['mean', 'max', 'count']).round(2)
print(perf_by_opt)

print(f"\n📈 PERFORMANCE BY LEARNING RATE:")
perf_by_lr = df.groupby(['ssl_method', 'lr'])['accuracy'].agg(['mean', 'max', 'count']).round(2)
print(perf_by_lr)