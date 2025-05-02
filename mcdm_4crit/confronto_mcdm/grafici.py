
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection

def plot_heatmap_corr(results, save_path='heatmap_rankings_correlation.png'):
    corr_matrix = results[['Rank_TOPSIS', 'Rank_VIKOR', 'Rank_PROMETHEE']].corr(method='spearman')
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", vmin=0.65, vmax=1)
    plt.title('Correlazione di Spearman tra Rankings')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_scatter_comparison(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.scatterplot(data=results, x='Score_TOPSIS', y='Score_VIKOR', ax=axes[0])
    axes[0].set(title='TOPSIS vs VIKOR', xlabel='TOPSIS (High)', ylabel='VIKOR (Low)')

    sns.scatterplot(data=results, x='Score_TOPSIS', y='Score_PROMETHEE', ax=axes[1])
    axes[1].set(title='TOPSIS vs PROMETHEE', xlabel='TOPSIS (High)', ylabel='PROMETHEE (High)')

    sns.scatterplot(data=results, x='Score_VIKOR', y='Score_PROMETHEE', ax=axes[2])
    axes[2].set(title='VIKOR vs PROMETHEE', xlabel='VIKOR (Low)', ylabel='PROMETHEE (High)')

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('scatter_comparison_scores.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_sensitivity_barplot(sensitivity_df):
    plt.figure(figsize=(10, 6))
    sensitivity_plot_data = sensitivity_df.melt(
        id_vars=['Scenario'],
        value_vars=['TOPSIS_Stability', 'VIKOR_Stability', 'PROMETHEE_Stability'],
        var_name='Method', value_name='Stability'
    )

    sns.barplot(x='Scenario', y='Stability', hue='Method', data=sensitivity_plot_data)
    plt.axhline(y=0.7, color='r', linestyle='--', alpha=0.7, label='Soglia di stabilità (0.7)')
    plt.title('Stabilità dei metodi alla variazione dei pesi', fontsize=14)
    plt.xlabel('Scenario', fontsize=12)
    plt.ylabel('Correlazione di Spearman con ranking originale', fontsize=12)
    plt.ylim(0, 1)
    plt.legend(title='Metodo')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_execution_times(summary_table):
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        x="Tempo Medio (ms)",
        y="Metodo",
        hue="Metodo",
        data=summary_table,
        palette="Blues_d",
        dodge=False,
        legend=False
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", padding=3, fontsize=9)
    plt.title('Tempo Medio di Esecuzione per Metodo MCDM', fontsize=14)
    plt.xlabel('Tempo Medio (millisecondi)', fontsize=12)
    plt.ylabel('Metodo', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig('execution_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_top10_radar(means_df):
    def radar_factory(num_vars, frame='circle'):
        theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
        class RadarAxes(PolarAxes):
            name = 'radar'
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.set_theta_zero_location('N')
            def fill(self, *args, **kwargs):
                return super().fill(*args, **kwargs)
            def plot(self, *args, **kwargs):
                return super().plot(*args, **kwargs)
        register_projection(RadarAxes)
        return theta

    means_norm = means_df.copy()
    for col in means_norm.columns:
        if 'Volatility' in means_norm.index and col == 'Volatility':
            means_norm.loc['Volatility', col] = 1 / means_norm.loc['Volatility', col]
    for idx in means_norm.index:
        min_val = means_norm.loc[idx].min()
        max_val = means_norm.loc[idx].max()
        if max_val > min_val:
            means_norm.loc[idx] = (means_norm.loc[idx] - min_val) / (max_val - min_val)

    theta = radar_factory(len(means_norm.index))
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_xticks(theta)
    ax.set_xticklabels(means_norm.index)

    colors = ['b', 'g', 'r']
    for i, (method, values) in enumerate(means_norm.items()):
        values_list = values.values.flatten().tolist()
        values_list += values_list[:1]
        ax.plot(np.append(theta, theta[0]), values_list, color=colors[i], label=method)
        ax.fill(np.append(theta, theta[0]), values_list, color=colors[i], alpha=0.1)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.title('Confronto delle caratteristiche medie nelle Top 10 alternative', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('top10_characteristics_radar.png', dpi=300, bbox_inches='tight')
    plt.show()
