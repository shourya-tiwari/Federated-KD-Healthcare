import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

def plot_hospital_distribution(hospitals, save_dir="experiments"):
    """
    Creates a bar chart showing data distribution across hospitals.
    """
    os.makedirs(save_dir, exist_ok=True)

    hospital_ids  = [f"H{h['hospital_id']}" for h in hospitals]
    normal_counts = [h['num_normal']      for h in hospitals]
    pneumonia_counts = [h['num_pneumonia'] for h in hospitals]
    is_expert     = [h['is_expert']        for h in hospitals]

    x = range(len(hospitals))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_normal    = ax.bar([i - width/2 for i in x], normal_counts,
                            width, label='Normal', color='steelblue')
    bars_pneumonia = ax.bar([i + width/2 for i in x], pneumonia_counts,
                            width, label='Pneumonia', color='tomato')

    # Add expert/non-expert background shading
    for i, expert in enumerate(is_expert):
        color = 'lightgreen' if expert else 'lightyellow'
        ax.axvspan(i - 0.5, i + 0.5, alpha=0.2, color=color, zorder=0)

    # Labels on bars
    for bar in bars_normal + bars_pneumonia:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 5,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Hospital', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_title('Non-IID Data Distribution Across Hospitals\n(Green = Expert, Yellow = Non-Expert)',
                 fontsize=13)
    ax.set_xticks(list(x))
    ax.set_xticklabels(hospital_ids)
    ax.legend()

    expert_patch    = mpatches.Patch(color='lightgreen', alpha=0.5, label='Expert Hospital')
    nonexpert_patch = mpatches.Patch(color='lightyellow', alpha=0.5, label='Non-Expert Hospital')
    ax.legend(handles=[bars_normal, bars_pneumonia, expert_patch, nonexpert_patch],
              labels=['Normal', 'Pneumonia', 'Expert Hospital', 'Non-Expert Hospital'])

    plt.tight_layout()
    save_path = os.path.join(save_dir, "hospital_distribution.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    # Simulate hospital data for standalone testing
    from data.partition import partition_into_hospitals

    DATA_DIR = r"G:\PROJECTS\FL with KD\Kaggle (Pneumonia) Chest X-ray Dataset\chest_xray\train"
    hospitals = partition_into_hospitals(DATA_DIR)
    plot_hospital_distribution(hospitals)