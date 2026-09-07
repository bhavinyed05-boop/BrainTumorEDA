import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path

# Bring in all required pipeline workflows from your data script module
from data_pipeline import (
    configure_kaggle_credentials, 
    download_and_extract_mri_dataset, 
    generate_mri_metadata
)

def plot_vertical_dashboard(df_meta, categories):
    """
    Generates a unified visual quality control report stacking each section
    vertically with optimized, tight spacing below the main application title.
    """
    # Force a large, crisp vertical grid canvas size
    fig = plt.figure(figsize=(14, 28))

    # Define vertical grid with increased margin space (hspace) to stop layout clipping
    gs = fig.add_gridspec(4, 1, hspace=0.45)

    # ----------------------------------------------------
    # SECTION 1: Clinical Pathology Sample Matrix (Grid)
    # ----------------------------------------------------
    sub_gs = gs[0].subgridspec(4, 4, hspace=0.3, wspace=0.15)
    ax_title1 = fig.add_subplot(gs[0])
    ax_title1.set_title("SECTION 1: Clinical Pathology Matrix (Random Samples)",
                        fontsize=15, fontweight='bold', pad=15, color='#2c3e50', loc='left')
    ax_title1.axis('off')

    for row_idx, cat in enumerate(categories):
        subset = df_meta[df_meta['Class'] == cat]
        if len(subset) < 4: continue
        sampled_rows = subset.sample(4, random_state=42)

        for col_idx, (_, row) in enumerate(sampled_rows.iterrows()):
            ax = fig.add_subplot(sub_gs[row_idx, col_idx])
            with Image.open(row['Path']) as img:
                img_matrix = np.array(img.convert('L').resize((128, 128))) / 255.0
            ax.imshow(img_matrix, cmap='bone')
            ax.axis('off')
            if col_idx == 0:
                ax.text(-25, 64, cat.upper(), rotation=90, va='center', ha='right',
                        fontsize=11, fontweight='bold', color='darkred')

    # ----------------------------------------------------
    # SECTION 2: Voxel Intensity Distribution Profile
    # ----------------------------------------------------
    ax_intensity = fig.add_subplot(gs[1])
    ax_intensity.set_title("SECTION 2: Grayscale Voxel Intensity Distribution Profile",
                           fontsize=15, fontweight='bold', pad=15, color='#2c3e50', loc='left')

    for cat in categories:
        subset = df_meta[df_meta['Class'] == cat]
        sns.kdeplot(subset['MeanIntensity'], ax=ax_intensity, label=cat.upper(), fill=True, alpha=0.1)

    ax_intensity.set_xlabel("Mean Gray Level Voxel Value (0-255)", fontsize=11)
    ax_intensity.set_ylabel("Signal Density", fontsize=11)
    ax_intensity.legend(fontsize=10, loc='upper right')
    ax_intensity.grid(True, linestyle='--', alpha=0.5)

    # ----------------------------------------------------
    # SECTION 3: Spatial Dimension Mapping
    # ----------------------------------------------------
    ax_spatial = fig.add_subplot(gs[2])
    ax_spatial.set_title("SECTION 3: Spatial Dimension Mapping & Aspect Ratio Clustering",
                         fontsize=15, fontweight='bold', pad=15, color='#2c3e50', loc='left')

    sns.scatterplot(data=df_meta, x='Width', y='Height', hue='Class', alpha=0.6, ax=ax_spatial, palette="Set2", s=60)
    ax_spatial.set_xlabel("Image Width (Pixels)", fontsize=11)
    ax_spatial.set_ylabel("Image Height (Pixels)", fontsize=11)
    ax_spatial.legend(fontsize=10, title="Pathology Class")
    ax_spatial.grid(True, linestyle='--', alpha=0.5)

    # ----------------------------------------------------
    # SECTION 4: Data Balance & Class Distribution
    # ----------------------------------------------------
    ax_bar = fig.add_subplot(gs[3])
    ax_bar.set_title("SECTION 4: Dataset Balance & Categorical Case Volume Summary",
                     fontsize=15, fontweight='bold', pad=15, color='#2c3e50', loc='left')

    sns.countplot(data=df_meta, x='Class', ax=ax_bar, palette="viridis", order=categories)
    ax_bar.set_xlabel("Diagnosed Pathology Category", fontsize=11)
    ax_bar.set_ylabel("Total Cataloged Case Count", fontsize=11)
    
    ax_bar.set_xticks(range(len(categories)))
    ax_bar.set_xticklabels([cat.upper() for cat in categories], fontsize=10)

    for p in ax_bar.patches:
        ax_bar.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontweight='bold')

    ax_bar.grid(axis='y', linestyle='--', alpha=0.5)

    # CRITICAL FIX: Add overall title via subplots_adjust so it cannot cross tracking boundaries
    fig.subplots_adjust(top=0.94)
    fig.suptitle("Advanced Clinical MRI Analytics Dashboard", fontsize=22, fontweight='bold', x=0.5)

    # Save to your local folder base BEFORE calling plt.show() so canvas constraints hold true
    plt.savefig('clinical_mri_dashboard.png', dpi=300, bbox_inches='tight')
    print("Dashboard image rendered successfully and saved to: clinical_mri_dashboard.png")
    plt.show()