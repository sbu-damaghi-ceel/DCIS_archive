import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import openslide
import cv2
from shapely.geometry import Polygon, MultiPolygon
import os
join = os.path.join

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scipy.stats import mannwhitneyu

import networkx as nx
import matplotlib.cm as cm

def plot_single(level,stain_colors, stain_classes,patient_df,patient_num,slide_path,
duct_slide=None,hulls_slide=None,save_path=None,highlight_hulls_slide=None):
    
    with openslide.OpenSlide(slide_path) as slide:
        downsampling_factor = slide.level_downsamples[level]
        print(f'downsampling factor at level {level} is {downsampling_factor}')
        
        width,height = slide.level_dimensions[level]
        #center_region = (width // 5, height // 4, 3*width // 5, height // 2)
        center_region = (0,0, width, height)#for excision
        topLeft_point = center_region[:2]
        topLeft_point0 = (int(center_region[0]*downsampling_factor), int(center_region[1]*downsampling_factor))#in level 0
        base_slide = slide.read_region(topLeft_point0, level, (center_region[2], center_region[3]))
        np_slide = np.array(base_slide)[:, :, :3]  # Convert to RGB
    
    
    
    # Plot the slide
    plt.figure(figsize=(20, 15))
    plt.imshow(np_slide)
    

    # Overlay the points
    for stain, color in stain_colors.items():
        valid_classes = stain_classes[stain]
        stain_group = patient_df[(patient_df['stain'] == stain)&((patient_df['Class'].isin(valid_classes)))]
        if not stain_group.empty:
            xy = (stain_group[['X_HE', 'Y_HE']] / downsampling_factor-np.array(topLeft_point)).to_numpy()
            plt.scatter(xy[:, 0], xy[:, 1], color=color, alpha=0.5, label=stain,s=.005)#,marker='.'

    #plot mannual annotation
    if duct_slide is not None:
        
        for duct in duct_slide.values():
            if isinstance(duct,Polygon):
                exterior_coords = np.array(duct.exterior.coords)
                exterior_coords_lv = exterior_coords / downsampling_factor - np.array(topLeft_point)
                plt.plot(exterior_coords_lv[:, 0], exterior_coords_lv[:, 1], color='red', linewidth=1)
            elif isinstance(duct,MultiPolygon):
                for poly in duct.geoms:
                    exterior_coords = np.array(poly.exterior.coords)
                    exterior_coords_lv = exterior_coords / downsampling_factor- np.array(topLeft_point)
                    plt.plot(exterior_coords_lv[:, 0], exterior_coords_lv[:, 1], color='red', linewidth=1)
    
    

    if hulls_slide is not None:
        
        for ductId_IHC,niches_dict in hulls_slide.items():
            for niche_id, region in niches_dict.items():
                if highlight_hulls_slide is not None and niche_id in highlight_hulls_slide:
                    hull_form = 'y-'
                else:
                    hull_form = 'k-'
                if isinstance(region, Polygon):
                    exterior_points = np.array(region.exterior.coords.xy).T
                    converted_points = np.array(exterior_points)  / downsampling_factor - topLeft_point
                    plt.plot(converted_points[:, 0], converted_points[:, 1], hull_form, linewidth=0.5)
                elif isinstance(region, MultiPolygon):
                    for polygon in region.geoms:
                        exterior_points = np.array(polygon.exterior.coords.xy).T
                        converted_points = np.array(exterior_points)  / downsampling_factor - topLeft_point
                        plt.plot(converted_points[:, 0], converted_points[:, 1], hull_form, linewidth=0.5)

    lgnd =plt.legend()
    for handle in lgnd.legend_handles:
        handle.set_sizes([10])


    plt.tight_layout() 
    plt.axis('off')

    if save_path != None:
        plt.savefig(save_path,bbox_inches='tight', pad_inches=0,dpi=300)
    else:
        plt.show()
    plt.close()

def plot_pca(df, excluded_columns,label_col,discrete=True,output=False,dotsize=5):
    features = df.drop(columns=excluded_columns)
    # Clip extreme values to handle numerical stability issues
    features = np.clip(features, a_min=-1e10, a_max=1e10)
    labels = df[label_col]
    upstage_labels = df['Upstage']

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    fig, ax = plt.subplots(figsize=(12, 8))

    #### PCA
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(scaled_features)
    pca_df = pd.DataFrame(pca_results, columns=['PCA1', 'PCA2'])
    pca_df[label_col] = labels.values
    pca_df['Upstage'] = upstage_labels.values

    if not discrete:
        # If the label column is continuous
        scatter = ax.scatter(pca_df['PCA1'], pca_df['PCA2'], c=pca_df[label_col], cmap='coolwarm', alpha=0.7,s=dotsize)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(label_col)
    else:
        # If the label column is categorical
        sns.scatterplot(
            x='PCA1', y='PCA2', hue=label_col, style='Upstage',
            markers={0: 'o', 1: '^'}, palette='tab10', data=pca_df, legend='full', ax=ax,s=dotsize
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # sns.scatterplot(
    #     x='PCA1', y='PCA2', hue=label_col, style='Upstage',
    #     markers={0: 'o', 1: '^'}, palette='tab10', data=pca_df, legend='full', ax=ax
    # )
    ax.set_title(f'PCA Plot Colored by {label_col}')
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
def plot_tsne(df, excluded_columns,label_col,discrete=True,output=False,dotsize=5):
    features = df.drop(columns=excluded_columns)
    # Clip extreme values to handle numerical stability issues
    features = np.clip(features, a_min=-1e10, a_max=1e10)
    labels = df[label_col]
    upstage_labels = df['Upstage']

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    fig, ax = plt.subplots(figsize=(12, 8))
    #### TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=100)
    tsne_results = tsne.fit_transform(scaled_features)
    tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    tsne_df[label_col] = labels.values
    tsne_df['Upstage'] = upstage_labels.values

    if not discrete:
        # If the label column is continuous
        scatter = ax.scatter(tsne_df['TSNE1'], tsne_df['TSNE2'], c=tsne_df[label_col], cmap='coolwarm', alpha=0.7,s=dotsize)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(label_col)
    else:
        # If the label column is categorical
        sns.scatterplot(
            x='TSNE1', y='TSNE2', hue=label_col, style='Upstage',
            markers={0: 'o', 1: '^'}, palette='tab10', data=tsne_df, legend='full', ax=ax,s=dotsize
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # sns.scatterplot(
    #     x='TSNE1', y='TSNE2', hue=label_col, style='Upstage',
    #     markers={0: 'o', 1: '^'}, palette='tab10', data=tsne_df, legend='full', ax=ax
    # )
    ax.set_title(f't-SNE Plot Colored by {label_col}')
    ax.set_xlabel('TSNE1')
    ax.set_ylabel('TSNE2')
    #ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    if output:
        return fig
    else:
        plt.show()
        plt.close(fig)
def plot_dotplot_violin(labeled_df,output=False):
    unique_pattern_ids = sorted(labeled_df['patternId'].unique())
    colors = sns.color_palette('tab10', len(unique_pattern_ids))
    
    pattern_counts = labeled_df.groupby(['slideId', 'Upstage'])['patternId'].value_counts().unstack().fillna(0)
    figs=[]
    p_list=[]
    for pattern_id, color in zip(unique_pattern_ids, colors):
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [3, 1]})

        # Dot plot for the specific patternId
        ax = axes[0]
        pattern_count_upstage = pattern_counts.loc[pattern_counts.index.get_level_values('Upstage') == 1, pattern_id].sort_values()
        pattern_count_non_upstage = pattern_counts.loc[pattern_counts.index.get_level_values('Upstage') == 0, pattern_id].sort_values()
        
        combined_counts = pd.concat([pattern_count_upstage, pattern_count_non_upstage])
        
        separator_index = len(pattern_count_upstage)

        for idx, (slideId, count) in enumerate(combined_counts.items()):
            marker = '^' if labeled_df.loc[labeled_df['slideId'] == slideId[0], 'Upstage'].iloc[0] == 1 else 'o'
            ax.plot([idx] * int(count), range(int(count)), marker, color=color, label=f"{slideId[0]}, Upstage={slideId[1]}" if idx == 0 else "")
        
        ax.axvline(x=separator_index - 0.5, color='black', linestyle='--')
        ax.set_title(f'Dot Plot of patternId {pattern_id} Counts by slideId (Grouped by Upstage)')
        ax.set_xlabel('Index')
        ax.set_ylabel('Count')
        ax.set_xticks(range(len(combined_counts)))
        ax.set_xticklabels([slideId[0] for slideId in combined_counts.index], rotation=90)
        ax.xaxis.grid(True)
        #ax.yaxis.grid(True, which='major')
        # Violin plot with Wilcoxon test
        ax = axes[1]
        upstage_counts = [combined_counts[slideId] for slideId in pattern_count_upstage.index]
        non_upstage_counts = [combined_counts[slideId] for slideId in pattern_count_non_upstage.index]

        stat, p_value = mannwhitneyu(upstage_counts, non_upstage_counts, alternative='two-sided')
        p_list.append(p_value)

        sns.violinplot(data=[upstage_counts, non_upstage_counts], inner="point", palette=[color, color], ax=ax)
        ax.set_title(f'Violin Plot\nWilcoxon test p-value: {p_value:.4f}')
        ax.set_xlabel('Upstage')
        ax.set_ylabel('Count')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Upstage=1', 'Upstage=0'])

        plt.tight_layout()
        if output:
            figs.append(fig)
        else:
            plt.show()
            plt.close(fig)

    if output:
        return figs,p_list
def plot_bar_violin(labeled_df, output=False):
    unique_pattern_ids = sorted(labeled_df['patternId'].unique())
    colors = sns.color_palette('tab10', len(unique_pattern_ids))
    
    # Calculate proportions
    total_counts = labeled_df.groupby(['slideId'])['patternId'].count()
    labeled_df['proportion'] = labeled_df.groupby(['slideId', 'patternId'])['patternId'].transform('count') / labeled_df['slideId'].map(total_counts)
    pattern_proportions = labeled_df.groupby(['slideId', 'Upstage', 'patternId'])['proportion'].mean().unstack().fillna(0)
    
    figs = []
    p_list = []
    for pattern_id, color in zip(unique_pattern_ids, colors):
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [3, 1]})

        # Bar plot for the specific patternId
        ax = axes[0]
        pattern_proportion_upstage = pattern_proportions.loc[pattern_proportions.index.get_level_values('Upstage') == 1, pattern_id].sort_values()
        pattern_proportion_non_upstage = pattern_proportions.loc[pattern_proportions.index.get_level_values('Upstage') == 0, pattern_id].sort_values()

        combined_proportions = pd.concat([pattern_proportion_upstage, pattern_proportion_non_upstage])
        
        separator_index = len(pattern_proportion_upstage)

        ax.bar(range(len(combined_proportions)), combined_proportions, color=color)
        
        ax.axvline(x=separator_index - 0.5, color='black', linestyle='--')
        ax.set_title(f'Bar Plot of patternId {pattern_id} Proportions by slideId (Grouped by Upstage)')
        ax.set_xlabel('Index')
        ax.set_ylabel('Proportion')
        ax.set_xticks(range(len(combined_proportions)))
        ax.set_xticklabels([slideId[0] for slideId in combined_proportions.index], rotation=90)
        ax.xaxis.grid(True)

        # Violin plot with Mann-Whitney U test
        ax = axes[1]
        upstage_proportions = pattern_proportion_upstage.tolist()
        non_upstage_proportions = pattern_proportion_non_upstage.tolist()

        stat, p_value = mannwhitneyu(upstage_proportions, non_upstage_proportions, alternative='two-sided')
        p_list.append(p_value)

        sns.violinplot(data=[upstage_proportions, non_upstage_proportions], inner="point", palette=[color, color], ax=ax)
        ax.set_title(f'Violin Plot\nMann-Whitney U test p-value: {p_value:.4f}')
        ax.set_xlabel('Upstage')
        ax.set_ylabel('Proportion')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Upstage=1', 'Upstage=0'])

        plt.tight_layout()
        if output:
            figs.append(fig)
        else:
            plt.show()
            plt.close(fig)

    if output:
        return figs, p_list
def visualize_graph_and_regions(cells, graph, regions):
    plt.figure(figsize=(20, 10))
    
    pos = {i: (cells[i][0], cells[i][1]) for i in range(len(cells))}
    nx.draw(graph, pos, with_labels=False, node_color='blue', edge_color='black', node_size=50, alpha=0.7)
    
    colormap = cm.get_cmap('jet', len(regions))  #
    for i,region in enumerate(regions):
        if isinstance(region, Polygon):
            x, y = region.exterior.xy
            plt.fill(x, y, alpha=0.4, fc=colormap(i), ec='none')
    plt.xlabel('X')
    plt.ylabel('Y')
    #flip the y-axis
    ax = plt.gca()
    ax.invert_yaxis()
    plt.show()