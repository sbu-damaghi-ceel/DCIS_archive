import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scipy.stats import mannwhitneyu

from matplotlib.backends.backend_pdf import PdfPages
import json

import os
join = os.path.join
import multiprocessing as mp
def kmeans_clustering(df, exclude_column, num_clusters):
    features = df.drop(columns=exclude_column)
    # Clip extreme values to handle numerical stability issues
    features = np.clip(features, a_min=-1e10, a_max=1e10)
    

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42,n_init=10)
    clusters = kmeans.fit_predict(scaled_features)
    
    # Add cluster assignments to a new column
    df['patternId'] = clusters
    
    return df

def plot_pca_tsne(df, excluded_columns,output=False):
    features = df.drop(columns=excluded_columns)
    # Clip extreme values to handle numerical stability issues
    features = np.clip(features, a_min=-1e10, a_max=1e10)
    labels = df['patternId']
    upstage_labels = df['Upstage']

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    fig, axs = plt.subplots(2, 1, figsize=(12, 16))

    #### PCA
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(scaled_features)
    pca_df = pd.DataFrame(pca_results, columns=['PCA1', 'PCA2'])
    pca_df['patternId'] = labels.values
    pca_df['Upstage'] = upstage_labels.values

    ax = axs[0]
    sns.scatterplot(
        x='PCA1', y='PCA2', hue='patternId', style='Upstage',
        markers={0: 'o', 1: '^'}, palette='tab10', data=pca_df, legend='full', ax=ax
    )
    ax.set_title('PCA Plot Colored by patternId')
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    #### TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=100)
    tsne_results = tsne.fit_transform(scaled_features)
    tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    tsne_df['patternId'] = labels.values
    tsne_df['Upstage'] = upstage_labels.values

    ax = axs[1]
    sns.scatterplot(
        x='TSNE1', y='TSNE2', hue='patternId', style='Upstage',
        markers={0: 'o', 1: '^'}, palette='tab10', data=tsne_df, legend='full', ax=ax
    )
    ax.set_title('t-SNE Plot Colored by patternId')
    ax.set_xlabel('TSNE1')
    ax.set_ylabel('TSNE2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

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

def plot_save_main(labeled_df,out_dir,naming_column):
    all_p_lists = {}
    for num_clusters in range(3,11):
        df = kmeans_clustering(labeled_df, naming_column+['Upstage'], num_clusters)
        df.to_csv(join(out_dir,f'cluster{num_clusters}.csv'),index=False)
        
        pattern_counts = df.groupby('slideId')['patternId'].value_counts().unstack().fillna(0)

        pca_fig = plot_pca_tsne(df, naming_column+['patternId','Upstage'],output=True)
        dot_figs,p_list = plot_dotplot_violin(labeled_df,output=True)

        pdf_path = join(out_dir,f'cluster{num_clusters}.pdf')
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(pca_fig)
            plt.close(pca_fig)
            for fig in dot_figs:
                pdf.savefig(fig)
                plt.close(fig)
        all_p_lists[f'cluster{num_clusters}'] = p_list

    json_path = join(out_dir, 'p_values.json')
    with open(json_path, 'w') as json_file:
        json.dump(all_p_lists, json_file)
    print(f'{out_dir} done.')
def main():
    n_process = 10

    id_df = pd.read_csv('/mnt/data10/shared/yujie/new_DCIS/patientNum_slideId_upstage_biopsy.csv')


    roi = 'Niche'
    naming_column = ['slideId','ductId_IHC','clusterId']

    tasks = []
    for stain in ['CA9','LAMP2b']:
        for thres in range(10,45,5):
            parent_dir = f'/mnt/data10/shared/yujie/new_DCIS/Features/{roi}/{stain}_{thres}/'
            for na_type in ['imputed','noNAcol']:
                if (stain,thres,na_type)   == ('LAMP2b',40,'imputed'): continue
                feat_path = join(parent_dir,f'merged_feat_{na_type}.csv')
                feat_df = pd.read_csv(feat_path)
                labeled_df = pd.merge(feat_df, id_df[['Upstage', 'slideId']], on='slideId')

                out_dir = join(parent_dir,na_type)
                os.makedirs(out_dir,exist_ok=True)
                tasks.append((labeled_df, out_dir,naming_column))
    with mp.Pool(n_process) as pool:
        pool.starmap(plot_save_main, tasks)

if __name__=='__main__':
    main()

        