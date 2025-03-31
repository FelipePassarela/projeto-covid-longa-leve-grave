from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder
from umap import UMAP

from utils.models.evaluate_models import extract_subset
from utils.models.model_dumping import load_rfe_selector
from utils.pipelines import data_preparing_pipeline


def load_and_preprocess_data(
        genomic_path: Path,
        selector_path: Path,
        target: str,
        n_feats: int,
        missing_threshold: float = 10.0,
        oversample: bool = False
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads and preprocesses data for clustering and projection.
    
    :param genomic_path: Path to genomic data
    :param selector_path: Path to RFE selector
    :param target: Target column name
    :param n_feats: Number of features to select
    :param missing_threshold: Maximum percentage of missing values allowed
    :param oversample: Whether to perform data oversampling
    
    :return: Tuple with (X dataframe, processed X, y_train, y_test)
    """
    X, X_train, X_test, y_train, y_test = data_preparing_pipeline(
        genomic_path,
        target,
        oversample=oversample,
        missing_threshold=missing_threshold,
    )

    selector = load_rfe_selector(1, selector_path)
    selected_features, feature_indices = extract_subset(selector, X.values, n_feats)
    feature_names = X.columns[feature_indices]
    
    X_selected = pd.DataFrame(selected_features, columns=feature_names)
    X_onehot = OneHotEncoder(sparse_output=False, drop='first').fit_transform(X_selected)
    print(f"Shape of X_onehot: {X_onehot.shape}")
    
    return X_selected, X_onehot, y_train, y_test


def perform_clustering(
        X: np.ndarray,
        metric: str = 'hamming',
        min_cluster_size: int = 4,
        min_samples: Optional[int] = None,
        cluster_selection_method: str = 'eom'
    ) -> np.ndarray:
    """
    Performs clustering using HDBSCAN.
    
    :param X: Data for clustering
    :param metric: Distance metric to be used
    :param min_cluster_size: Minimum cluster size
    :param min_samples: Minimum number of samples
    :param cluster_selection_method: Cluster selection method
    
    :return: Array with cluster labels
    """
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size, 
        min_samples=min_samples, 
        metric=metric,
        cluster_selection_method=cluster_selection_method,
    )
    clusters = clusterer.fit_predict(X)    
    return clusters


def project_data(
        X: np.ndarray,
        method: str = 'tsne',
        metric: str = 'hamming',
        random_state: int = 42,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        perplexity: float = 100
    ) -> Tuple[np.ndarray, str, str]:
    """
    Projects data in 2D using UMAP or t-SNE.
    
    :param X: Data for projection
    :param method: Projection method ('umap' or 'tsne')
    :param metric: Distance metric
    :param random_state: Random seed
    :param n_neighbors: Number of neighbors for UMAP
    :param min_dist: Minimum distance for UMAP
    :param perplexity: Perplexity value for t-SNE
    
    :return: Tuple with (projected data, x axis name, y axis name)
    """
    if method.lower() == 'umap':
        projector = UMAP(
            random_state=random_state,
            metric=metric,
            n_neighbors=n_neighbors,
            min_dist=min_dist
        )
        axis_names = ("UMAP1", "UMAP2")
    else:
        projector = TSNE(
            random_state=random_state,
            metric=metric,
            perplexity=perplexity
        )
        axis_names = ("tSNE1", "tSNE2")
        
    X_projected = projector.fit_transform(X)
    
    return X_projected, axis_names[0], axis_names[1]


def visualize_clusters(
        X_projected: np.ndarray,
        clusters: np.ndarray,
        targets: np.ndarray,
        axis1: str,
        axis2: str,
        title: str,
        save_path: Path = None,
        show_plot: bool = False,
        figsize: Tuple[int, int] = (6, 5)
    ) -> None:
    """
    Visualizes the found clusters.
    
    :param X_projected: Data projected in 2D
    :param clusters: Array with cluster labels
    :param targets: Array with target values
    :param axis1: X axis name
    :param axis2: Y axis name
    :param title: Plot title
    :param save_path: Path to save the plot
    :param show_plot: Whether to display the plot
    :param figsize: Figure size
    """
    results = pd.DataFrame({
        axis1: X_projected[:, 0],
        axis2: X_projected[:, 1],
        'Cluster': clusters,
        'Target': targets
    })
    
    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        results[axis1], 
        results[axis2], 
        c=results['Cluster'], 
        cmap='viridis', 
        alpha=0.7,
    )

    plt.title(title)
    plt.xlabel(axis1)
    plt.ylabel(axis2)

    # Add legends for clusters
    unique_clusters = results['Cluster'].unique()
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                         markerfacecolor=scatter.cmap(scatter.norm(c)), 
                         markersize=10) for c in unique_clusters]
    cluster_legend = [f'Cluster {c}' if c != -1 else 'Noise' for c in unique_clusters]
    plt.legend(handles, cluster_legend, title='Clusters', loc='best')

    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
        
    plt.close()


def evaluate_variants_importances(
        X_df: pd.DataFrame, 
        clusters: np.ndarray,
        output_dir: Path,
        max_variants_to_report: int = 25
    ) -> None:
    """
    Evaluates the importance of variants for the found clusters using chi-square test.

    :param X_df: DataFrame containing genomic variants
    :param clusters: Array containing cluster labels
    :param output_dir: Directory to save results
    :param max_variants_to_report: Maximum number of variants to report
    """
    p_values = {}
    for col in X_df:
        contingency_table = pd.crosstab(X_df[col], clusters)
        _, p, _, _ = chi2_contingency(contingency_table)
        p_values[col] = p

    sorted_p = sorted(p_values.items(), key=lambda x: x[1])
    with open(output_dir / f"p_values.txt", 'w') as f:
        print("Most important variants for clustering:", file=f)
        for variant, p_val in sorted_p[:max_variants_to_report]:
            print(f"{variant}: p = {p_val:.4f}", file=f)


def evaluate_variants_modes(X_df: pd.DataFrame, clusters: np.ndarray, output_dir: Path) -> None:
    """
    Evaluates the modes of variants in each found cluster.
    
    :param X_df: DataFrame containing genomic variants
    :param clusters: Array containing cluster labels
    :param output_dir: Directory to save results
    """
    X_df['Cluster'] = clusters
    variants_modes = X_df.groupby("Cluster").agg(lambda x: f"{x.mode()[0]} ({(x==x.mode()[0]).mean():.2%})")
    X_df.drop(columns=['Cluster'], inplace=True)

    print(f"Number of clusters found: {len(variants_modes)}")
    print("Variants modes:\n", variants_modes)

    save_path = output_dir / f"variants_modes.csv"
    variants_modes.reset_index(inplace=True)
    variants_modes.rename(columns={'index': 'Cluster'}, inplace=True)
    variants_modes.to_csv(save_path, index=False)


def main(
        selector_path: Path,
        genomic_path: Path,
        target: str,
        n_feats: int = 25,
        projection_method: str = 'tsne',
        metric: str = 'hamming',
        min_cluster_size: int = 4,
        output_dir: Path = Path("../clusters_results"),
        show_plot: bool = False
    ) -> None:
    """
    Main function that executes the clustering and projection pipeline.
    
    :param selector_path: Path to RFE selector
    :param genomic_path: Path to genomic data
    :param target: Target column name
    :param n_feats: Number of features to select
    :param projection_method: Projection method ('umap' or 'tsne')
    :param metric: Distance metric
    :param min_cluster_size: Minimum cluster size
    :param output_dir: Directory to save results
    :param show_plot: Whether to display the plot
    """
    X_df, X_onehot, y_train, y_test = load_and_preprocess_data(
        genomic_path,
        selector_path,
        target,
        n_feats
    )
    
    clusters = perform_clustering(
        X_onehot,
        metric=metric,
        min_cluster_size=min_cluster_size
    )
    X_projected, axis1, axis2 = project_data(
        X_onehot,
        method=projection_method,
        metric=metric
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    evaluate_variants_importances(X_df, clusters, output_dir, n_feats)
    evaluate_variants_modes(X_df, clusters, output_dir)
        
    title_suffix = "(Long COVID)" if target == "Long_COVID" else "(Risk COVID)"
    title = f"Clusters Found {title_suffix}"
    save_path = output_dir / f"clusters.png"
    
    visualize_clusters(
        X_projected,
        clusters,
        np.hstack([y_train, y_test]),
        axis1,
        axis2,
        title,
        save_path=save_path,
        show_plot=show_plot,
    )


if __name__ == "__main__":
    datasets = [
        ("grave", "nao_vacinados", "risk"),
        ("longa", "nao_vacinados", "Long_COVID"),
    ]

    for category, subcategory, target in datasets:
        print(f"Processing dataset: {category}/{subcategory} with target: {target}")
        main(
            selector_path=Path(f"../resultados/30-03-2025/{category}/{subcategory}/models/selectors"),
            genomic_path=Path(f"../data/{category}/{subcategory}/merged.csv"),
            target=target,
            n_feats=25,
            projection_method='tsne',
            metric='hamming',
            min_cluster_size=5,
            show_plot=False,
            output_dir=Path(f"../clusters_results/{category}/{subcategory}")
        )
