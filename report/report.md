# Report

## Part 1: Core Pipeline Implementation 

### Introduction
The objective of Part 1 was to design and implement a scalable pipeline for unsupervised cellular phenotyping in Whole Slide Images (WSIs). The pipeline integrates cell detection, embedding extraction, unsupervised clustering, and standardized GeoJSON export. I based my implementation on the CellViT-Hibou-L model, which combines instance segmentation with transformer-based feature extraction, making it well-suited for fine-grained cellular analysis.

Note that I have not used the CellViT and Hibou-L models separately. The motivation behind this choice relies on two main factors:
- Using only one model allows me to perform inference only once to extract everything needed for my analysis.
- Simplified pipeline, without the need to combine results coming from two different inferences and relying on a model built on a foundational model, which has proven to be better than the CellViT model.

### Pipeline Overview 
The repository is structured into modular components, where the main scripts for this first pipeline are:

- **src/data/wsi_reader.py**: Efficiently handles large WSIs using OpenSlide and provides multi-resolution access.
- **src/data/tiling.py**: Splits WSIs into manageable 224×224 tiles, with optional padding for edge coverage.
- **src/model/cellvit_hibou_wrapper.py**: A thin wrapper around the CellViT-Hibou model, exposing a clean API for segmentation and token retrieval.
- **src/embedding/embedding_extraction.py**: Implements the strategy for deriving single-cell embeddings from overlapping tokens.
- **src/clustering/clustering.py**: Clusters embeddings into six groups (1–6) using unsupervised methods (e.g., KMeans).
- **src/utils/geojson_export.py**: Outputs final results as a GeoJSON file conforming to the provided schema.

### Embedding Extraction Strategy 

#### 1. Method Description 
A key challenge arises from the fact that Vision Transformers (such as Hibou-L) process images using a regular token grid (e.g., $16 \times 16$ tokens for a $256 \times 256$ tile), whereas cell nuclei are represented as irregular polygons that frequently span multiple tokens. A naïve approach, such as using the [CLS] embedding token from a tile centered on a cell, is suboptimal because this token captures global tile information, leading to contamination from adjacent cells and background tissue.

Several alternative strategies were considered:
* Weighted averaging of token embeddings, with weights determined by the fraction of each token occupied by the cell.
* Upsampling token embeddings to the patch size, applying the cell mask, and computing a weighted average of the masked embeddings.
* Modeling a linear relationship between token embeddings and cell nuclei embeddings, and solving a least squares problem.

The pipeline adopts the third approach, implementing a least-squares unmixing method as follows:

##### **Fractional overlaps**  
For each cell, compute the fraction of each token covered by its mask. This yields weights $f_{t,c} \in [0,1]$, forming a matrix $W \in \mathbb{R}^{T \times C}$, where $T$ = the number of tokens and $C$ = the number of cells in the tile.

##### **Background column**  
An additional column is introduced to account for background coverage:

$$
f_{t,\text{bg}} = \max\left(0,\, 1 - \sum_{c} f_{t,c}\right)
$$

This ensures that any residual tissue signal not assigned to a cell is absorbed into a dedicated background component, preventing contamination of nuclei embeddings.

##### **Linear system**  
Let $E \in \mathbb{R}^{T \times D}$ be the matrix of token embeddings ($D$ = embedding dimension). We solve:

$$
E \approx W Z
$$

where $Z \in \mathbb{R}^{(C+1) \times D}$ contains the per-cell (and background) embeddings.

##### **Ridge regression closed-form**  
To stabilize the solution, especially when cells overlap heavily, we use ridge regression:

$$
Z = (W^\top W + \lambda I)^{-1} W^\top E
$$

with a small regularization parameter $\lambda > 0$ (e.g., $1 \times 10^{-2}$). Each row of $Z$ is the refined embedding for one cell (last row = background).

##### **Output**  
The final per-cell embeddings are the rows of $Z$ corresponding to cells. Each embedding is a 1024-dimensional vector aligned with the model’s token space.

#### 2. Why This Is Superior To Simpler Approaches



- **Unmixing vs. contamination**: The [CLS] token from a tile centered on a cell captures global tile information, making it incapable of isolating features specific to individual cells and background. The least-squares approach, under the linear decomposition assumption, explicitly separates these components, thereby reducing cross-cell contamination and mitigating bias from tissue or empty space through the inclusion of a background column in the embedding matrix.

- **Inference efficiency**: Extracting the [CLS] token for each cell would require performing inference for every cell within a tile, which is computationally prohibitive for WSIs containing large numbers of cells.


#### 3. Trade-offs and Potential Failure Points


- **Partial cell coverage**: The current approach does not account for cells that are partially cut by tile boundaries, which may affect embedding accuracy for these cells.

- **Linear assumption**: The methodology assumes a linear relationship between token, cell, and background embeddings. In practice, this relationship may be more complex and not fully captured by the linear model.

- **Matrix conditioning**: In cases of excessive cell overlap or noisy segmentation, the overlap matrix $W$ may become ill-conditioned due to highly correlated or nearly linearly dependent columns. Ridge regularization addresses this issue, but it can still be a problem.


### Clustering Strategy 


> **⚠️ Important Note: Filtering the Background Class**
>
> Among the supervised classes, one is labeled as **"Background"**. This class is typically used for negative samples in the instance segmentation map and does **not** correspond to a true cellular phenotype. To ensure that the clustering analysis focuses exclusively on biologically meaningful cell types, I explicitly **filtered out all background-class instances** during embedding extraction and clustering. As a result, the analysis and all reported metrics consider **only the five true cell classes** (excluding background).


#### 1. Method Description 

After obtaining refined 1024-dimensional embeddings for each nucleus, the next step is to group cells into phenotypic clusters. The clustering process is designed to produce five unsupervised clusters, as follows:

##### **Embedding aggregation**
Per-cell embeddings and associated metadata from all tiles are concatenated into a unified table.

##### **Preprocessing**
- **Standardization (z-score):** Each embedding dimension is centered and scaled to unit variance, preventing dimensions with larger numeric ranges from disproportionately influencing the clustering.
- **Dimensionality reduction (PCA):** Embeddings are projected onto the top ~200 principal components. PCA serves to reduce noise and redundancy while preserving most of the variance, thereby enhancing clustering efficiency and stability.

##### **Clustering algorithm**
- **Gaussian Mixture Model (GMM)** with $K=5$ has been used for the clustering.
- Each cell embedding is assigned to one of five centroids, resulting in cluster IDs ranging from 1 to 5.
- Cluster labels are reordered such that the largest cluster is designated as Cluster 1, the second largest as Cluster 2, and so forth, ensuring consistent cluster ordering.

#### 2.Potential Failure Points
- **Curse of dimensionality**: Clustering in high-dimensional spaces is challenging due to the curse of dimensionality. While PCA mitigates this issue by reducing dimensionality, the selected latent space may still be too high for optimal clustering performance.
- **Strong Imbalance Dataset**: Standard clustering algorithms, such as K-Means or GMM, are prone to placing more centroids in or near large clusters, leading to bias against smaller, minority clusters. In our scenario, class 0 (background) and 4 (dead cells) are a minority compared to other huge clusters, therefore the clustering algorithms could be tempted to learn other features than the one that are predicted by the model.
---

## Part 2: Creative Evaluation & Cluster Analysis

For the following discussion and results, please keep in mind this mapping of the supervised classes:

* **1 → Neoplastic Cell**
* **2 → Inflammatory Cell**
* **3 → Connective Cell**
* **4 → Dead Cell**
* **5 → Epithelial Cell**

To quantitatively evaluate the quality of the unsupervised clustering results against the supervised labels provided by the Hibou–CellViT model, we computed several complementary clustering evaluation metrics. Each metric captures different aspects of the agreement between clusters and true biological cell types, providing a multi-faceted assessment of performance.

#### Adjusted Rand Index (ARI)

The **Rand Index (RI)** measures similarity between two partitions by counting pairs of points that are consistently grouped or separated. However, RI does not correct for chance. The **Adjusted Rand Index** adjusts this baseline:

$$
ARI = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]}
$$

* Values range from **-1 (worse than random)** to **1 (perfect match)**, with **0 indicating random clustering**.
* ARI is sensitive to both cluster purity and how well all points are assigned.

#### Adjusted Mutual Information (AMI)

**Mutual Information (MI)** quantifies how much knowing one partition reduces uncertainty about the other. The **Adjusted MI** corrects for chance:

$$
AMI = \frac{MI(U, V) - \mathbb{E}[MI(U, V)]}{\max(H(U), H(V)) - \mathbb{E}[MI(U, V)]}
$$

Where:

* $H(\cdot)$ is entropy of the partition.
* $U$ are the predicted cluster labels, $V$ the true labels.

AMI ranges from **0 (no mutual information, random)** to **1 (perfect agreement)**.

#### Normalized Mutual Information (NMI)

**NMI** is a normalized version of MI that scales between 0 and 1:

$$
NMI = \frac{2 \cdot MI(U, V)}{H(U) + H(V)}
$$

* High NMI means clusters and labels share significant information.
* Unlike AMI, NMI does not adjust for chance, so it may overestimate performance with many small clusters.

#### Homogeneity

A cluster assignment is **homogeneous** if each cluster contains only members of a single ground-truth class:

$$
\text{Homogeneity} = 1 - \frac{H(C|K)}{H(C)}
$$

Where:

* $H(C|K)$ is conditional entropy of the classes given clusters.
* Values range from **0 (worst)** to **1 (perfect homogeneity)**.

#### Completeness

A clustering assignment is **complete** if all members of a given class are assigned to the same cluster:

$$
\text{Completeness} = 1 - \frac{H(K|C)}{H(K)}
$$

* Completeness complements homogeneity.
* High values indicate that supervised classes are not split across multiple clusters.

#### V-measure

The **V-measure** is the harmonic mean of homogeneity and completeness:

$$
V = 2 \cdot \frac{\text{Homogeneity} \cdot \text{Completeness}}{\text{Homogeneity} + \text{Completeness}}
$$

* Provides a balanced score combining both aspects.
* Useful to summarize clustering quality in a single number.

#### Fowlkes–Mallows Index (FMI)

The **FMI** compares clustering against ground truth using precision and recall defined over pairs of points:

$$
FMI = \sqrt{\frac{TP}{TP+FP} \cdot \frac{TP}{TP+FN}}
$$

Where:

* **TP** = pairs in the same cluster and same class

* **FP** = pairs in the same cluster but different classes

* **FN** = pairs in different clusters but same class

* Values range from **0 to 1**, with higher meaning better cluster–class agreement.




### 1. Quantitative Concordance Analysis


The confusion matrix comparing the five unsupervised clusters to the five supervised classes predicted by the model is:

| cluster_id |   1 |   2 |   3 |  4 |   5 |
|------------|----:|----:|----:|---:|----:|
| 1          | 336 | 364 | 3091|  1 | 140 |
| 2          | 882 |  22 |  113|  1 |1666 |
| 3          | 637 | 135 |  526|  0 | 562 |
| 4          | 254 | 172 |  794|  3 | 243 |
| 5          |  55 | 172 |  749| 14 |  24 |

- **Adjusted Rand Index (ARI)**: 0.239 (close to 0, indicating weak-to-moderate agreement between unsupervised clusters and supervised classes).
- **Normalized Mutual Information (NMI)**: 0.221 (low, showing limited shared information).
- **Homogeneity**: 0.246 (clusters are not pure; each contains a mix of classes).
- **Completeness**: 0.200 (classes are split across multiple clusters).

These metrics indicate that unsupervised clustering does not strongly align with supervised labels. Clusters are mixed and classes are fragmented, reflecting the challenges of unsupervised phenotyping in complex tissue images.


In addition to global metrics, per-cluster summary statistics were computed to provide further insight into cluster composition and purity. For each cluster, the following statistics are reported:

- **size**                : total number of cells in the cluster
- **top_supervised**      : supervised class with maximum count in the cluster
- **top_count**           : count of that majority class
- **purity**              : top_count / size ∈ [0, 1]
- **entropy**             : Shannon entropy (nats)
- **normalized_entropy**  : entropy / log(#classes) ∈ [0, 1]

The table below summarizes these statistics for each cluster:

| cluster_id | size | top_supervised | top_count | purity | entropy | normalized_entropy |
|------------|------|---------------|-----------|--------|---------|--------------------|
| 1          | 3932 | 3             | 3091      | 0.79   | 0.74    | 0.46               |
| 2          | 2684 | 5             | 1666      | 0.62   | 0.84    | 0.52               |
| 3          | 1860 | 1             | 637       | 0.34   | 1.28    | 0.79               |
| 4          | 1466 | 3             | 794       | 0.54   | 1.20    | 0.74               |
| 5          | 1014 | 3             | 749       | 0.74   | 0.83    | 0.52               |


From this table, it is evident that the first cluster is the purest, consisting of approximately 78% connective cells. This suggests that, despite overall weak concordance, certain clusters may still capture meaningful biological structure.



### 2. Visual & Spatial Analysis

<div style="display: flex; gap: 16px; align-items: flex-start;">
    <div style="flex: 1; text-align: center;">
        <img src="results/roi_visualization/roi_clusters_on_wsi.png" alt="Clusters visualized on WSI" style="max-width: 100%; height: auto;">
        <br>
        <span><strong>Clusters visualized on WSI</strong></span>
    </div>
    <div style="flex: 1; text-align: center;">
        <img src="results/roi_visualization/roi_supervised_on_wsi.png" alt="Supervised classes visualized on WSI" style="max-width: 100%; height: auto;">
        <br>
        <span><strong>Supervised classes visualized on WSI</strong></span>
    </div>
</div>



When comparing the supervised and unsupervised visualizations, an interesting coherence emerges. The unsupervised method consistently groups **neoplastic** and **epithelial** cells into two main clusters (clusters 1 and 2). Both of these clusters contain a balanced mixture of epithelial and neoplastic cells, suggesting that the unsupervised algorithm is primarily recognizing the shared epithelial origin of these two classes.

This makes biological sense: **neoplastic cells arise from epithelial cells**, and despite their abnormal morphology, they still retain many epithelial features. While the supervised classifier distinguishes them into separate categories (“normal epithelium” vs. “tumor epithelium”), the unsupervised clustering appears to capture a **broader, less granular grouping** that reflects their fundamental similarity. In this sense, the unsupervised clusters provide a coarser but biologically coherent representation: epithelial cells are identified as a single family, whether they are normal or tumor-derived.


### 3. Probing the Meaning of Disagreements

#### **Issue A**

From my results, we can see that cluster 3, with a total size of 1860, is mixed between neoplastic cells (637, 34%), ephitelial cells (562, 30%) and connective cells (526, 28%). This is quite an heterogeneous cluster, confirmed also by an overall purity score of 0.34 and a normalized entropy of 0.79.

From a biological point of view, this cluster could somehow catch the tumor-stroma interface zones, where neoplastic, connective tissue and epithelial tissue intermingle. This is because cancer cells don't live in isolation, but they could grow from a microenvironment that includes:
- **Connective tissue (stroma)**: fibroblasts around the tumor.
- **Normal epithelium**: the healthy cells that tumors originally come from.
- **Tumor epithelium**: the tumor cells that can original from epithelial ones.

Therefore, at the tumor boundary, these different cell types are physically mixed together. A patch in that region may contain cells that look similar or overlappping. An unsupervised algorithm might cluster them together, because it is picking up signals from cells that are adjacent or interacting biologically, not just isolated.


From a technical point of view, we need to remind that vision transformers work on image patches. If a patch contains part of tumor cell plus part of a connective or epithelial neighbor, the embedding may become a blend between different embeddings. That can lead to artificial mixing that the proposed pipeline was not able to separate correctly.


#### **Issue B**

Neoplastic cells (supervised type 1) are not concentrated in one cluster, but they are spread across clusters 1 (15%), 2 (40%), 3 (29%) and 4 (12%). From a biological point of view, this could reflect **intra-tumoral heterogeneity**. Tumor cells, although all derived from epithelium, do not form a uniform population. Instead, they vary in morphology, proliferative activity, and interaction with the surrounding stroma and immune cells.
For example, neoplastic cells located in the tumor core may appear more compact and differentiated, whereas those at the invasive front often show irregular shapes and transitional features as they interact with connective tissue and inflammatory infiltrates. Unsupervised clustering may therefore be capturing distinct subpopulations of tumor cells, each representing a different biological state or microenvironmental niche. This heterogeneity is compressed into a single “Neoplastic” class by the supervised model, but emerges more clearly in the unsupervised grouping.

Although the supervised classifier groups all of them as "Neoplastic", the unsupervised clustering is probably splitting them into meaningful sub-groups that correspond to different microenvironments.

From a technical point of view, neoplastic and epithelial cells are morphologically very similar, since tumors originate from epithelium. Patch-based embeddings may therefore fail to capture the subtle differences between normal and tumor epithelium, causing part of the neoplastic population to group with epithelial-like features. Second, variability in staining intensity can introduce artificial variation in embeddings, leading the algorithm to separate neoplastic cells into different clusters. Finally, cell boundary overlap in crowded tumor regions may produce mixed embeddings containing both neoplastic and stromal features, further contributing to cluster splitting.

#### Final Conclusion 

Based on my analysis, I believe the **pre-defined supervised labels** are the more immediately insightful representation for a pathologist. These labels (Neoplastic, Epithelial, Connective, Inflammatory, Dead) map directly to the categories that pathologists already use in their daily workflow. They are interpretable, stable across slides, and actionable for diagnostic purposes, such as estimating tumor burden, identifying immune infiltration, or quantifying necrosis.

By contrast, my unsupervised clusters showed **low quantitative concordance** with supervised labels (ARI ≈ 0.239, AMI ≈ 0.221, optimal mapping accuracy ≈ 0.50), meaning they are not reliable enough to stand alone for clinical decision-making. Cluster assignments varied in purity, and the same cluster ID could correspond to different biological contexts in different slides, which reduces their utility for direct interpretation.

However, the unsupervised representation adds exploratory insight that supervised labels flatten. In particular:

- **Issue A**: A highly mixed cluster revealed the tumor–stroma interface, where neoplastic, epithelial, and connective cells interact. This region is biologically crucial for invasion and may include cancer-associated fibroblasts (CAFs).

**Issue B**: The supervised class “Neoplastic” was split across multiple clusters, uncovering intra-tumoral heterogeneity. 

These findings suggest that while supervised labels remain the **primary, pathologist-facing representation**, unsupervised clustering can serve as a complementary overlay, flagging regions of biological complexity and generating hypotheses for further investigation. A side-by-side visualization of supervised and unsupervised maps provides the most comprehensive view: one grounded in diagnostic categories, the other highlighting tissue substructure and microenvironmental diversity.

