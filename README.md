
# Unsupervised Cellular Phenotyping

## 📌 Overview

Unsupervised Cellular Phenotyping is a scalable pipeline for analyzing Whole Slide Images (WSIs) to identify and cluster single-cell phenotypes. The pipeline enables:

- **Cell segmentation and embedding extraction** using the CellViT-Hibou-L model
- **Unsupervised clustering** of cellular embeddings
- **GeoJSON export** of results for interoperability
- **Quantitative and visual evaluation** of discovered phenotypes

---

## 🚀 Installation

### 1. Clone the Repository (with Submodules)

```bash
git clone --recurse-submodules https://github.com/paolomotta/unsupervised-cellular-phenotyping.git
cd unsupervised-cellular-phenotyping
```

### 2. Download Data and Model Weights

Download the reference Whole Slide Image from the GDC Cancer Portal and place the `.svs` file it in the `data/` directory:

- **WSI File:** [TCGA-V5-A7RE-11A-01-TS1.57401526-EF9E-49AC-8FF6-B4F9652311CE.svs](https://portal.gdc.cancer.gov/files/f9147f06-2902-4a64-b293-5dbf9217c668)


### 3. Create HuggingFace Access Token

To download the required model weights, you need a HuggingFace access token:

1. Visit the [HuggingFace token page](https://huggingface.co/docs/hub/security-tokens) and create an access token if you don’t already have one.
2. Copy your token into the `env.example` file, then rename it to `.env`.
3. If you encounter issues downloading the weights, refer to the [model documentation](https://huggingface.co/histai/cellvit-hibou-l) for manual download instructions.
4. If you download the weights [manually](https://huggingface.co/histai/cellvit-hibou-l), place them in the `data/` folder.

> **Note:** Authentication (via access token or SSH) is always required to access the model weights.

### 4. Build the Docker Image

```bash
docker build -t cell-phenotyping .
```

---


## ▶️ Usage

### Project Structure
```text
unsupervised-cellular-phenotyping/
├── dockerfile                  # Docker setup for reproducible environment
├── main.py                     # Main pipeline entry point
├── pyproject.toml              # Python project metadata
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── data/                       # Place WSI files and model weights here
│   └── TCGA-V~1.GEO            # Example Region of Interest for plotting
├── hibou/                      # Submodule: Hibou repository
├── src/                        # Source code for pipeline modules
│   └── clustering/             # Clustering algorithms
│   └── data/                   # Data processing utilities
│   └── embedding/              # Embedding extraction code
│   └── models/                 # Model wrappers
│   └── utils/                  # Utility functions (analysis, export, viz)
```

---

### Running the Pipeline

The main pipeline is executed via `main.py`, which serves as the entrypoint in the Docker container. You can customize the run using the following command-line arguments:

| Argument          | Required | Default           | Description                                       |
|-------------------|----------|-------------------|---------------------------------------------------|
| `--input`         | Yes      | —                 | Path to the input WSI file (`.svs`).              |
| `--output`        | Yes      | —                 | Path to the output `.geojson` file.               |
| `--checkpoint`    | Yes      | —                 | Path to the model checkpoint (`.pth`).            |
| `--tile-size`     | No       | `256`             | Tile size for WSI tiling.                         |
| `--stride`        | No       | `256`             | Stride between tiles.                             |
| `--device`        | No       | `cuda`            | Device for inference (`cuda` or `cpu`).           |
| `--k`             | No       | `6`               | Number of clusters for KMeans.                    |
| `--pca`           | No       | `50`              | Number of PCA components before clustering.       |
| `--model-name`    | No       | `HibouLCellVIT`   | Model name written in output metadata.            |
| `--model-version` | No       | `1.0`             | Model version written in output metadata.         |
| `--magnification` | No       | `40.0`            | Magnification level of the WSI.                   |
| `--roi`           | No       | —                 | ROI `.geojson` file to restrict analysis.         |
| `--log_level`     | No       | `INFO`            | Logging level (`DEBUG`, `INFO`, `WARNING`, etc.). |
| `--help`          | No       | —                 | Show help message and exit.                       |

#### Example: Run the Pipeline in Docker

```bash
docker run --rm \
	--user "$(id -u):$(id -g)" \
	--gpus all \
	-v $(pwd):/app \
	cell-phenotyping \
	--input data/TCGA-V5-A7RE-11A-01-TS1.57401526-EF9E-49AC-8FF6-B4F9652311CE.svs \
	--output results/prediction.geojson \
	--checkpoint data/cellvit-hibou-l.pth \
	--pca 50 \
	--roi "data/TCGA-V~1.GEO"
```

> **Note:** Adjust the paths to your input WSI file and checkpoint as needed. Results will be saved in the directory specified by `--output`.


#### Example: Run the Pipeline with a Python Environment

If you prefer not to use Docker, you can run the pipeline directly in your local Python environment. Follow these steps:

1. **Create a Python Environment**
	 - Use Conda, venv, or another tool to create an environment with Python 3.10.
	 - Example with Conda:
		 ```bash
		 conda create -n cell-phenotyping python=3.10
		 conda activate cell-phenotyping
		 ```

2. **Install Dependencies**
	 - Install all required packages from `requirements.txt`:
		 ```bash
		 pip install -r requirements.txt
		 ```

3. **Run the Pipeline**
	 - Execute the main script with the desired arguments:
		 ```bash
		 python main.py \
			 --input data/TCGA-V5-A7RE-11A-01-TS1.57401526-EF9E-49AC-8FF6-B4F9652311CE.svs \
			 --output results/prediction.geojson \
			 --checkpoint data/cellvit-hibou-l.pth \
			 --pca 50 \
			 --roi "data/TCGA-V~1.GEO"
		 ```

> **Tip:** You can use all the same command-line arguments as described above for Docker. Adjust paths as needed for your setup.


#### Output Structure

After running the pipeline, results will be generated in the output directory (e.g., `./results/`):

- `prediction.geojson` — GeoJSON file containing clustering results.
- `cluster_analysis/` — Folder with confusion matrix, ARI, and other clustering analysis.
- `roi_visualization/` — Visualizations of clustering and supervised types for the selected ROI.

---


