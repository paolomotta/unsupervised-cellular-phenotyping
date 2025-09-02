import argparse 
import logging
import os

from src.data.wsi_reader import WSIReader
from src.data.tiling import generate_tiles, pad_to_size
from src.models import CellViTHibouWrapper
from src.clustering.clustering import cluster_rows
from src.utils.geojson_export import export_df_to_geojson
from src.embedding.embedding_extraction import per_tile_cell_embeddings, build_rows_for_saving
from src.utils.cluster_analysis import load_inputs, evaluate
from src.utils.roi_visualization import generate_roi_visualizations
from src.logging_config import configure_logging
from src.utils.reproducibility import set_seed
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run unsupervised cellular phenotyping on a WSI.")

    # Required arguments
    parser.add_argument("--input", required=True, help="Input WSI file.")
    parser.add_argument("--output", required=True, help="Output GeoJSON file.")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path.")
    parser.add_argument("--tile-size", type=int, default=256, help="Tile size.")
    parser.add_argument("--stride", type=int, default=256, help="Tile stride.")
    parser.add_argument("--device", default="cuda", help="Device for inference.")
    parser.add_argument("--k", type=int, default=6, help="Number of clusters (KMeans).")
    parser.add_argument("--pca", type=int, default=50, help="PCA components.")
    parser.add_argument("--model-name", default="HibouLCellVIT")
    parser.add_argument("--model-version", default="1.0")
    parser.add_argument("--magnification", type=float, default=40.0, help="Magnification level.")
    parser.add_argument("--roi", type=str, help="Path to the ROI GeoJSON file. If not provided, it won't generate the visualizations.")
    parser.add_argument("--log_level", default="INFO", help="Logging level.")
    return parser.parse_args()



def main():

    # Parse arguments
    args = parse_args()

    # Set up logging
    configure_logging(level=args.log_level)

    # Set the seed
    set_seed(42)

    # Download the checkpoint file if not present 
    if not os.path.exists(args.checkpoint):
        logging.info(f"Checkpoint {args.checkpoint} not found locally. Downloading from Hugging Face Hub...")
        load_dotenv()  # Load environment variables from .env file
        hf_token = os.getenv("HF_ACCESS_TOKEN")
        if hf_token is None:
            logging.error("Hugging Face access token not found in environment variables.")
            return
        args.checkpoint = hf_hub_download(
            repo_id="histai/cellvit-hibou-l",
            filename=os.path.basename(args.checkpoint),   # Extract filename from the provided path
            local_dir=os.path.dirname(args.checkpoint),    # Save in the same directory as specified
            token=hf_token
        )
        logging.info(f"Checkpoint downloaded to {args.checkpoint}.")


    logging.info(f"Processing WSI: {args.input}")

    # 1. Read WSI
    wsi_reader = WSIReader(args.input)
    h, w = wsi_reader.size()


    # 2. Generate tiles
    logging.info(f"Generating tiles with size {args.tile_size} and stride {args.stride}...")
    tiles = generate_tiles(h, w, tile_size=args.tile_size, stride=args.stride)

    # 3. Load the model 
    logging.info(f"Loading model from checkpoint: {args.checkpoint}...")
    model = CellViTHibouWrapper(ckpt_path=args.checkpoint, 
                                device=args.device, 
                                input_size=args.tile_size, 
                                patch_size=14, 
                                use_autocast=True, 
                                magnification=args.magnification)

    # 4. Run inference 
    logging.info(f"Running inference on {len(tiles)} tiles...")
    all_rows = []
    for tile in tiles:

        logging.debug(f"Processing tile at {tile.xywh}...")

        x, y, tw, th = tile.xywh
        tile_img = wsi_reader.read_region(x, y, tw, th)
        # Pad if needed
        if tile_img.shape[0] != args.tile_size or tile_img.shape[1] != args.tile_size:
            logging.debug(f"Padding tile from {tile_img.shape} to {(args.tile_size, args.tile_size)}...")
            tile_img, pad = pad_to_size(tile_img, args.tile_size, args.tile_size)
            tile.pad = pad

        # Model inference
        logging.debug(f"Running model inference...")
        tokens, instance_map = model.forward_tile(tile_img)

        # Embedding extraction
        logging.debug(f"Extracting embeddings...")
        cell_embeds, cell_meta = per_tile_cell_embeddings(
            tok=tokens,
            cells=instance_map,
            tile_hw=tile_img.shape[:2],
            input_size=args.tile_size
        )
        # Build rows
        logging.debug(f"Building dictionaries...")
        rows = build_rows_for_saving(cell_embeds, cell_meta, tile.xywh, tile.index)
        all_rows.extend(rows)
    

    # 5. Aggregate and cluster
    logging.info(f"Aggregating and clustering results...")
    df = cluster_rows(all_rows, algo="kmeans", k=args.k, pca=args.pca)


    # 6. Export to GeoJSON
    logging.info(f"Exporting results to GeoJSON: {args.output}...")
    export_df_to_geojson(
        df,
        slide_name=args.input,
        model_name=args.model_name,
        model_version=args.model_version,
        model_magnification=args.magnification,
        output_path=args.output
    )


    # 7. Running cluster analysis
    logging.info("Running cluster analysis...")
    df = load_inputs([args.output])
    save_dir = os.path.join(os.path.dirname(args.output), "cluster_analysis")
    evaluate(df, outdir=save_dir, plot=True)
    logging.info(f"Cluster analysis saved at {save_dir}.")

    # 8. Generate ROI visualizations
    if args.roi:
        logging.info("Generating ROI visualizations...")
        save_dir = os.path.join(os.path.dirname(args.output), "roi_visualization")
        generate_roi_visualizations(svs_path=args.input, roi_path=args.roi, cells_path=args.output, outdir=save_dir)

    logging.info("Pipeline complete.")


if __name__ == "__main__":
    main()
    