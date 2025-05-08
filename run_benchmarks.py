import os
import json
from pathlib import Path
from datetime import datetime
import logging
from dotenv import load_dotenv

from river import metrics, datasets
from river.evaluate import progressive_val_score

from benchmarks.tracks import get_tracks
from utils.kappaml_client import KappaMLClient
from utils.results_handler import save_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_benchmarks():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('KAPPA_API_KEY')
    
    if not api_key:
        raise ValueError("KAPPA_API_KEY environment variable not set")

    # Initialize KappaML client
    client = KappaMLClient(api_key)
    
    # Get all tracks to benchmark
    tracks = get_tracks()
    
    results = {}
    
    for track_name, track in tracks.items():
        logger.info(f"Running benchmark for track: {track_name}")
        
        # Get the dataset
        dataset = track.dataset
        
        # Get the metric
        metric = track.metric
        
        # Get the model from KappaML
        model = client.get_model(track_name)
        
        # Run the benchmark
        scores = progressive_val_score(
            dataset=dataset,
            model=model,
            metric=metric,
            print_every=100
        )
        
        results[track_name] = {
            'scores': list(scores),
            'final_score': metric.get(),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Finished {track_name} - Final score: {metric.get()}")
    
    # Save results
    save_results(results)

if __name__ == '__main__':
    run_benchmarks() 