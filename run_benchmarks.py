import os
import time
import json
import datetime
from tqdm import tqdm
from river import datasets
from river.metrics import Accuracy, MAE
from kappaml import (
    KappaML,
    ModelNotFoundError,
    ModelDeploymentError
)

# Get API key from https://app.kappaml.com/api-keys and set as env variable
# export KAPPAML_API_KEY="your_api_key_here"

DATASETS = {
    "regression": {
        "real": [
            # datasets.AirlinePassengers,
            # datasets.Bikes,
            datasets.ChickWeights,
            # datasets.MovieLens100K,
            # datasets.Restaurants,
            # datasets.TrumpApproval,
            # datasets.WaterFlow,
            # datasets.WebTraffic
        ],
        "synthetic": [
            datasets.synth.Friedman,
            # datasets.synth.FriedmanDrift,
            # datasets.synth.Mv,
            # datasets.synth.Planes2D
        ],
    },
    "classification": {
        "real": [
            # datasets.Bananas,
            # datasets.CreditCard,
            # datasets.Elec2,
            # datasets.Higgs,
            # datasets.HTTP,
            # datasets.MaliciousURL,
            datasets.Phishing,
        ],
        "synthetic": [
            # datasets.synth.Agrawal,
            # datasets.synth.AnomalySine,
            # datasets.synth.ConceptDriftStream,
            # datasets.synth.Hyperplane,
            # datasets.synth.Mixed,
            # datasets.synth.SEA,
            # datasets.synth.Sine,
            # datasets.synth.STAGGER,
            # datasets.synth.LED,
            # datasets.synth.LEDDrift,
            # datasets.synth.RandomRBF,
            # datasets.synth.RandomRBFDrift,
            # datasets.synth.RandomTree,
            datasets.synth.Waveform
        ],
    },
}


def run_benchmark(client: KappaML, task: str, dataset, is_synthetic=False):
    """Run benchmark for a single dataset.
    
    Args:
        client: KappaML client instance
        task: ML task type (regression/classification)
        dataset: River dataset class
        is_synthetic: Whether the dataset is synthetic
        
    Returns:
        dict: Benchmark results
    """
    dataset_name = dataset.__name__
    print(f"Running benchmark for {dataset_name}")
    print(f"Synthetic: {is_synthetic}")
    
    # Set number of samples to run the benchmark on
    n_samples = 10_000
    if is_synthetic:
        dataset = dataset().take(n_samples)
    else:
        dataset = dataset()
        n_samples = dataset.n_samples
    
    # Intialise local metrics
    metric = Accuracy() if task == "classification" else MAE()
        
    result = {
        "dataset": dataset_name,
        "is_synthetic": is_synthetic,
        "n_samples": n_samples,
        "task": task,
        "timestamp": datetime.datetime.now().isoformat(),
        "status": "completed",
        # Metrics at every 100 samples
        "metrics": [],
        # Local metrics
        "local_metrics": [],
        # Final metrics: metrics at the end of the dataset
        "final_metrics": {},
        # Local final metrics
        "local_final_metrics": {}
    }

    try:
        # Create model
        model_id = client.create_model(
            name=f"benchmark-{dataset_name}",
            ml_type=task
        )
        result["model_id"] = model_id

        # Run the benchmark
        start_time = time.time()
        for i, (x, y) in tqdm(
            enumerate(dataset),
            total=n_samples,
            desc=dataset_name
        ):
            # Update local metrics
            metric.update(y, client.predict(model_id, x))

            # Learn from the data point
            client.learn(model_id=model_id, features=x, target=y)

            # Get metrics every 100 samples
            if i % 100 == 0:
                metrics = client.get_metrics(model_id)
                result["metrics"].append({
                    "samples": i,
                    "time": time.time() - start_time,
                    "metrics": metrics
                })
                # Update local metrics
                result["local_metrics"].append({
                    "samples": i,
                    "time": time.time() - start_time,
                    "metrics": {
                        "metric": {
                            "name": metric.__class__.__name__,
                            "value": metric.get()
                        }
                    }
                })

        # Final metrics
        result["final_metrics"] = client.get_metrics(model_id)
        result["local_final_metrics"] = metric.get()

    except (ModelNotFoundError, ModelDeploymentError) as e:
        print(f"Error during benchmark: {str(e)}")
        result["status"] = "failed"
        result["error"] = str(e)

    finally:
        # Clean up - delete the model
        try:
            if "model_id" in result:
                client.delete_model(result["model_id"])
                print(f"Successfully deleted model {result['model_id']}")
        except Exception as e:
            print(f"Failed to delete model: {str(e)}")
            result["status"] = "failed_deletion"
            
    return result


def run_benchmarks():
    """Run benchmarks for all configured datasets."""
    # Initialize KappaML client
    client = KappaML()
    
    results = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run benchmarks for regression and classification
    for task in DATASETS:
        for dataset in DATASETS[task]["real"]:
            result = run_benchmark(client, task, dataset, is_synthetic=False)
            results.append(result)
        for dataset in DATASETS[task]["synthetic"]:
            result = run_benchmark(client, task, dataset, is_synthetic=True)
            results.append(result)
    
    # Save results to JSON file
    fname = f"results_{timestamp}.json"
    results_file = os.path.join(results_dir, fname)
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "results": results
        }, f, indent=2)


if __name__ == "__main__":
    run_benchmarks()
