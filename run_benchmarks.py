import os
import time
import json
import datetime
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore
from tqdm import tqdm
from river import datasets
from river.metrics import Accuracy, MAE
from kappaml import KappaML


# Get API key from https://app.kappaml.com/api-keys and set as env variable
# export KAPPAML_API_KEY="your_api_key_here"

DATASETS = {
    "regression": {
        "real": [
            # datasets.Bikes,
            datasets.ChickWeights,
            # datasets.Restaurants,
            datasets.TrumpApproval,
            # datasets.WaterFlow,
            # datasets.WebTraffic
        ],
        "synthetic": [
            datasets.synth.Friedman(
                seed=42
            ),
            datasets.synth.FriedmanDrift(
                position=(1_000, 5_000, 8_000),
                transition_window=1_000,
                seed=42
            ),
            datasets.synth.Mv(
                seed=42
            ),
            datasets.synth.Planes2D(
                seed=42
            ),
        ],
    },
    "classification": {
        "real": [
            datasets.Bananas,
            datasets.CreditCard,
            datasets.Elec2,
            datasets.Higgs,
            datasets.HTTP,
            datasets.MaliciousURL,
            datasets.Phishing,
        ],
        "synthetic": [
            datasets.synth.Agrawal(
                seed=42
            ),
            datasets.synth.AnomalySine(
                seed=42
            ),
            datasets.synth.ConceptDriftStream(
                seed=42
            ),
            datasets.synth.Hyperplane(
                seed=42
            ),
            datasets.synth.Mixed(
                seed=42
            ),
            datasets.synth.SEA(
                seed=42
            ),
            datasets.synth.Sine(
                seed=42
            ),
            datasets.synth.STAGGER(
                seed=42
            ),
            datasets.synth.LED(
                seed=42
            ),
            datasets.synth.LEDDrift(
                seed=42
            ),
            datasets.synth.RandomRBF(
                seed_model=42,
                seed_sample=42
            ),
            datasets.synth.RandomRBFDrift(
                seed_model=42,
                seed_sample=42
            ),
            datasets.synth.Waveform(
                seed=42
            ),
        ],
    },
}


def run_benchmark(client: KappaML, task: str, dataset, is_synthetic=False, semaphore=None):
    """Run benchmark for a single dataset.
    
    Args:
        client: KappaML client instance
        task: ML task type (regression/classification)
        dataset: River dataset class
        is_synthetic: Whether the dataset is synthetic
        semaphore: Optional semaphore to limit concurrent benchmarks
        
    Returns:
        dict: Benchmark results
    """
    if semaphore:
        semaphore.acquire()
    try:
        dataset_name = dataset.__name__
        print(f"Running benchmark for {dataset_name} ({task}) - Synth: {is_synthetic}")
        
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

        except Exception as e:
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
    finally:
        if semaphore:
            semaphore.release()


def run_worker_benchmark(task_dataset):
    """Run benchmark for a single dataset.
    
    Args:
        task_dataset: Tuple of (task, dataset, is_synthetic)
        
    Returns:
        dict: Benchmark results for the dataset
    """
    # Each worker gets its own KappaML client
    client = KappaML()
    
    # Each worker runs one benchmark at a time
    concurrent_limit = 1
    semaphore = Semaphore(concurrent_limit)
    
    task, dataset, is_synthetic = task_dataset
    result = run_benchmark(
        client, task, dataset, is_synthetic, 
        semaphore=semaphore
    )
    
    return result


def run_benchmarks():
    """Run benchmarks for all configured datasets using parallel workers."""
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare all tasks to be distributed among workers
    all_tasks = []
    for task in DATASETS:
        for dataset in DATASETS[task]["real"]:
            all_tasks.append((task, dataset, False))
        for dataset in DATASETS[task]["synthetic"]:
            all_tasks.append((task, dataset, True))
    
    results = []
    
    n_workers = 4
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = list(executor.map(run_worker_benchmark, all_tasks))
        for worker_results in futures:
            results.append(worker_results)
    
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
