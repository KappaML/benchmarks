import os
import time
import json
import random
import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore
from tqdm import tqdm
from river import datasets
from river.metrics import Accuracy, MAPE
from river import dummy, stats, preprocessing, linear_model, neighbors
from kappaml import KappaML


# Get API key from https://app.kappaml.com/api-keys and set as env variable
# export KAPPAML_API_KEY="your_api_key_here"

"""
Real Datasets from river.datasets

**Regression**

| Name                                               | Samples   |   Features |
|:---------------------------------------------------|:----------|-----------:|
| [AirlinePassengers](../datasets/AirlinePassengers) | 144       |          1 |
| [Bikes](../datasets/Bikes)                         | 182,470   |          8 |
| [ChickWeights](../datasets/ChickWeights)           | 578       |          3 |
| [MovieLens100K](../datasets/MovieLens100K)         | 100,000   |         10 |
| [Restaurants](../datasets/Restaurants)             | 252,108   |          7 |
| [Taxis](../datasets/Taxis)                         | 1,458,644 |          8 |
| [TrumpApproval](../datasets/TrumpApproval)         | 1,001     |          6 |
| [WaterFlow](../datasets/WaterFlow)                 | 1,268     |          1 |


**Binary classification**

| Name                                     | Samples    | Features   |Sparse |
|:-----------------------------------------|:-----------|:-----------|:-----|
| [Bananas](../datasets/Bananas)           | 5,300      | 2          |      |
| [CreditCard](../datasets/CreditCard)     | 284,807    | 30         |      |
| [Elec2](../datasets/Elec2)               | 45,312     | 8          |      |
| [Higgs](../datasets/Higgs)               | 11,000,000 | 28         |      |
| [HTTP](../datasets/HTTP)                 | 567,498    | 3          |      |
| [Phishing](../datasets/Phishing)         | 1,250      | 9          |      |
| [SMSSpam](../datasets/SMSSpam)           | 5,574      | 1          |      |
| [SMTP](../datasets/SMTP)                 | 95,156     | 3          |      |
| [TREC07](../datasets/TREC07)             | 75,419     | 5          |      |

**Multi-class classification**

| Name                                       | Samples   |   Features |Classes|
|:-------------------------------------------|:----------|-----------:|------:|
| [ImageSegments](../datasets/ImageSegments) | 2,310     |         18 |     7 |
| [Insects](../datasets/Insects)             | 52,848    |         33 |     6 |
| [Keystroke](../datasets/Keystroke)         | 20,400    |         31 |    51 |
"""

"""
Synthetic Datasets from river.datasets.synth
**Regression**

| Name                                             |   Features |
|:-------------------------------------------------|-----------:|
| [Friedman](../datasets/synth/Friedman)           |         10 |
| [FriedmanDrift](../datasets/synth/FriedmanDrift) |         10 |
| [Mv](../datasets/synth/Mv)                       |         10 |
| [Planes2D](../datasets/synth/Planes2D)           |         10 |

**Binary classification**

| Name                                                       |   Features |
|:-----------------------------------------------------------|-----------:|
| [Agrawal](../datasets/synth/Agrawal)                       |          9 |
| [AnomalySine](../datasets/synth/AnomalySine)               |          2 |
| [ConceptDriftStream](../datasets/synth/ConceptDriftStream) |          9 |
| [Hyperplane](../datasets/synth/Hyperplane)                 |         10 |
| [Mixed](../datasets/synth/Mixed)                           |          4 |
| [SEA](../datasets/synth/SEA)                               |          3 |
| [Sine](../datasets/synth/Sine)                             |          2 |
| [STAGGER](../datasets/synth/STAGGER)                       |          3 |


**Multi-class classification**

| Name                                               |   Features |   Classes |
|:---------------------------------------------------|-----------:|----------:|
| [LED](../datasets/synth/LED)                       |          7 |        10 |
| [LEDDrift](../datasets/synth/LEDDrift)             |          7 |        10 |
| [RandomRBF](../datasets/synth/RandomRBF)           |         10 |         2 |
| [RandomRBFDrift](../datasets/synth/RandomRBFDrift) |         10 |         2 |
| [RandomTree](../datasets/synth/RandomTree)         |         10 |         2 |
| [Waveform](../datasets/synth/Waveform)             |         21 |         3 |

"""

# Baseline models for comparison
BASELINE_MODELS = {
    "regression": {
        "Linear Regression": preprocessing.StandardScaler() | linear_model.LinearRegression(),
        # "Dummy - Mean": dummy.StatisticRegressor(stats.Mean()),
    },
    "classification": {
        "KNN": preprocessing.StandardScaler() | neighbors.KNNClassifier(),
        # "Dummy - No Change": dummy.NoChangeClassifier(),
    }
}

# River datasets to run the benchmark on
DATASETS = {
    "regression": {
        "real": [
            datasets.ChickWeights,
            datasets.TrumpApproval,
            # datasets.Bikes,
            # datasets.Restaurants,
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
            # datasets.synth.Mv(
            #     seed=42
            # ),
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

# Maximum number of samples to run the benchmark on
MAX_N_SAMPLES = 20_000


def create_dataset_instance(dataset_class, dataset_name, is_synthetic, n_samples):
    """Create a fresh dataset instance with appropriate parameters.
    
    Args:
        dataset_class: The dataset class to instantiate
        dataset_name: Name of the dataset
        is_synthetic: Whether it's a synthetic dataset
        n_samples: Number of samples to take
        
    Returns:
        A dataset instance limited to n_samples
    """
    if is_synthetic:
        if dataset_name == "FriedmanDrift":
            return dataset_class(
                position=(1_000, 5_000, 8_000),
                transition_window=1_000,
                seed=42
            ).take(n_samples)
        elif dataset_name in ["RandomRBF", "RandomRBFDrift"]:
            return dataset_class(
                seed_model=42, 
                seed_sample=42
            ).take(n_samples)
        else:
            try:
                return dataset_class(seed=42).take(n_samples)
            except TypeError:
                # Fallback for other datasets with special parameters
                print(f"Warning: Special parameter handling needed for {dataset_name}")
                return dataset_class(
                    seed_model=42, 
                    seed_sample=42
                ).take(n_samples)
    else:
        # Real datasets
        return dataset_class().take(n_samples)


async def run_benchmark(task: str, dataset, is_synthetic=False, semaphore=None):
    """Run benchmark for a single dataset.
    
    Args:
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
        # Get dataset name
        if is_synthetic:
            dataset_name = dataset.__class__.__name__
            dataset_class = dataset.__class__
        else:
            dataset_name = dataset.__name__
            dataset_class = dataset
            
        print(f"Running benchmark for {dataset_name} ({task}) - Synth: {is_synthetic}")
        
        # Set number of samples to run the benchmark on
        n_samples = MAX_N_SAMPLES
        if is_synthetic:
            dataset = dataset.take(n_samples)
        else:
            n_samples = min(dataset().n_samples, n_samples)
            dataset = create_dataset_instance(dataset_class, dataset_name, is_synthetic, n_samples)

        # Initialize local metrics
        metric = Accuracy() if task == "classification" else MAPE()
            
        result = {
            "dataset": dataset_name,
            "is_synthetic": is_synthetic,
            "n_samples": n_samples,
            "task": task,
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "completed",
            "metrics": [],
            "local_metrics": [],
            "final_metrics": {},
            "local_final_metrics": {},
            "baseline_metrics": {},
            "baseline_final_metrics": {}
        }

        try:
            # Create async client and run event loop for API calls
            client = KappaML()
            
            # Create model
            time.sleep(random.random() * 10)
            model_id = await client.create_model(
                name=f"benchmark-{dataset_name}",
                ml_type=task,
                wait_for_deployment=True,
                timeout=3 * 60
            )
            result["model_id"] = model_id

            # Run the benchmark
            start_time = time.time()
            # Number of samples to process in parallel
            batch_size = 10
            batch_features = []
            batch_targets = []
            
            for i, (x, y) in tqdm(
                enumerate(dataset),
                total=n_samples,
                desc=dataset_name
            ):
                batch_features.append(x)
                batch_targets.append(y)
                
                # When batch is full or at end of dataset, process it
                if len(batch_features) == batch_size or i == n_samples - 1:
                    # Run predictions and learning in parallel
                    try:
                        predictions = await asyncio.gather(*[
                            client.predict(model_id, x) for x in batch_features
                        ])
                        
                        # Update metrics
                        for y_true, y_pred in zip(batch_targets, predictions):
                            metric.update(y_true, y_pred)
                        
                        # Run learning operations in parallel
                        await asyncio.gather(*[
                            client.learn(model_id=model_id, features=x, target=y) 
                            for x, y in zip(batch_features, batch_targets)
                        ])
                    except Exception as e:
                        # Skip this batch if any prediction fails, it's the simplest way
                        print(f"Error during batch processing: {str(e)}; skipping batch")
                        pass
                    
                    # Clear batches
                    batch_features = []
                    batch_targets = []
                
                # Get metrics every 250 samples
                if i % 250 == 0:
                    metrics = await client.get_metrics(model_id)
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

            # # Final metrics
            result["final_metrics"] = await client.get_metrics(model_id)
            result["local_final_metrics"] = metric.get()

            # Run baseline models
            baseline_metrics = {}
            baseline_final_metrics = {}
            for model_name, model in BASELINE_MODELS[task].items():
                # Create new metric and model instance for each baseline
                model = model.clone()
                baseline_metric = Accuracy() if task == "classification" else MAPE()
                baseline_metrics[model_name] = []
                model_start = time.time()
                
                # Create fresh dataset instance for each baseline model
                current_dataset = create_dataset_instance(
                    dataset_class, dataset_name, is_synthetic, n_samples
                )
                
                for i, (x, y) in tqdm(
                    enumerate(current_dataset),
                    total=n_samples,
                    desc=f"Running baseline {model_name}"
                ):
                    y_pred = model.predict_one(x)
                    baseline_metric.update(y, y_pred)
                    model.learn_one(x, y)
                
                    if i % 250 == 0:
                        baseline_metrics[model_name].append({
                            "samples": i,
                            "time": time.time() - model_start,
                            "metrics": {
                                "metric": {
                                    "name": baseline_metric.__class__.__name__,
                                    "value": baseline_metric.get()
                                }
                            }
                        })
                
                baseline_final_metrics[model_name] = baseline_metric.get()
                
            result["baseline_metrics"] = baseline_metrics
            result["baseline_final_metrics"] = baseline_final_metrics

        except Exception as e:
            print(f"Error during benchmark: {str(e)}")
            result["status"] = "failed"
            result["error"] = str(e)

        finally:
            # Clean up - delete the model
            try:
                if "model_id" in result:
                    await client.delete_model(result["model_id"])
                    print(f"Successfully deleted model {result['model_id']}")
            except Exception as e:
                print(f"Failed to delete model: {str(e)}")
                result["status"] = "failed_deletion"
            
        return result
        
    finally:
        if semaphore:
            semaphore.release()


async def run_worker_benchmark(task_dataset):
    """Run benchmark for a single dataset.
    
    Args:
        task_dataset: Tuple of (task, dataset, is_synthetic)
        
    Returns:
        dict: Benchmark results for the dataset
    """
    # Each worker runs one benchmark at a time
    concurrent_limit = 1
    semaphore = Semaphore(concurrent_limit)
    
    task, dataset, is_synthetic = task_dataset
    result = await run_benchmark(
        task, dataset, is_synthetic, 
        semaphore=semaphore
    )
    
    return result


async def run_benchmarks():
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
    
    n_workers = 8
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(executor, lambda t=t: asyncio.run(run_worker_benchmark(t)))
            for t in all_tasks
        ]
        results = await asyncio.gather(*futures)
    
    # Save results to JSON file
    fname = f"results_{timestamp}.json"
    results_file = os.path.join(results_dir, fname)
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "results": results
        }, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
