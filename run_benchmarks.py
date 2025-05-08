import os
import time
import requests
from river import datasets

# Get API key from https://app.kappaml.com/api-keys and set it as an env variable
# export KAPPAML_API_KEY="your_api_key_here"
API_KEY = os.getenv("KAPPAML_API_KEY")
BASE_URL = "https://api.kappaml.com/v1"

# Create a session object to reuse connections
session = requests.Session()
session.headers.update({"X-API-Key": API_KEY})

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
| [MaliciousURL](../datasets/MaliciousURL) | 2,396,130  | 3,231,961  | ✔️   |
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

DATASETS = {
    "regression": {
        "real": [
            datasets.AirlinePassengers,
            datasets.Bikes,
            datasets.ChickWeights,
            datasets.MovieLens100K,
            datasets.Restaurants,
            datasets.TrumpApproval,
            datasets.WaterFlow,
            datasets.WebTraffic
        ],
        "synthetic": [
            datasets.synth.Friedman,
            datasets.synth.FriedmanDrift,
            datasets.synth.Mv,
            datasets.synth.Planes2D
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
            datasets.synth.Agrawal,
            datasets.synth.AnomalySine,
            datasets.synth.ConceptDriftStream,
            datasets.synth.Hyperplane,
            datasets.synth.Mixed,
            datasets.synth.SEA,
            datasets.synth.Sine,
            datasets.synth.STAGGER,
            datasets.synth.LED,
            datasets.synth.LEDDrift,
            datasets.synth.RandomRBF,
            datasets.synth.RandomRBFDrift,
            datasets.synth.RandomTree,
            datasets.synth.Waveform
        ],
    },
}


def run_benchmark(model_id, dataset, is_synthetic=False):
    print(f"Running benchmark for {dataset.__name__}")
    print(f"Synthetic: {is_synthetic}")
    print(f"Model: {model_id}")


def create_and_verify_model(dataset_name, task):
    """Create a model and verify its deployment.

    Args:
        dataset_name (str): Name of the dataset/model
        task (str): ML task type (regression/classification)
   
    Returns:
        str: Model ID if successfully deployed, None otherwise
    """
    dataset_name = "benchmark-" + dataset_name
    model_data = {
        "id": dataset_name,
        "name": dataset_name,
        "ml_type": task
    }

    # Create the model
    response = session.post(f"{BASE_URL}/models", json=model_data)
    if response.status_code != 201:
        print(f"Failed to create model for {dataset_name}: {response.text}")
        return None
        
    model_id = response.json()["id"]
    
    # Wait for deployment and verify status
    max_retries = 6  # Maximum number of retries (1 minute total)
    for _ in range(max_retries):
        time.sleep(10)  # Wait 10 seconds between checks
        
        status_response = session.get(f"{BASE_URL}/models/{model_id}")
        if status_response.status_code != 200:
            print(f"Failed to get model status for {dataset_name}")
            return None
            
        status = status_response.json()["status"]
        if status == "Deployed":
            print(f"Model {dataset_name} successfully deployed")
            return model_id
        elif status == "Failed":
            print(f"Model {dataset_name} deployment failed")
            return None
            
    print(f"Timeout waiting for model {dataset_name} to deploy")
    return None


def run_benchmarks():
    for task in DATASETS:
        for dataset in DATASETS[task]["real"]:
            model_id = create_and_verify_model(dataset.__name__, task)
            if model_id:
                run_benchmark(model_id, dataset, is_synthetic=False)
        for dataset in DATASETS[task]["synthetic"]:
            model_id = create_and_verify_model(dataset.__name__, task)
            if model_id:
                run_benchmark(model_id, dataset, is_synthetic=True)


if __name__ == "__main__":
    run_benchmarks()
