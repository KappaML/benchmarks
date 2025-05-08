from river import datasets

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
    "Regression": {
        "Real": [
            datasets.AirlinePassengers,
            datasets.Bikes,
            datasets.ChickWeights,
            datasets.MovieLens100K,
            datasets.Restaurants,
            datasets.TrumpApproval,
            datasets.WaterFlow,
            datasets.WebTraffic
        ],
        "Synthetic": [
            datasets.synth.Friedman,
            datasets.synth.FriedmanDrift,
            datasets.synth.Mv,
            datasets.synth.Planes2D
        ],
    },
    "Classification": {
        "Real": [
            datasets.Bananas,
            datasets.CreditCard,
            datasets.Elec2,
            datasets.Higgs,
            datasets.HTTP,
            datasets.MaliciousURL,
            datasets.Phishing,
        ],
        "Synthetic": [
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


def run_benchmark(dataset, is_synthetic=False):
    print(f"Running benchmark for {dataset.__name__}")
    print(f"Synthetic: {is_synthetic}")


def run_benchmarks():
    for task in DATASETS:
        for dataset in DATASETS[task]["Real"]:
            run_benchmark(dataset, is_synthetic=False)
        for dataset in DATASETS[task]["Synthetic"]:
            run_benchmark(dataset, is_synthetic=True)


if __name__ == "__main__":
    run_benchmarks()
