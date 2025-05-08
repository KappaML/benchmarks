# KappaML Benchmarks

Benchmarking code and results of the [KappaML](https://kappaml.com) platform. 📊

Platform: https://kappaml.com
API Documentation: https://api.kappaml.com/docs
OpenAPI Schema: https://api.kappaml.com/openapi.json
API Keys: https://app.kappaml.com/api-keys


## Overview

This repository contains the code and results of the KappaML benchmarks.

### Datasets


### Results


## Running the benchmarks

Requirements:
- Python 3.10+
- pip

1. Install the dependencies

```bash
pip install -r requirements.txt
```

2. Get API key from https://app.kappaml.com/api-keys and set it as an environment variable:

```bash
export KAPPA_API_KEY=<your-api-key>
```

3. Run the benchmarks

```bash
python run_benchmarks.py
```

Check the results in the `results` folder.
