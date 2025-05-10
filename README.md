# KappaML Python SDK

A Python client library for interacting with the KappaML platform. This SDK provides a simple interface for creating, training, and managing online machine learning models.

## Installation

```bash
pip install kappaml
```

## Quick Start

```python
from kappaml import KappaML

# Initialize the client
client = KappaML(api_key="your_api_key")  # Or set KAPPAML_API_KEY env variable

# Create a new model
model_id = client.create_model(
    name="my-regression-model",
    ml_type="regression"
)

# Train the model with a single data point
client.learn(
    model_id=model_id,
    features={"x1": 1.0, "x2": 2.0},
    target=3.0
)

# Make predictions
prediction = client.predict(
    model_id=model_id,
    features={"x1": 1.5, "x2": 2.5}
)

# Get model metrics
metrics = client.get_metrics(model_id)

# Clean up
client.delete_model(model_id)
```

## API Reference

### KappaML Class

#### Constructor

```python
client = KappaML(api_key: Optional[str] = None)
```

- `api_key`: Your KappaML API key. If not provided, will look for `KAPPAML_API_KEY` environment variable.

#### Methods

##### create_model

```python
def create_model(
    name: str,
    ml_type: str,
    wait_for_deployment: bool = True,
    timeout: int = 60
) -> str
```

Creates a new model on KappaML.

- `name`: Name of the model
- `ml_type`: Type of ML task ('regression' or 'classification')
- `wait_for_deployment`: Whether to wait for model deployment to complete
- `timeout`: Maximum time to wait for deployment in seconds
- Returns: The model ID

##### learn

```python
def learn(
    model_id: str,
    features: Dict[str, Any],
    target: Union[float, int, str]
) -> Dict[str, Any]
```

Train the model with a single data point.

- `model_id`: The model ID
- `features`: Dictionary of feature names and values
- `target`: The target value to learn from
- Returns: Response from the learning API

##### predict

```python
def predict(
    model_id: str,
    features: Dict[str, Any]
) -> Dict[str, Any]
```

Make predictions using a model.

- `model_id`: The model ID
- `features`: Dictionary of feature names and values
- Returns: Model predictions

##### get_metrics

```python
def get_metrics(model_id: str) -> Dict[str, Any]
```

Get current metrics for a model.

- `model_id`: The model ID
- Returns: Model metrics

##### delete_model

```python
def delete_model(model_id: str) -> None
```

Delete a model.

- `model_id`: The model ID to delete

## Error Handling

The SDK defines several exception classes for handling errors:

- `KappaMLError`: Base exception for all SDK errors
- `ModelNotFoundError`: Raised when a model is not found
- `ModelDeploymentError`: Raised when model deployment fails or times out

Example error handling:

```python
from kappaml import KappaML, ModelNotFoundError

client = KappaML()

try:
    metrics = client.get_metrics("non-existent-model")
except ModelNotFoundError:
    print("Model not found!")
```

## License

MIT
