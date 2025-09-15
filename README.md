# AWS SageMaker ML Model Deployment Demo

This demo shows how to train and deploy a simple machine learning model using AWS SageMaker. We'll create a basic linear regression model to predict house prices.

## Prerequisites

- AWS account with appropriate permissions
- Python 3.7+ installed
- AWS CLI configured
- SageMaker execution role

## Step 1: Environment Setup

### Install Required Libraries
```bash
pip install boto3 sagemaker pandas scikit-learn numpy matplotlib
```

### Import Libraries and Setup
```python
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import joblib
import os

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = get_execution_role()  # Or specify your SageMaker execution role ARN
bucket = sagemaker_session.default_bucket()
prefix = 'sagemaker-demo'

print(f"Using bucket: {bucket}")
print(f"Using role: {role}")
```

## Step 2: Create Sample Dataset

```python
# Generate synthetic house price data
def create_sample_data():
    # Features: house_size, bedrooms, age, location_score
    X, y = make_regression(
        n_samples=1000,
        n_features=4,
        noise=0.1,
        random_state=42
    )
    
    # Create feature names
    feature_names = ['house_size', 'bedrooms', 'age', 'location_score']
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['price'] = y
    
    # Make data more realistic
    df['house_size'] = (df['house_size'] * 500 + 2000).abs()  # 1500-2500 sq ft
    df['bedrooms'] = (df['bedrooms'] * 2 + 3).abs().round()   # 1-5 bedrooms
    df['age'] = (df['age'] * 10 + 15).abs()                  # 5-25 years
    df['location_score'] = (df['location_score'] * 2 + 8).abs()  # 6-10 score
    df['price'] = (df['price'] * 50000 + 300000).abs()       # $250K-$350K
    
    return df

# Create and save dataset
df = create_sample_data()
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Split data
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Save to CSV files
train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)

print(f"\nTraining data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
```

## Step 3: Create Training Script

Create a file called `train.py`:

```python
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def model_fn(model_dir):
    """Load model for inference"""
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return model

def input_fn(request_body, content_type='text/csv'):
    """Parse input data for inference"""
    if content_type == 'text/csv':
        df = pd.read_csv(pd.StringIO(request_body))
        return df.values
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """Make predictions"""
    prediction = model.predict(input_data)
    return prediction

def output_fn(prediction, accept='text/csv'):
    """Format output"""
    if accept == 'text/csv':
        return ','.join(map(str, prediction))
    else:
        raise ValueError(f"Unsupported accept type: {accept}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    
    args = parser.parse_args()
    
    # Load training data
    train_file = os.path.join(args.train, 'train.csv')
    df = pd.read_csv(train_file)
    
    # Prepare features and target
    feature_columns = ['house_size', 'bedrooms', 'age', 'location_score']
    X = df[feature_columns]
    y = df['price']
    
    print(f"Training data shape: {X.shape}")
    print(f"Features: {feature_columns}")
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate training metrics
    train_pred = model.predict(X)
    train_rmse = np.sqrt(mean_squared_error(y, train_pred))
    train_r2 = r2_score(y, train_pred)
    
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Model coefficients: {model.coef_}")
    print(f"Model intercept: {model.intercept_:.2f}")
    
    # Save model
    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))
    print("Model saved successfully!")
```

## Step 4: Upload Data to S3

```python
# Upload training data to S3
train_input = sagemaker_session.upload_data(
    path='train.csv',
    bucket=bucket,
    key_prefix=f'{prefix}/data'
)

test_input = sagemaker_session.upload_data(
    path='test.csv',
    bucket=bucket,
    key_prefix=f'{prefix}/data'
)

print(f"Training data uploaded to: {train_input}")
print(f"Test data uploaded to: {test_input}")
```

## Step 5: Create and Train SageMaker Estimator

```python
# Create SKLearn estimator
sklearn_estimator = SKLearn(
    entry_point='train.py',
    role=role,
    instance_type='ml.m5.large',
    framework_version='0.23-1',
    py_version='py3',
    script_mode=True,
    hyperparameters={
        'model-dir': '/opt/ml/model'
    }
)

# Start training job
print("Starting training job...")
sklearn_estimator.fit({'training': train_input})

print("Training completed!")
print(f"Model artifacts location: {sklearn_estimator.model_data}")
```

## Step 6: Deploy Model to Endpoint

```python
# Deploy model
print("Deploying model to endpoint...")

predictor = sklearn_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium',
    endpoint_name='house-price-predictor'
)

print(f"Model deployed to endpoint: {predictor.endpoint_name}")
```

## Step 7: Test the Deployed Model

```python
# Prepare test data for prediction
test_sample = test_data[['house_size', 'bedrooms', 'age', 'location_score']].head(5)
print("Test samples:")
print(test_sample)

# Convert to CSV format for prediction
import io
csv_buffer = io.StringIO()
test_sample.to_csv(csv_buffer, index=False, header=False)
test_csv = csv_buffer.getvalue()

# Make predictions
predictions = predictor.predict(test_csv)
print(f"\nPredictions: {predictions}")

# Compare with actual values
actual_prices = test_data['price'].head(5).values
print(f"Actual prices: {actual_prices}")

# Calculate differences
if isinstance(predictions, str):
    pred_list = [float(x) for x in predictions.split(',')]
else:
    pred_list = predictions

differences = actual_prices - pred_list
print(f"Differences: {differences}")
```

## Step 8: Monitor and Manage

```python
# Get endpoint status
endpoint_status = sagemaker_session.describe_endpoint(predictor.endpoint_name)
print(f"Endpoint Status: {endpoint_status['EndpointStatus']}")

# List all endpoints
endpoints = sagemaker_session.list_endpoints()
print("All endpoints:")
for ep in endpoints['Endpoints']:
    print(f"  - {ep['EndpointName']}: {ep['EndpointStatus']}")
```

## Step 9: Cleanup Resources

```python
# Delete endpoint to avoid charges
print("Cleaning up resources...")
predictor.delete_endpoint()
print("Endpoint deleted successfully!")

# Optional: Delete model and endpoint configuration
# predictor.delete_model()
```

## Demo Script Summary

This demo covers:

1. **Environment Setup**: Installing dependencies and configuring SageMaker
2. **Data Preparation**: Creating synthetic house price data
3. **Model Training**: Using scikit-learn LinearRegression in SageMaker
4. **Model Deployment**: Creating a real-time inference endpoint
5. **Testing**: Making predictions on new data
6. **Cleanup**: Removing resources to avoid costs

## Key SageMaker Concepts Demonstrated

- **Training Jobs**: Managed training with automatic scaling
- **Model Registry**: Storing trained model artifacts
- **Endpoints**: Real-time inference infrastructure
- **Batch Transform**: For batch predictions (not shown but available)

## Cost Considerations

- Training: ~$0.10-0.50 for ml.m5.large
- Endpoint: ~$0.05/hour for ml.t2.medium
- Storage: Minimal S3 costs for model artifacts

## Next Steps

1. Try different algorithms (Random Forest, XGBoost)
2. Add model validation and cross-validation
3. Implement A/B testing with multiple model variants
4. Set up automated model retraining
5. Add monitoring and logging with CloudWatch

## Troubleshooting Tips

- Ensure your execution role has SageMaker permissions
- Check CloudWatch logs for training job issues
- Verify S3 bucket permissions
- Monitor endpoint health in SageMaker console
