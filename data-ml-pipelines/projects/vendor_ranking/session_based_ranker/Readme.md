# Recommender Model

This repository contains the implementation of a recommender model built using PyTorch and Hugging Face transformers. The model supports training, saving, and loading for serving for both offline and online components.

## Overview

### Features
1. **Training the Recommender Model**: Train the whole recommender model using offline and online features.
2. **Saving the Model**: Save the model in three parts:
   - Whole model: The entire recommender system including both offline and online models.
   - Offline model: The offline model used to generate user embeddings.
   - Online model: The online model used to combine offline embeddings with online features.
3. **Serving the Model**: 
   - Load the offline model and generate offline user embeddings.
   - Load the online model, use online features and offline user embeddings to get the output of the online model.

### What's Remaining
1. Adapting the code to register models to mlflow and load it from mlflow
2. Save only one tokenizer if it's shared (not crucial currently as files are small, possibly due to limited training batches).
2. Ensure the whole model can be loaded and generate the same logits.
3. Refactor logic for offline feature detection (currently, all chain_id features are classified to the offline model, limiting the addition of features like `sess_clicks` easily).
4. Extend the model to support various feature types, including online features (currently only categorical features like `prev_chains` and numeric features like `order_hour` and `area_id` are supported).
5. Extend the feature configuration to include all model features (currently, it only covers offline features to reduce code redundancy).
6. loading the right tokenizer for evaluation (now all are the same so it's not causing a problem)
7. saving area_id_to_index from training (it's currently used from training)

## How to Run the Code
### Step 1: Set Up the Environment
1. Create a Conda Environment:
   - Run the following command to create a conda environment from the cloud_env.yaml file:
      ```bash
      conda env create -n <yourenvname> python=3.9
      ````
   - Activate the environment:
      ```bash
      conda activate <yourenvname> 
      pip install -r session_cloud.txt
      ```
-  Step 2: Train the Model
   - Run the train_refactored.py script to train the recommender model:
      ```bash
      python train_refactored.py
      ```
## Saving and Using the Models
After training, the model can be saved and used for generating predictions. Below are the steps to save the models and use them for generating recommendations.

### Saving the Model
```python
model.save_models("./trained_model")
```
### Generating Offline Embeddings
Load the offline model and generate offline embeddings:
```python
# Load the offline model
offline_model = RecommenderClass.load_offline_model("./trained_model")

# Prepare inputs for offline model (example)
inputs = {
    'sess_clicks_ids': session_clicks_ids,
    'sess_clicks_mask': session_clicks_mask,
    'prev_chains_ids': prev_chains_ids,
    'prev_chains_mask': prev_chains_mask,
    'freq_chains_ids': freq_chains_ids,
    'freq_chains_mask': freq_chains_mask,
}

# Generate offline embeddings
offline_embeddings = offline_model(**inputs)
```

### Generating Online Recommendations
Load the online model and use it along with the offline embeddings to generate recommendations:

```python
# Load the online model
online_model = RecommenderClass.load_online_model("./trained_model")

# Prepare inputs for online model (example)
order_hour = torch.tensor([...])  # Replace with actual data
delivery_area_id = torch.tensor([...])  # Replace with actual data

# Generate recommendations using online features and offline embeddings
loss, logits = online_model(offline_embeddings, order_hour, delivery_area_id, labels)
```
### Loading the Whole Model
To load the entire recommender model, including both the offline and online parts:

```python
model = RecommenderClass.load_model("./trained_model")
```
### Making Predictions with the Loaded Model
After loading the model, you can use it to make predictions:

### Prepare inputs
```python
inputs = {
    'sess_clicks_ids': session_clicks_ids,
    'sess_clicks_mask': session_clicks_mask,
    'prev_chains_ids': prev_chains_ids,
    'prev_chains_mask': prev_chains_mask,
    'freq_chains_ids': freq_chains_ids,
    'freq_chains_mask': freq_chains_mask,
    'order_hour': order_hour,
    'delivery_area_id': delivery_area_id,
    'labels': labels
}
```

###  Make predictions
```python
with torch.no_grad():
    loss, logits = model(**inputs)
```

## How to Add Features to the Model
The feature_configs dictionary is used to define the features used in the model. Each entry in the list represents a feature and its associated configurations.

### Example
```python
feature_configs = [
    {"name": "sess_clicks", "tokenizer": fast_tokenizer, "max_length": 10, "model_type": "T5"},
    {"name": "prev_chains", "tokenizer": fast_tokenizer, "max_length": 10, "model_type": "T5"},
    {"name": "freq_chains", "tokenizer": fast_tokenizer, "max_length": 10, "model_type": "T5"},
]
```
### Explanation
   - name: The name of the feature (e.g., sess_clicks, prev_chains, freq_chains).
   - tokenizer: The tokenizer used to preprocess this feature.
   - max_length: The maximum length for the tokenized inputs.
   - model_type: The type of transformer model used for this feature (e.g., T5, DistilBERT).

### Future Extensions
The feature_configs will be extended to include additional tags per feature, such as:

   - offline/online model: Specify whether the feature belongs to the offline or online model.
   - feature type: Define the type of feature (e.g., embedding vector, integer embedding, string embedding).
