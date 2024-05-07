# Recommender System with Transformer

Welcome to our recommender system project! This project uses a custom transformer-based model, `CustomBertNet`, and `Bert4Rec` for generating movie recommendations. The code and models are hosted in the `recommender_transformer-main` directory.

## Project Structure

- **`./data`**: This directory contains the datasets.
  - `kaggle_movie`: Dataset sourced from Kaggle.
  - `ml-25m`: Dataset from the MovieLens 25M collection.
- **`recommender/training_custom`**: Contains the custom implementation of our transformer-based model, `CustomBertNet`.
- **`recommender/train_utils`**: Utility scripts for training and model utilities.
- **`Final_inference.ipynb`**: Jupyter notebook for running the final inference after models are trained and data is processed.
- **`FinalProcessing.ipynb`**: Jupyter notebook for data preprocessing and preparation for training and inference.

## Setup Instructions

Follow these steps to set up and run the project:

### 1. Clone the Repository

Ensure you have Git installed and clone this repository to your local machine:

```bash
git clone [https://github.com/your-github/recommender_transformer-main.git](https://github.com/lianghaoDeng/EE541_movie_system.git)
cd recommender_transformer-main
```

### 2. Download and Prepare the Data

Download the datasets `kaggle_movie` and `ml-25m` and place them in the `/data` directory within the project:

```plaintext
- data/
  - kaggle_movie/
  - ml-25m/
```

### 3. Install Required Packages

Use the following command to install required Python packages and set up the environment:

```bash
python setup.py develop
```

### 4. Data Processing

Before running the inference notebook, process the data using:

```bash
jupyter notebook ../FinalProcessing.ipynb
```

Follow the instructions within the notebook to process and prepare your datasets. Note: data processing is within the EE541_movie_system Directory

### 5. Run Final Inference

After processing the data, start the Jupyter notebook to run the final inference:

```bash
jupyter notebook Final_inference.ipynb
```

## Additional Information

- **CustomBertNet**: Fully implemented by our team, located in `recommender/training_custom.py` and `recommender/train_utils.py`.
- **Bert4Rec**: Implementation adapted from an existing blog, details and adjustments are available within the project directory.

