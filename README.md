# Recommendation System

This is a simple content-based movie recommendation system built with Python.  
It uses `pandas`, `CountVectorizer`, and `cosine_similarity` from `scikit-learn` to recommend similar movies based on feature descriptions.

## How It Works
- A small dataset of movies with feature keywords
- Text features are vectorized and compared using cosine similarity
- The system recommends 3 movies similar to the one selected

## Requirements
- Python 3.10+
- pandas
- scikit-learn

## Run the Code
```bash
python3 simple_recommendation.py
