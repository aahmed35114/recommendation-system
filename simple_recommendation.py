print("Running the recommendation script...")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Sample data
movies = {
    'title': ['Inception', 'Titanic', 'The Matrix', 'Interstellar', 'Avatar'],
    'features': ['sci-fi mind-bending', 'romantic ocean drama', 'sci-fi virtual reality', 'space sci-fi', 'sci-fi fantasy']
}

df = pd.DataFrame(movies)

# Vectorize the features
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['features'])

# Calculate similarity
similarity = cosine_similarity(count_matrix)

# Function to recommend movies
def recommend(title):
    if title not in df['title'].values:
        print("Movie not found.")
        return
    index = df[df['title'] == title].index[0]
    similar_movies = list(enumerate(similarity[index]))
    sorted_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    print(f"Because you watched '{title}', we recommend:")
    for i in sorted_movies[1:4]:
        print("-", df.iloc[i[0]].title)

# Example usage
recommend('Inception')
