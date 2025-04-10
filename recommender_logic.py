import sqlite3
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, db_path="Movies.db"):
        self.conn = sqlite3.connect(db_path)
        self.movies = pd.read_sql_query("SELECT * FROM movies_titles", self.conn)
        self.ratings = pd.read_sql_query("SELECT * FROM movies_ratings", self.conn)
        self.prepare_models()

    def prepare_models(self):
        self.user_movie_df = self.ratings.pivot(index='user_id', columns='show_id', values='rating').fillna(0)
        self.svd = TruncatedSVD(n_components=20, random_state=42)
        self.sparse_matrix = csr_matrix(self.user_movie_df.values)
        self.reduced = self.svd.fit_transform(self.sparse_matrix)
        self.reconstructed = self.svd.inverse_transform(self.reduced)
        self.reconstructed_df = pd.DataFrame(self.reconstructed, index=self.user_movie_df.index, columns=self.user_movie_df.columns)

    def get_similar_movies_by_genre(self, genre, exclude_ids=[], top_n=15):
        genre_columns = [col for col in self.movies.columns if self.movies[col].isin([0, 1]).all()]
        
        # Dynamically create a genre string column
        self.movies['genre'] = self.movies[genre_columns].apply(
            lambda row: ', '.join([col for col in genre_columns if row[col] == 1]),
            axis=1
        )

        # Now filter using the reconstructed genre string
        filtered = self.movies[
            self.movies['genre'].str.contains(genre, na=False) &
            (~self.movies['show_id'].isin(exclude_ids))
        ].copy()

        filtered['image'] = filtered['show_id'] + '.jpg'

        return filtered[['show_id', 'title', 'image', 'genre']].head(top_n).to_dict(orient='records')


    def get_user_recommendations(self, user_id, top_n=20):
        if user_id not in self.reconstructed_df.index:
            return []

        scores = self.reconstructed_df.loc[user_id]
        already_rated = self.ratings[self.ratings['user_id'] == user_id]['show_id'].tolist()
        recommended = scores[~scores.index.isin(already_rated)].nlargest(top_n).index.tolist()

        # Grab movie metadata for the recommended IDs
        result = self.movies[self.movies['show_id'].isin(recommended)].copy()

        # 🔍 Auto-detect genre by looking for genre columns with value == 1
        genre_columns = [col for col in result.columns if result[col].isin([0, 1]).all()]
        result['genre'] = result[genre_columns].apply(lambda row: ', '.join([col for col in genre_columns if row[col] == 1]), axis=1)

        # 🧪 Optionally simulate image file names for now
        result['image'] = result['show_id'] + '.jpg'

        # Only return the fields the frontend needs
        return result[['show_id', 'title', 'image', 'genre']].to_dict(orient='records')

    def get_user_recommendations_with_genre_rows(self, user_id, top_n=20, genre_expansion_n=15):
        if user_id not in self.reconstructed_df.index:
            return {'recommended': [], 'by_genre': {}}

        scores = self.reconstructed_df.loc[user_id]
        already_rated = self.ratings[self.ratings['user_id'] == user_id]['show_id'].tolist()
        recommended_ids = scores[~scores.index.isin(already_rated)].nlargest(top_n).index.tolist()

        # Get metadata for these recommended movies
        recommended_movies = self.movies[self.movies['show_id'].isin(recommended_ids)].copy()

        # Dynamically build genre string
        genre_columns = [col for col in self.movies.columns if self.movies[col].isin([0, 1]).all()]
        recommended_movies['genre'] = recommended_movies[genre_columns].apply(
            lambda row: ', '.join([col for col in genre_columns if row[col] == 1]),
            axis=1
        )
        recommended_movies['image'] = recommended_movies['show_id'] + '.jpg'

        # Build initial response
        result = {
            "recommended": recommended_movies[['show_id', 'title', 'image', 'genre']].to_dict(orient="records"),
            "by_genre": {}
        }

        # Expand: Find more movies from genres in the recommendations
        seen_ids = set(recommended_ids)
        added_genres = set()

        for genre_str in recommended_movies['genre']:
            if not genre_str:
                continue

            for genre in genre_str.split(', '):
                if genre and genre not in added_genres:
                    similar = self.get_similar_movies_by_genre(
                        genre=genre,
                        exclude_ids=list(seen_ids),
                        top_n=genre_expansion_n
                    )
                    if similar:
                        result['by_genre'][genre] = similar
                        added_genres.add(genre)
                        seen_ids.update([movie['show_id'] for movie in similar])

                if len(added_genres) >= 3:
                    break

            if len(added_genres) >= 3:
                break

        return result

def get_content_based_recommendations(movie_id, db_path="Movies.db", top_n=5):
   # Load movie data
   conn = sqlite3.connect(db_path)
   movies_df = pd.read_sql_query("SELECT * FROM movies_titles", conn)
   conn.close()


   if 'show_id' not in movies_df.columns or movie_id not in movies_df['show_id'].values:
       return []


   # Create a combined content field
   content_fields = [col for col in ['title', 'description', 'director', 'cast'] if col in movies_df.columns]
   if not content_fields:
       return []


   movies_df['content'] = movies_df[content_fields].fillna('').agg(' '.join, axis=1)


   tfidf = TfidfVectorizer(stop_words='english')
   tfidf_matrix = tfidf.fit_transform(movies_df['content'])


   idx = movies_df[movies_df['show_id'] == movie_id].index[0]
   cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
   similar_indices = cosine_sim.argsort()[::-1][1:top_n+1]


   result = movies_df.iloc[similar_indices].copy()


   # Build genre field from one-hot genre flags
   genre_columns = [col for col in result.columns if result[col].dropna().isin([0, 1]).all()]
   result['genre'] = result[genre_columns].apply(
       lambda row: ', '.join([col for col in genre_columns if row[col] == 1]), axis=1
   ) if genre_columns else ""


   result['image'] = result['show_id'] + '.jpg'


   return result[['show_id', 'title', 'image', 'genre']].to_dict(orient='records')

def get_hybrid_recommendations(movie_id, db_path="Movies.db", top_n=5):
    import sqlite3
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import TruncatedSVD
    from scipy.sparse import csr_matrix

    # Load data
    conn = sqlite3.connect(db_path)
    movies = pd.read_sql_query("SELECT * FROM movies_titles", conn)
    ratings = pd.read_sql_query("SELECT * FROM movies_ratings", conn)
    conn.close()

    if movie_id not in movies['show_id'].values:
        return []

    # 1. Combine genre tags
    genre_cols = [col for col in movies.columns if movies[col].dropna().isin([0, 1]).all()]
    movies['genres_combined'] = movies[genre_cols].apply(
        lambda row: ' '.join([col for col, val in row.items() if val == 1]), axis=1
    )

    # 2. Build content field
    movies['description'] = movies['description'].fillna('')
    movies['content'] = movies['description'] + ' ' + movies['genres_combined']

    # 3. TF-IDF content scoring
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['content'])
    idx = movies[movies['show_id'] == movie_id].index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    movies['content_score'] = cosine_sim

    # 4. Collaborative filtering (SVD)
    user_movie_df = ratings.pivot(index='user_id', columns='show_id', values='rating').fillna(0)
    svd = TruncatedSVD(n_components=20, random_state=42)
    reduced = svd.fit_transform(csr_matrix(user_movie_df.values))
    reconstructed = svd.inverse_transform(reduced)
    reconstructed_df = pd.DataFrame(reconstructed, index=user_movie_df.index, columns=user_movie_df.columns)

    users_who_rated = ratings[ratings['show_id'] == movie_id]['user_id'].unique()
    if len(users_who_rated) > 0:
        predicted_scores = reconstructed_df.loc[users_who_rated].mean()
        predicted_scores.name = 'collab_score'
        movies = movies.merge(predicted_scores, left_on='show_id', right_index=True, how='left')
        movies['collab_score'] = movies['collab_score'].fillna(0)
    else:
        movies['collab_score'] = 0

    # 5. Start hybrid score
    movies['hybrid_score'] = (
        0.8 * movies['content_score'] +
        0.2 * movies['collab_score']
    )

    # 6. Region/language boost
    input_country = movies.loc[idx, 'country'] if 'country' in movies.columns else None
    if input_country:
        movies['region_score'] = movies['country'].apply(lambda x: 0.05 if x == input_country else 0)
        movies['hybrid_score'] += movies['region_score']

    # 7. Format boost (TV vs Movie)
    input_type = movies.loc[idx, 'type'] if 'type' in movies.columns else None
    if input_type:
        movies['format_score'] = movies['type'].apply(lambda x: 0.05 if x == input_type else 0)
        movies['hybrid_score'] += movies['format_score']

    # 8. Add genre string and image path
    movies['genre'] = movies[genre_cols].apply(
        lambda row: ', '.join([col for col in genre_cols if row[col] == 1]), axis=1
    ) if genre_cols else ''
    movies['image'] = movies['show_id'] + '.jpg'

    # 9. Filter results
    result = movies[(movies['show_id'] != movie_id)].copy()

    # Exclude kids/anime unless input contains them
    input_genre = movies.loc[idx, 'genres_combined'].lower()
    allow_kids = any(k in input_genre for k in ['kids', 'anime', 'cartoon'])
    if not allow_kids:
        exclude_keywords = ['Kids', 'Anime', 'Cartoon']
        result = result[~result['genre'].str.contains('|'.join(exclude_keywords), case=False, na=False)]

    # Drop duplicates by title and sort
    result = result.drop_duplicates(subset='title')
    result = result.sort_values('hybrid_score', ascending=False)

    # 10. Fallback if fewer than top_n
    top_results = result.head(top_n)
    if len(top_results) < top_n:
        backup = movies[
            ~movies['show_id'].isin(top_results['show_id']) &
            (movies['show_id'] != movie_id)
        ].sort_values('content_score', ascending=False).head(top_n - len(top_results))
        top_results = pd.concat([top_results, backup])

    # Final output
    return top_results[['show_id', 'title', 'image', 'genre']].head(top_n).to_dict(orient='records')