from flask import Flask, jsonify, request
from recommender_logic import Recommender, get_content_based_recommendations, get_hybrid_recommendations
from flask_cors import CORS
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

recommender = Recommender()

@app.route("/api/recommend/user/<int:user_id>", methods=["GET"])
def recommend_for_user(user_id):
    try:
        recs = recommender.get_user_recommendations_with_genre_rows(user_id)
        return jsonify(recs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/recommend/movie/<movie_id>", methods=["GET"])
def recommend_for_movie(movie_id):
    try:
        recs = get_content_based_recommendations(movie_id)
        return jsonify(recs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/recommend/hybrid/<movie_id>", methods=["GET"])
def hybrid_recommendations(movie_id):
    try:
        recs = get_hybrid_recommendations(movie_id)
        return jsonify(recs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/search", methods=["GET"])
def search_movies():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([])

    conn = sqlite3.connect("Movies.db")
    df = pd.read_sql_query("SELECT show_id, title, description FROM movies_titles", conn)
    conn.close()

    df['description'] = df['description'].fillna("")
    df['combined'] = df['title'] + " " + df['description']

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined'])

    query_vec = tfidf.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    df['similarity'] = similarities

    top_matches = df.sort_values(by='similarity', ascending=False).head(15)
    top_matches['image'] = top_matches['show_id'] + ".jpg"

    return jsonify(top_matches[['show_id', 'title', 'image']].to_dict(orient='records'))

@app.route("/")
def index():
    return "✅ Movie recommender API is live!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
