from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
import pandas as pd
import spacy
import re
import numpy as np
from langdetect import detect
import musicbrainzngs
from collections import Counter, defaultdict
import en_core_web_sm
import pt_core_news_sm
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist
from flask import redirect

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///music.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize caching
cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
cache.init_app(app)

# Initialize database
db = SQLAlchemy(app)
musicbrainzngs.set_useragent("RYM-Analyzer", "0.1", "your@email.com")

# Load NLP models
nlp_models = {
    'en': en_core_web_sm.load(),
    'pt': pt_core_news_sm.load()
}

# Database models
class Album(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    artist = db.Column(db.String(200), nullable=False)
    year = db.Column(db.Integer)
    genre = db.Column(db.String(100))
    rating = db.Column(db.Float)

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Float)
    album_id = db.Column(db.Integer, db.ForeignKey('album.id'))
    language = db.Column(db.String(2))
    matching_confidence = db.Column(db.Float)

# Text processing
def clean_text(text):
    text = re.sub(r'!\[.*?\]\(.*?\)', '', str(text))
    text = re.sub(r'[^a-zA-Z0-9\s\.,!?\'"-]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

class VectorSearch:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        self.review_vectors = None
        self.review_ids = None
        
    def build_index(self, reviews):
        """Build search index from reviews"""
        texts = [review.text for review in reviews]
        self.review_ids = [review.id for review in reviews]
        self.review_vectors = self.vectorizer.fit_transform(texts)
        
    def search(self, query, method='cosine', top_k=50):
        """Search reviews using specified similarity method"""
        query_vector = self.vectorizer.transform([query])
        
        if method == 'cosine':
            similarities = self._cosine_similarity(query_vector)
        elif method == 'dice':
            similarities = self._dice_coefficient(query_vector)
        else:  # jaccard
            similarities = self._jaccard_similarity(query_vector)
            
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.review_ids[i], similarities[i]) for i in top_indices]
    
    def _cosine_similarity(self, query_vector):
        """Calculate cosine similarity"""
        return (query_vector @ self.review_vectors.T).toarray().flatten()
    
    def _dice_coefficient(self, query_vector):
        """Calculate Dice coefficient"""
        intersection = (query_vector @ self.review_vectors.T).toarray().flatten()
        sum_vectors = np.asarray(query_vector.sum(1) + self.review_vectors.sum(1)).flatten()
        return 2 * intersection / sum_vectors
    
    def _jaccard_similarity(self, query_vector):
        """Calculate Jaccard similarity"""
        intersection = (query_vector @ self.review_vectors.T).toarray().flatten()
        sum_vectors = np.asarray(query_vector.sum(1) + self.review_vectors.sum(1)).flatten()
        return intersection / (sum_vectors - intersection)

class ReviewMatcher:
    def __init__(self):
        self.nlp_en = spacy.load("en_core_web_sm")
        self.nlp_pt = spacy.load("pt_core_news_sm") 
        self.albums = []
        self.artist_index = {}
        self.title_index = {}
        self.mb_cache = {}

    def _build_indices(self):
        """Build search indices from album data"""
        self.artist_index = defaultdict(list)
        self.title_index = defaultdict(list)
        
        for album in self.albums:
            # Index artist tokens
            for token in album['artist'].split():
                self.artist_index[token].append(album)
            # Index title tokens
            for token in album['title'].split():
                self.title_index[token].append(album)

    def _get_candidates(self, entities):
        """Get potential album matches using indices"""
        candidates = {}
        
        # Search by artist entities
        for artist in entities['ARTIST']:
            for token in artist.lower().split():
                for album in self.artist_index.get(token, []):
                    candidates[album['id']] = album
        
        # Search by title entities
        for title in entities['WORK_OF_ART']:
            for token in title.lower().split():
                for album in self.title_index.get(token, []):
                    candidates[album['id']] = album
                    
        return list(candidates.values()) or self.albums

    def _score_candidates(self, candidates, entities):
        """Score candidates using similarity metrics"""
        if not candidates:
            return (None, 0)
            
        artist_scores = [
            max([self._jaccard_similarity(a, c['artist']) 
                for a in entities['ARTIST']] + [0])
            for c in candidates
        ]
        
        title_scores = [
            max([self._levenshtein_similarity(t, c['title'])
                for t in entities['WORK_OF_ART']] + [0])
            for c in candidates
        ]
        
        total_scores = 0.6 * np.array(artist_scores) + 0.4 * np.array(title_scores)
        best_idx = np.argmax(total_scores)
        
        return (candidates[best_idx]['id'], total_scores[best_idx])

    def _jaccard_similarity(self, a, b):
        a_words = set(str(a).lower().split())
        b_words = set(str(b).lower().split())
        intersection = len(a_words & b_words)
        union = len(a_words | b_words)
        return intersection / union if union else 0

    def _levenshtein_similarity(self, a, b):
        return fuzz.ratio(str(a).lower(), str(b).lower()) / 100

    def batch_match(self, reviews):
        """Batch processing of reviews"""
        if not self.albums:
            raise ValueError("No albums loaded for matching")
            
        self._build_indices()
        
        # Batch detect languages
        languages = [detect(r.text) for r in reviews]
        
        # Process texts with appropriate NLP model
        docs = []
        for text, lang in zip([r.text for r in reviews], languages):
            if lang == 'pt':
                docs.append(self.nlp_pt(text))
            else:
                docs.append(self.nlp_en(text))
        
        # Process matches
        results = []
        for doc in docs:
            entities = self._extract_entities(doc)
            candidates = self._get_candidates(entities)
            match = self._score_candidates(candidates, entities)
            results.append(match)
            
        return results

    def _extract_entities(self, doc):
        """Entity extraction from processed document"""
        return {
            'ARTIST': [ent.text for ent in doc.ents 
                      if ent.label_ in ['PERSON', 'ORG', 'PER']],
            'WORK_OF_ART': [ent.text for ent in doc.ents 
                           if ent.label_ in ['WORK_OF_ART', 'OBRA']]
        }

def load_initial_data(sample_size=100):
    """Optimized data loading with batch processing and caching"""
    with app.app_context():
        try:
            # Clear existing data
            db.session.execute(db.delete(Review))
            db.session.execute(db.delete(Album))
            db.session.commit()
            print("🧹 Cleared existing data")

            # Bulk insert albums
            albums_df = pd.read_csv('data/albums.csv', index_col=0)
            albums_df = albums_df.head(sample_size)
            
            albums = []
            for _, row in albums_df.iterrows():
                albums.append(Album(
                    title=row['release_name'],
                    artist=row['artist_name'],
                    genre=row['primary_genres'],
                    year=pd.to_datetime(row['release_date']).year,
                    rating=row['avg_rating']
                ))
            
            db.session.bulk_save_objects(albums)
            db.session.commit()
            print(f"✅ Loaded {len(albums)} albums (bulk insert)")

            # Batch process reviews
            reviews_df = pd.read_csv('data/reviews.csv').sample(sample_size)
            reviews_df = reviews_df.dropna(subset=['Review'])
            
            # Preload albums once
            all_albums = Album.query.all()
            matcher = ReviewMatcher()
            matcher.albums = [{
                'id': a.id,
                'artist': a.artist.lower(),
                'title': a.title.lower()
            } for a in Album.query.all()]

            # Process in memory first
            review_objects = []
            batch_size = 50  # Adjust based on your RAM
            
            for idx, row in reviews_df.iterrows():
                try:
                    text = clean_text(str(row['Review']))
                    lang = detect(text)
                    
                    # Create review object
                    review = Review(
                        text=text,
                        rating=float(row['Rating']),
                        language=lang,
                        album_id=None,
                        matching_confidence=None
                    )
                    
                    # Batch matching
                    if idx % batch_size == 0 and idx > 0:
                        matched = matcher.batch_match(review_objects[-batch_size:])
                        for r, match in zip(review_objects[-batch_size:], matched):
                            r.album_id, r.matching_confidence = match
                    
                    review_objects.append(review)
                    
                except Exception as e:
                    print(f"⚠️ Error processing row {idx}: {str(e)}")
                    continue

            # Final batch match
            matched = matcher.batch_match(review_objects[-(len(review_objects)%batch_size):])
            for r, match in zip(review_objects[-(len(review_objects)%batch_size):], matched):
                r.album_id, r.matching_confidence = match

            # Bulk insert reviews
            db.session.bulk_save_objects(review_objects)
            db.session.commit()
            print(f"🚀 Loaded {len(review_objects)} reviews (bulk processed)")

        except Exception as e:
            db.session.rollback()
            print(f"🔥 Critical error: {str(e)}")

def format_result(review_id, score):
    review = db.session.get(Review, review_id)
    album = db.session.get(Album, review.album_id) if review.album_id else None
    return {
        'id': review_id,
        'preview': review.text[:100] + '...',
        'score': f"{score:.2f}",
        'album': f"{album.artist} - {album.title}" if album else None
    }

vector_search=VectorSearch()

def initialize_search():
    with app.app_context():
        reviews = Review.query.all()
        vector_search.build_index(reviews)

# Routes
@app.route('/')
def home():
    return redirect('/search')

@app.route('/search')
def search():
    query = request.args.get('q', '')
    method = request.args.get('method', 'cosine')
    
    if not query:
        return render_template('search.html', results=[])
    
    try:
        results = vector_search.search(query, method=method)
        formatted_results = [format_result(review_id, score) 
                           for review_id, score in results if score > 0]
    except Exception as e:
        print(f"Search error: {str(e)}")
        formatted_results = []
    
    return render_template('search.html',
        results=formatted_results,
        query=query,
        method=method
    )

@app.route('/review/<int:review_id>')
def review_detail(review_id):
    try:
        review = db.session.get(Review, review_id)
        print(f"Review confidence: {review.matching_confidence}")  # Debug
        if not review:
            return "Review not found", 404
        
        album = db.session.get(Album, review.album_id) if review.album_id else None
        return render_template('review_detail.html', review=review, album=album)
    except Exception as e:
        print(f"Error: {str(e)}")
        return "Error loading review", 500

if __name__ == '__main__':
    with app.app_context():
        # Completely reset database
        db.drop_all()
        db.create_all()
        print("Created fresh database tables")
        
        # Load initial data
        load_initial_data(sample_size=100)  # Start with small sample
        
        # Initialize search index
        initialize_search()
        
    app.run(debug=True)