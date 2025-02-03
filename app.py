from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np
from typing import List, Tuple
from collections import defaultdict
import re
from fuzzywuzzy import fuzz
from textblob import TextBlob
from langdetect import detect
import concurrent.futures
from sqlalchemy import Index, func
from time import time
import gc

# Initialize Flask and extensions
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///music.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_POOL_SIZE'] = 20
app.config['SQLALCHEMY_MAX_OVERFLOW'] = 5

# Configure caching
cache = Cache(config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 300
})
cache.init_app(app)
db = SQLAlchemy(app)

class Album(db.Model):
    __tablename__ = 'album'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    artist = db.Column(db.String(200), nullable=False)
    year = db.Column(db.Integer)
    genre = db.Column(db.String(100))
    rating = db.Column(db.Float)
    reviews = db.relationship('Review', backref='album', lazy='dynamic')

    __table_args__ = (
        Index('idx_album_search', 'title', 'artist'),
        Index('idx_album_filters', 'year', 'genre', 'rating'),
    )

class Review(db.Model):
    __tablename__ = 'review'
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Float)
    album_id = db.Column(db.Integer, db.ForeignKey('album.id'))
    matching_confidence = db.Column(db.Float)
    sentiment = db.Column(db.String(20), default='neutral')
    sentiment_score = db.Column(db.Float)
    language = db.Column(db.String(10), default='unknown')
    vector = db.relationship('ReviewVector', uselist=False, backref='review', lazy='joined')

    __table_args__ = (
        Index('idx_review_filters', 'language', 'rating', 'album_id'),
        Index('idx_review_search', 'text'),
        Index('idx_review_sentiment', 'sentiment', 'sentiment_score'),
    )

class ReviewVector(db.Model):
    __tablename__ = 'review_vector'
    id = db.Column(db.Integer, primary_key=True)
    review_id = db.Column(db.Integer, db.ForeignKey('review.id'), nullable=False)
    vector = db.Column(db.PickleType, nullable=False)

    __table_args__ = (
        Index('idx_vector_review', 'review_id'),
    )

class GazetterMatcher:
    def __init__(self):
        self.artist_gazetteer = defaultdict(set)
        self.album_gazetteer = defaultdict(set)
        self.artist_to_albums = defaultdict(list)
        self.normalized_artists = {}
        self.normalized_albums = {}
        
    @staticmethod
    def normalize_text(text):
        """Normalize text for matching"""
        return ' '.join(re.sub(r'[^\w\s]', '', str(text).lower()).split())
    
    def build_from_dataframe(self, df):
        """Build gazetteer from DataFrame"""
        for _, row in df.iterrows():
            artist = str(row['artist_name']).strip()
            album = str(row['release_name']).strip()
            
            if not artist or not album:
                continue
                
            norm_artist = self.normalize_text(artist)
            norm_album = self.normalize_text(album)
            
            if not norm_artist or not norm_album:
                continue
            
            self.normalized_artists[norm_artist] = artist
            self.normalized_albums[norm_album] = album
            
            # Add full names and tokens
            self.artist_gazetteer[norm_artist].add(norm_artist)
            self.album_gazetteer[norm_album].add(norm_album)
            
            for token in norm_artist.split():
                if len(token) > 2:
                    self.artist_gazetteer[token].add(norm_artist)
            
            for token in norm_album.split():
                if len(token) > 2:
                    self.album_gazetteer[token].add(norm_album)
            
            self.artist_to_albums[norm_artist].append(norm_album)

    def find_best_match(self, text, threshold=0.4):
        text_lower = text.lower()
        norm_text = self.normalize_text(text)
        words = set(norm_text.split())
        
        # Direct matching first (faster)
        for artist in self.normalized_artists:
            if artist in text_lower:
                for album in self.artist_to_albums[artist]:
                    if album in text_lower:
                        return self.normalized_artists[artist], self.normalized_albums[album], 1.0
        
        # Fuzzy matching if needed
        potential_artists = {artist for word in words if word in self.artist_gazetteer 
                           for artist in self.artist_gazetteer[word]}
        
        best_score = threshold
        best_match = (None, None, 0)
        
        for artist in potential_artists:
            artist_score = fuzz.token_set_ratio(text_lower, artist) / 100.0
            if artist_score < 0.4:
                continue
                
            for album in self.artist_to_albums[artist]:
                album_score = fuzz.token_set_ratio(text_lower, album) / 100.0
                combined_score = (0.7 * artist_score) + (0.3 * album_score)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = (self.normalized_artists[artist], 
                                self.normalized_albums[album], 
                                best_score)
        
        return best_match

class VectorSearch:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(min_df=5)
        self.index = None
        self.vectors = None
        self.id_map = {}
        self.is_initialized = False
        self.batch_size = 10000  # For batch processing

    def fit(self, documents):
        """Initialize and fit the vectorizer"""
        self.vectorizer.fit(documents)
        self.is_initialized = True

    def transform(self, documents):
        """Transform documents to vectors"""
        if not self.is_initialized:
            raise RuntimeError("Vectorizer not initialized")
        return self.vectorizer.transform(documents).toarray()

    def build_index(self, vectors: np.ndarray, review_ids: List[int]):
        """Build FAISS index from vectors"""
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.vectors = vectors  # Store for dice/jaccard
        normalized_vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
        self.index.add(normalized_vectors.astype('float32'))
        self.id_map = {i: rid for i, rid in enumerate(review_ids)}
        
    def cosine_search(self, query_vector: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """Perform cosine similarity search using FAISS"""
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)  # Convert to 2D array
            
        if query_vector.shape[1] != self.index.d:
            raise ValueError(f"Query vector dimension {query_vector.shape[1]} "
                           f"does not match index dimension {self.index.d}")
            
        try:
            normalized_query = query_vector / np.linalg.norm(query_vector)
            scores, indices = self.index.search(
                normalized_query.astype('float32'), 
                min(k, self.index.ntotal)
            )
            
            return [(self.id_map[idx], float(score)) 
                    for score, idx in zip(scores[0], indices[0])
                    if idx != -1]  # Filter invalid indices
                    
        except Exception as e:
            print(f"Search failed: {str(e)}")
            return []
    
    def search(self, query: str, method: str = 'cosine', k: int = 20) -> List[Tuple[int, float]]:
        if not self.is_initialized:
            raise RuntimeError("Vector search not initialized")
            
        try:
            # Convert to dense array once
            query_vector = self.vectorizer.transform([query]).toarray()[0]
            
            if method == 'cosine':
                return self.cosine_search(query_vector, k)
            
            # Process in batches
            scores = []
            query_sum = np.sum(np.abs(query_vector))
            
            for i in range(0, self.vectors.shape[0], self.batch_size):
                batch = self.vectors[i:i+self.batch_size]
                
                # Dense numpy operations
                dot_products = np.dot(batch, query_vector)
                vector_sums = np.sum(np.abs(batch), axis=1)
                
                if method == 'dice':
                    batch_scores = 2 * dot_products / (query_sum + vector_sums + 1e-8)
                else:  # jaccard  
                    batch_scores = dot_products / (query_sum + vector_sums - dot_products + 1e-8)
                
                # Filter near-zero scores
                nonzero_scores = [(i+idx, score) for idx, score in enumerate(batch_scores) if score > 1e-6]
                scores.extend(nonzero_scores)
                
            if not scores:
                return []
                
            # Sort and return top k
            scores.sort(key=lambda x: x[1], reverse=True)
            return [(self.id_map[idx], float(score)) for idx, score in scores[:k]]
            
        except Exception as e:
            print(f"Search failed: {str(e)}")
            return []

def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.1:
        return 'positive', polarity
    elif polarity < -0.1:
        return 'negative', polarity
    return 'neutral', polarity

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'

def process_reviews_batch(batch_df, matcher, album_lookup):
    reviews = []
    processed_count = 0
    total_in_batch = len(batch_df)
    
    for idx, row in batch_df.iterrows():
        processed_count += 1
        if processed_count % 100 == 0:  # Log every 100 reviews
            print(f"Processing review {processed_count}/{total_in_batch} in current batch...")
            
        if pd.isna(row['Review']) or pd.isna(row['Rating']):
            continue
            
        text = str(row['Review']).strip()
        if not text:
            continue
            
        artist, album, confidence = matcher.find_best_match(text)
        album_id = album_lookup.get((artist, album)) if artist and album else None
        
        if artist and album:  # Log successful matches
            print(f"Found match: {artist} - {album} (confidence: {confidence:.2f})")
        
        sentiment, sentiment_score = analyze_sentiment(text)
        reviews.append(Review(
            text=text,
            rating=float(row['Rating']),
            album_id=album_id,
            matching_confidence=confidence,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            language=detect_language(text)
        ))

    print(f"✅ Batch complete: processed {len(reviews)} valid reviews out of {total_in_batch} total")
    return reviews

def initialize_database(sample_size=16000):
    try:
        # Clear existing data
        db.session.execute(db.delete(Review))
        db.session.execute(db.delete(Album))
        db.session.commit()
        print("🧹 Cleared existing data")
        
        # Process albums
        albums_df = pd.read_csv('data/albums.csv')
        matcher = GazetterMatcher()
        matcher.build_from_dataframe(albums_df)
        
        album_lookup = {}
        for _, row in albums_df.iterrows():
            album = Album(
                title=row['release_name'],
                artist=row['artist_name'],
                genre=row['primary_genres'],
                year=pd.to_datetime(row['release_date']).year,
                rating=row['avg_rating']
            )
            db.session.add(album)
            db.session.flush()
            album_lookup[(row['artist_name'], row['release_name'])] = album.id
        
        db.session.commit()
        print(f"✅ Loaded {len(album_lookup)} albums")
        
        # Process reviews in chunks
        reviews_df = pd.read_csv('data/reviews.csv')
        if sample_size:
            reviews_df = reviews_df.sample(sample_size)
        
        total_reviews = 0
        chunk_size = 5000

        print(f"\n🚀 Starting review processing with chunk size: {chunk_size}")
        print(f"Total reviews to process: {len(reviews_df)}")

        for i in range(0, len(reviews_df), chunk_size):
            print(f"\n📦 Processing batch {(i//chunk_size)+1}/{(len(reviews_df)//chunk_size)+1}")
            chunk = reviews_df.iloc[i:i+chunk_size]
            reviews = process_reviews_batch(chunk, matcher, album_lookup)
            
            if reviews:
                print("💾 Saving batch to database...")
                db.session.bulk_save_objects(reviews)
                db.session.commit()
                total_reviews += len(reviews)
                print(f"✨ Total reviews processed so far: {total_reviews}/{len(reviews_df)}")
        
        return matcher
        
    except Exception as e:
        db.session.rollback()
        print(f"🔥 Critical error: {str(e)}")
        raise

vector_search = VectorSearch()

@cache.memoize(timeout=300)
def get_filtered_reviews(language=None, year_from=None, year_to=None, 
                        rating_from=None, rating_to=None, genre=None):
    query = Review.query.join(Album)
    
    if language:
        query = query.filter(Review.language == language)
    if rating_from:
        query = query.filter(Review.rating >= rating_from)
    if rating_to:
        query = query.filter(Review.rating <= rating_to)
    if year_from:
        query = query.filter(Album.year >= year_from)
    if year_to:
        query = query.filter(Album.year <= year_to)
    if genre:
        query = query.filter(Album.genre == genre)
        
    return query.all()
@app.route('/')
@app.route('/search')
def search():
    query = request.args.get('q', '')
    method = request.args.get('method', 'cosine')
    language = request.args.get('language')
    year_from = request.args.get('year_from', type=int)
    year_to = request.args.get('year_to', type=int)
    rating_from = request.args.get('rating_from', type=float)
    rating_to = request.args.get('rating_to', type=float)
    genre = request.args.get('genre')
    page = request.args.get('page', 1, type=int)
    per_page = 20

    results = []
    if query:
        filtered_reviews = get_filtered_reviews(
            language, year_from, year_to, rating_from, rating_to, genre)
        review_ids = {r.id for r in filtered_reviews}
        
        similarities = vector_search.search(query, method=method, k=len(review_ids))
        matched_results = [
            (rid, score) for rid, score in similarities
            if rid in review_ids
        ]
        
        # Paginate results
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_results = matched_results[start_idx:end_idx]
        
        for review_id, score in page_results:
            review = Review.query.get(review_id)
            if review:
                album = review.album
                results.append({
                    'id': review_id,
                    'preview': review.text[:200] + '...',
                    'score': f"{score:.2f}",
                    'album': f"{album.artist} - {album.title}" if album else "Unknown Album",
                    'relevance': f"{(score * 100):.1f}%"
                })

    # Get counts and metadata
    album_count = db.session.query(func.count(Album.id)).scalar()
    review_count = db.session.query(func.count(Review.id)).scalar()
    languages = db.session.query(Review.language).distinct().all()
    genres = db.session.query(Album.genre).distinct().all()

    return render_template(
        'search.html',
        results=results,
        query=query,
        method=method,
        album_count=album_count,
        review_count=review_count,
        languages=[l[0] for l in languages if l[0]],
        genres=[g[0] for g in genres if g[0]],
        language=language,
        year_from=year_from,
        year_to=year_to,
        rating_from=rating_from,
        rating_to=rating_to,
        genre=genre,
        page=page
    )

@app.route('/review/<int:review_id>')
@cache.memoize(timeout=300)
def review_detail(review_id):
    try:
        review = Review.query.options(
            db.joinedload(Review.album)
        ).get_or_404(review_id)
        
        sentiment_map = {
            'positive': 'Generally favorable review',
            'negative': 'Critical or negative feedback',
            'neutral': 'Balanced or factual commentary'
        }
        
        return render_template('review_detail.html',
                             review=review,
                             album=review.album,
                             sentiment_map=sentiment_map)
    except Exception as e:
        print(f"Error loading review {review_id}: {str(e)}")
        return "Error loading review", 500

@app.route('/visualizations')
@cache.memoize(timeout=600)
def visualizations():
    # Efficient aggregation queries
    ratings = db.session.query(Review.rating).filter(Review.rating.isnot(None)).all()
    rating_dist = np.histogram([r[0] for r in ratings], bins=10)[0].tolist()
    
    genre_counts = dict(
        db.session.query(Album.genre, func.count(Review.id))
        .join(Review)
        .filter(Album.genre.isnot(None))
        .group_by(Album.genre)
        .all()
    )
    
    year_counts = dict(
        db.session.query(Album.year, func.count(Review.id))
        .join(Review)
        .filter(Album.year.isnot(None))
        .group_by(Album.year)
        .all()
    )
    
    return render_template('visualization.html',
                         rating_dist=rating_dist,
                         genre_counts=genre_counts,
                         year_counts=dict(sorted(year_counts.items())))

@app.route('/about')
@cache.cached(timeout=3600)
def about():
    return render_template('about.html')

def initialize_search():
    """Initialize the vector search engine"""
    global vector_search
    
    try:
        reviews = Review.query.all()
        if not reviews:
            raise ValueError("No reviews found in database")
            
        documents = [review.text for review in reviews]
        review_ids = [review.id for review in reviews]
        
        vector_search = VectorSearch()
        vector_search.fit(documents)  # Changed from fit_vectorizer
        vectors = vector_search.transform(documents)
        vector_search.build_index(vectors, review_ids)
        
        print("Search engine initialized successfully")
        return vector_search
        
    except Exception as e:
        print(f"Failed to initialize search: {str(e)}")
        raise

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if Album.query.count() == 0:
            print("Database empty, initializing with sample dataset...")
            matcher = initialize_database(sample_size=1000)
            initialize_search()
            print("Database initialized")
        else:
            print("Database already populated, initializing search...")
            initialize_search()
        print('Search initialized')
    app.run(debug=True)