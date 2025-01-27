from flask import Flask, render_template, request, jsonify, redirect
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
import pandas as pd
import numpy as np
from collections import defaultdict
import re
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import pickle

# Initialize Flask and extensions
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///music.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
cache.init_app(app)

class GazetterMatcher:
    def __init__(self):
        self.artist_gazetteer = defaultdict(set)
        self.album_gazetteer = defaultdict(set)
        self.artist_to_albums = defaultdict(list)
        self.normalized_artists = {}
        self.normalized_albums = {}
        
    def normalize_text(self, text):
        """Normalize text for matching"""
        text = re.sub(r'[^\w\s]', '', str(text).lower())
        return ' '.join(text.split())
    
    def build_from_dataframe(self, df, artist_col='artist_name', album_col='release_name'):
        """Build gazetteer from DataFrame"""
        for _, row in df.iterrows():
            artist = str(row[artist_col]).strip()
            album = str(row[album_col]).strip()
            
            # Skip empty entries
            if not artist or not album:
                continue
                
            norm_artist = self.normalize_text(artist)
            norm_album = self.normalize_text(album)
            
            # Skip if normalized text is empty
            if not norm_artist or not norm_album:
                continue
            
            self.normalized_artists[norm_artist] = artist
            self.normalized_albums[norm_album] = album
            
            # Add full names
            self.artist_gazetteer[norm_artist].add(norm_artist)
            self.album_gazetteer[norm_album].add(norm_album)
            
            # Add tokens
            for token in norm_artist.split():
                if len(token) > 2:
                    self.artist_gazetteer[token].add(norm_artist)
            
            for token in norm_album.split():
                if len(token) > 2:
                    self.album_gazetteer[token].add(norm_album)
            
            self.artist_to_albums[norm_artist].append(norm_album)

    def find_best_match(self, text, threshold=0.4):
        potential_artists, potential_albums = self.extract_entities(text)
        text_lower = text.lower()
        
        best_score = threshold
        best_artist = None
        best_album = None
        
        # Direct artist/album mention check
        for artist in potential_artists:
            if artist.lower() in text_lower:  # Direct artist mention
                for album in self.artist_to_albums[artist]:
                    if album.lower() in text_lower:  # Direct album mention
                        return self.normalized_artists[artist], self.normalized_albums[album], 1.0
        
        # Fuzzy matching if no direct mention
        for artist in potential_artists:
            artist_score = fuzz.token_set_ratio(text_lower, artist) / 100.0
            if artist_score < 0.4:
                continue
                
            for album in self.artist_to_albums[artist]:
                album_score = fuzz.token_set_ratio(text_lower, album) / 100.0
                combined_score = (0.7 * artist_score) + (0.3 * album_score)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_artist = self.normalized_artists[artist]
                    best_album = self.normalized_albums[album]
        
        return best_artist, best_album, best_score
    
    def extract_entities(self, text):
        """Extract potential artist and album entities from text"""
        norm_text = self.normalize_text(text)
        words = set(norm_text.split())
        
        potential_artists = set()
        potential_albums = set()
        
        for word in words:
            if word in self.artist_gazetteer:
                potential_artists.update(self.artist_gazetteer[word])
            if word in self.album_gazetteer:
                potential_albums.update(self.album_gazetteer[word])
        
        return potential_artists, potential_albums

class Album(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    artist = db.Column(db.String(200), nullable=False)
    year = db.Column(db.Integer)
    genre = db.Column(db.String(100))
    rating = db.Column(db.Float)
    reviews = db.relationship('Review', backref='album', lazy=True)

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Float)
    album_id = db.Column(db.Integer, db.ForeignKey('album.id'))
    matching_confidence = db.Column(db.Float)

class VectorSearch:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=1,
            max_df=0.98,
            ngram_range=(1, 3),
            token_pattern=r'(?u)\b\w+\b'
        )
        self.review_vectors = None
        self.review_ids = None
    
    def build_index(self, reviews):
        """Build search index from reviews"""
        valid_reviews = [r for r in reviews if r.text and len(r.text.strip()) > 0]
        if not valid_reviews:
            raise ValueError("No valid text data found in reviews.")
        
        texts = []
        self.review_ids = []
        
        for r in valid_reviews:
            # Get album info if available
            album = Album.query.get(r.album_id) if r.album_id else None
            if album:
                text = f"{r.text} {album.title} {album.title} {album.artist} {album.artist}"
            else:
                text = r.text
                
            texts.append(text)
            self.review_ids.append(r.id)
        
        try:
            self.review_vectors = self.vectorizer.fit_transform(texts)
            print(f"✅ Built search index with {len(valid_reviews)} reviews")
        except Exception as e:
            print(f"Error building search index: {str(e)}")
            raise

def initialize_database(sample_size=1000):
    try:
        db.session.execute(db.delete(Review))
        db.session.execute(db.delete(Album))
        db.session.commit()
        print("🧹 Cleared existing data")
        
        albums_df = pd.read_csv('data/albums.csv')
        if sample_size:
            albums_df = albums_df.head(sample_size)
        
        matcher = GazetterMatcher()
        matcher.build_from_dataframe(albums_df)
        
        # Insert albums and create lookup
        albums = []
        album_lookup = {}
        
        for idx, row in albums_df.iterrows():
            album = Album(
                title=row['release_name'],
                artist=row['artist_name'],
                genre=row['primary_genres'],
                year=pd.to_datetime(row['release_date']).year,
                rating=row['avg_rating']
            )
            db.session.add(album)
            db.session.flush()  # Get ID
            album_lookup[(row['artist_name'], row['release_name'])] = album.id
        
        db.session.commit()
        print(f"✅ Loaded {len(albums)} albums")
        
        reviews_df = pd.read_csv('data/reviews.csv')
        if sample_size:
            reviews_df = reviews_df.sample(sample_size)
        
        batch_size = 100
        total_reviews = 0
        
        for i in range(0, len(reviews_df), batch_size):
            batch = reviews_df.iloc[i:i+batch_size]
            reviews = []
            
            for _, row in batch.iterrows():
                if pd.isna(row['Review']) or pd.isna(row['Rating']):
                    continue
                
                text = str(row['Review']).strip()
                if not text:
                    continue
                
                artist, album, confidence = matcher.find_best_match(text)
                album_id = None
                
                if artist and album:
                    album_id = album_lookup.get((artist, album))
                    print(f"Matched artist: {artist}, album: {album}, confidence: {confidence}")
                
                review = Review(
                    text=text,
                    rating=float(row['Rating']),
                    album_id=album_id,
                    matching_confidence=confidence
                )
                reviews.append(review)
            
            db.session.bulk_save_objects(reviews)
            db.session.commit()
            
            total_reviews += len(reviews)
            print(f"Processed {total_reviews} reviews")
        
        total_matched = Review.query.filter(Review.album_id.isnot(None)).count()
        print(f"✅ Loaded {total_reviews} reviews")
        print(f"Total reviews with matched albums: {total_matched}")
        
        return matcher
        
    except Exception as e:
        db.session.rollback()
        print(f"🔥 Critical error: {str(e)}")
        raise

vector_search = VectorSearch()

def initialize_search():
    """Initialize search index"""
    try:
        reviews = Review.query.all()
        if not reviews:
            print("No reviews found in database")
            return
        vector_search.build_index(reviews)
    except Exception as e:
        print(f"Error initializing search: {str(e)}")
        raise


@app.route('/review/<int:review_id>')
def review_detail(review_id):
    try:
        review = Review.query.get_or_404(review_id)
        album = Album.query.get(review.album_id) if review.album_id else None
        return render_template('review_detail.html', review=review, album=album, entities=None)
    except Exception as e:
        print(f"Error loading review {review_id}: {str(e)}")
        return "Error loading review", 500

@app.route('/')
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('q', '')
    method = request.args.get('method', 'cosine')
    
    album_count = Album.query.count()
    review_count = Review.query.count()
    results = []

    # Perform search using the specified method
    if vector_search.review_vectors is not None:
        query_vector = vector_search.vectorizer.transform([query])
        similarities = vector_search.review_vectors.dot(query_vector.T).toarray().flatten()
        
        # Sort based on similarity
        top_indices = np.argsort(similarities)[-10:][::-1]
        for idx in top_indices:
            if similarities[idx] > 0:
                review_id = vector_search.review_ids[idx]
                review = Review.query.get(review_id)
                if review:
                    album = Album.query.get(review.album_id) if review.album_id else None
                    results.append({
                        'id': review_id,
                        'preview': review.text[:200] + '...',
                        'score': f"{similarities[idx]:.2f}",
                        'album': f"{album.artist} - {album.title}" if album else "Unknown Album",
                        'relevance': f"{(similarities[idx] * 100):.1f}%"
                    })

    return render_template('search.html', results=results, query=query, method=method, album_count=album_count, review_count=review_count)


def format_result(review_id, score):
    review = Review.query.get(review_id)
    if not review:
        return None
    
    album = Album.query.get(review.album_id) if review.album_id else None
    return {
        'id': review_id,
        'preview': review.text[:200] + '...' if len(review.text) > 200 else review.text,
        'score': f"{score:.2f}",
        'album': f"{album.artist} - {album.title}" if album else "Unknown Album"
    }

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("Created fresh database tables")
        matcher = initialize_database(sample_size=1000)
        initialize_search()
        
    app.run(debug=True)