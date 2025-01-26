from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from textblob import TextBlob
import spacy
from collections import Counter
import en_core_web_sm
from flask import jsonify
import numpy as np
from difflib import SequenceMatcher
from fuzzywuzzy import process, fuzz
import time
import re
import musicbrainzngs
from langdetect import detect
import pt_core_news_sm  # Portuguese model

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///music.db'
nlp = spacy.load('en_core_web_sm')
db = SQLAlchemy(app)

def clean_text(text):
    """Basic text cleaning for reviews"""
    text = str(text)
    # Remove markdown/images links
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Remove special characters except basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.,!?\'"-]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

musicbrainzngs.set_useragent("RYM-Analyzer", "0.1", contact="k.miranowski@gmail.com")


class OptimizedReviewMatcher:
    def __init__(self):
        self.albums = self._preload_albums()
        self.artist_index = self._build_artist_index()
        self.title_index = self._build_title_index()
    
    def jaccard_similarity(self, a, b):
        set_a = set(str(a).lower().split())
        set_b = set(str(b).lower().split())
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union else 0

    def levenshtein_similarity(self, a, b):
        return fuzz.ratio(str(a).lower(), str(b).lower()) / 100
    
    def _preload_albums(self):
        """Safely load albums with null handling"""
        preloaded = []
        for album in Album.query.all():
            try:
                artist = str(album.artist) if album.artist else ""
                title = str(album.title) if album.title else ""
                
                preloaded.append({
                    'id': album.id,
                    'artist': artist.lower().strip(),
                    'title': title.lower().strip(),
                    'artist_tokens': set(artist.lower().split()),
                    'title_tokens': set(title.lower().split())
                })
            except Exception as e:
                print(f"Error processing album {album.id}: {str(e)}")
        return preloaded

    def _preload_albums(self):
        """Load all albums once with preprocessed data"""
        return [
            {
                'id': album.id,
                'artist': album.artist.lower().strip(),
                'title': album.title.lower().strip(),
                'artist_tokens': set(album.artist.lower().split()),
                'title_tokens': set(album.title.lower().split())
            }
            for album in Album.query.all()
        ]
    
    def _build_artist_index(self):
        """Create quick lookup index for artists"""
        index = {}
        for album in self.albums:
            for token in album['artist_tokens']:
                index.setdefault(token, []).append(album)
        return index
    
    def _build_title_index(self):
        """Create quick lookup index for title words"""
        index = {}
        for album in self.albums:
            for token in album['title_tokens']:
                index.setdefault(token, []).append(album)
        return index

    def _get_candidates(self, entities):
        """Safe candidate selection with validation"""
        candidates = {}
        
        # Artist token search
        for artist in entities['ARTIST']:
            for token in str(artist).lower().split():
                for album in self.artist_index.get(token, []):
                    if isinstance(album, dict):  # Validate type
                        candidates[album['id']] = album
        
        # Title token search
        for title in entities['WORK_OF_ART']:
            for token in str(title).lower().split():
                for album in self.title_index.get(token, []):
                    if isinstance(album, dict):  # Validate type
                        candidates[album['id']] = album
        
        return list(candidates.values()) or self.albums

    def match_review_to_album(self, review_text):
        """Debuggable matching process"""
        try:
            # Extract entities
            doc = nlp(review_text)
            entities = {
                'ARTIST': [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG']],
                'WORK_OF_ART': [ent.text for ent in doc.ents if ent.label_ == 'WORK_OF_ART']
            }
            print(f"\n[Debug] Review Text: {review_text[:100]}...")  # Truncated for readability
            print(f"[Debug] Entities Found - Artists: {entities['ARTIST']}, Titles: {entities['WORK_OF_ART']}")

            # Get candidates
            candidates = self._get_candidates(entities)
            print(f"[Debug] Number of Candidates: {len(candidates)}")

            if not candidates:
                print("[Debug] No candidates found!")
                return None, 0

            best_match = None
            max_score = 0

            for album in candidates:
                # Calculate scores
                artist_score = max(
                    [self.jaccard_similarity(a, album.get('artist', '')) 
                     for a in entities['ARTIST']] + [0]
                )
                title_score = max(
                    [self.levenshtein_similarity(t, album.get('title', ''))
                     for t in entities['WORK_OF_ART']] + [0]
                )
                total_score = 0.6 * artist_score + 0.4 * title_score
                
                print(f"[Debug] Album {album['id']} ({album['artist']} - {album['title']})")
                print(f"  Artist Score: {artist_score:.2f}, Title Score: {title_score:.2f}, Total: {total_score:.2f}")

                if total_score > max_score:
                    max_score = total_score
                    best_match = album

            if best_match:
                print(f"[Debug] Best Match: {best_match['artist']} - {best_match['title']} (Score: {max_score:.2f})")
            else:
                print("[Debug] No viable matches found")

            return (best_match['id'], max_score) if best_match else (None, 0)

        except Exception as e:
            print(f"[Error] Matching failed: {str(e)}")
            return None, 0

class ValidatedReviewMatcher(OptimizedReviewMatcher):
    def __init__(self):
        super().__init__()
        self.cache = {}  # Simple cache to avoid duplicate API calls
        
    def validate_pair(self, artist, title):
        """Check if artist-title pair exists in MusicBrainz"""
        cache_key = f"{artist}||{title}".lower()
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            result = musicbrainzngs.search_releases(
                artist=artist,
                release=title,
                limit=1
            )
            valid = bool(result.get("release-list", []))
            self.cache[cache_key] = valid
            return valid
        except Exception as e:
            print(f"Validation error for {artist} - {title}: {str(e)}")
            return False

    def match_review_to_album(self, review_text):
        """Enhanced matching with API validation"""
        album_id, confidence = super().match_review_to_album(review_text)
        
        if album_id:
            album = next(a for a in self.albums if a['id'] == album_id)
            is_valid = self.validate_pair(album['artist'], album['title'])
            
            # Penalize unvalidated matches
            if not is_valid:
                confidence *= 0.7  # Reduce confidence by 30%
                
        return album_id, confidence

class Album(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    artist = db.Column(db.String(200), nullable=False)
    genre = db.Column(db.String(100))
    year = db.Column(db.Integer)
    rating = db.Column(db.Float)
    position = db.Column(db.Integer)
    release_type = db.Column(db.String(100))
    secondary_genres = db.Column(db.String(200))
    descriptors = db.Column(db.Text)
    rating_count = db.Column(db.Integer)
    review_count = db.Column(db.Integer)

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Float)
    album_id = db.Column(db.Integer, db.ForeignKey('album.id'))
    sentiment_polarity = db.Column(db.Float)
    sentiment_subjectivity = db.Column(db.Float)
    entities = db.Column(db.JSON)
    pos_tags = db.Column(db.JSON)
    matching_confidence = db.Column(db.Float)

class TextAnalyzer:
    def __init__(self):
        self.nlp = en_core_web_sm.load()
    
    def analyze_sentiment(self, text):
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def extract_entities(self, text):
        doc = self.nlp(text)
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        return entities
    
    def morphological_analysis(self, text):
        doc = self.nlp(text)
        return {
            'tokens': len(doc),
            'pos_tags': Counter([token.pos_ for token in doc]),
            'lemmas': [token.lemma_ for token in doc],
        }

def load_initial_data(sample_size=100, reload_all=False):
    with app.app_context():
        try:
            if reload_all:
                # Only for initial setup/reset
                db.drop_all()
                db.create_all()
                print("Recreated all tables")

            db.session.query(Review).delete()
            db.session.query(Album).delete()
            db.session.commit()
            
            if not Album.query.first():
                print("Loading initial data...")
                albums_df = pd.read_csv('data/albums.csv', index_col=0)
                albums_df = albums_df.head(sample_size)
                print(f"Loading {len(albums_df)} albums")
                for _, row in albums_df.iterrows():
                    album = Album(
                        title=row['release_name'],
                        artist=row['artist_name'],
                        genre=row['primary_genres'],
                        year=pd.to_datetime(row['release_date']).year,
                        rating=row['avg_rating'],
                        position=row['position'],
                        release_type=row['release_type'],
                        secondary_genres=row['secondary_genres'],
                        descriptors=row['descriptors'],
                        rating_count=row['rating_count'],
                        review_count=row['review_count']
                    )
                    db.session.add(album)
                db.session.commit()
            
                reviews_df = pd.read_csv('data/reviews.csv').sample(sample_size)
                reviews_df = reviews_df.dropna(subset=['Review'])
                print(f"Loading {len(reviews_df)} reviews")
                
                matcher = ValidatedReviewMatcher()
                BATCH_SIZE = 100  # Process in chunks to allow partial saving
                total_reviews = len(reviews_df)
                
                for batch_start in range(0, total_reviews, BATCH_SIZE):
                    batch = reviews_df.iloc[batch_start:batch_start+BATCH_SIZE]
                    
                    for _, row in batch.iterrows():
                        review_text = clean_text(str(row['Review']))
                        album_id, confidence = matcher.match_review_to_album(review_text)
                        
                        review = Review(
                            text=review_text,
                            rating=float(row['Rating']),
                            album_id=album_id,
                            matching_confidence=confidence
                        )
                        db.session.add(review)
                    
                    try:
                        db.session.commit()
                        print(f"Processed {batch_start+BATCH_SIZE}/{total_reviews}")
                    except Exception as e:
                        db.session.rollback()
                        print(f"Error at batch {batch_start}: {str(e)}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            db.session.rollback()

def analyze_reviews_batch():
    analyzer = TextAnalyzer()
    reviews = Review.query.filter(Review.sentiment_polarity.is_(None)).limit(100).all()
    
    for review in reviews:
        sentiment = analyzer.analyze_sentiment(review.text)
        entities = analyzer.extract_entities(review.text)
        morphology = analyzer.morphological_analysis(review.text)
        
        review.sentiment_polarity = sentiment['polarity']
        review.sentiment_subjectivity = sentiment['subjectivity']
        review.entities = entities
        review.pos_tags = morphology['pos_tags']
    
    db.session.commit()

def add_text_analysis_routes(app, db, Review):
    analyzer = TextAnalyzer()
    
    @app.route('/api/analyze/<int:review_id>')
    def analyze_review(review_id):
        review = Review.query.get_or_404(review_id)
        
        sentiment = analyzer.analyze_sentiment(review.text)
        entities = analyzer.extract_entities(review.text)
        morphology = analyzer.morphological_analysis(review.text)
        
        return jsonify({
            'sentiment': sentiment,
            'entities': entities,
            'morphology': morphology
        })
    
    @app.route('/api/sentiment/stats')
    def sentiment_stats():
        reviews = Review.query.limit(1000).all()  # Sample for performance
        sentiments = [analyzer.analyze_sentiment(r.text)['polarity'] for r in reviews]
        
        return jsonify({
            'mean': np.mean(sentiments),
            'std': np.std(sentiments),
            'distribution': np.histogram(sentiments, bins=10)[0].tolist()
        })

@app.route('/')
def index():
    album_count = Album.query.count()
    review_count = Review.query.count()
    return render_template('index.html', 
                         album_count=album_count,
                         review_count=review_count)
@app.route('/test_analysis')
def test_analysis():
    review = Review.query.first()
    if not review:
        return jsonify({"error": "No reviews found"})
        
    analyzer = TextAnalyzer()
    results = {
        'review_text': review.text[:200] + '...',
        'sentiment': analyzer.analyze_sentiment(review.text),
        'entities': analyzer.extract_entities(review.text),
        'morphology': analyzer.morphological_analysis(review.text)
    }
    
    return jsonify(results)

@app.route('/api/sentiment/stats')
def sentiment_stats():
    analyzer = TextAnalyzer()
    reviews = Review.query.limit(1000).all()
    sentiments = [analyzer.analyze_sentiment(r.text)['polarity'] for r in reviews]
    
    return jsonify({
        'mean': np.mean(sentiments),
        'std': np.std(sentiments),
        'distribution': np.histogram(sentiments, bins=10)[0].tolist()
    })

@app.route('/visualization')
def visualization():
    return render_template('visualization.html')   

@app.route('/api/genre_stats')
def genre_stats():
    albums = Album.query.all()
    genres = Counter([a.genre for a in albums if a.genre])
    return jsonify({
        'labels': list(genres.keys()),
        'values': list(genres.values())
    })

@app.route('/api/rating_sentiment')
def rating_sentiment():
    analyzer = TextAnalyzer()
    reviews = Review.query.limit(500).all()
    data = [{
        'rating': r.rating,
        'sentiment': analyzer.analyze_sentiment(r.text)['polarity']
    } for r in reviews]
    return jsonify(data)

@app.route('/test_validation/<int:review_id>')
def test_validation(review_id):
    review = Review.query.get_or_404(review_id)
    matcher = ValidatedReviewMatcher()
    album_id, confidence = matcher.match_review_to_album(review.text)
    
    result = {
        'review_id': review_id,
        'matched_album': Album.query.get(album_id).title if album_id else None,
        'validated': bool(matcher.validate_pair(
            Album.query.get(album_id).artist if album_id else "",
            Album.query.get(album_id).title if album_id else ""
        )),
        'confidence': confidence
    }
    return jsonify(result)          

if __name__ == '__main__':
    with app.app_context():
        db.drop_all()
        db.create_all()
        load_initial_data(reload_all=True)
    app.run(debug=True)