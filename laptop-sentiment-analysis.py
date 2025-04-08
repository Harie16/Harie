import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import spacy
from textblob import TextBlob

# For reproducibility
np.random.seed(42)

# Ensure necessary NLTK resources are downloaded
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model for aspect-based sentiment analysis
nlp = spacy.load('en_core_web_sm')

# Step 1: Data Preparation
# Assuming the data is saved as a TSV file - if not, adjust the code accordingly
def prepare_data(file_path):
    """Read and prepare the dataset for analysis."""
    try:
        # For CSV with tab delimiter
        df = pd.read_csv(file_path, sep='\t')
    except:
        # If that fails, try comma delimiter
        df = pd.read_csv(file_path)
    
    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()
    
    # Rename columns if needed for clarity
    column_mapping = {
        'product': 'product',
        'title': 'title',
        'date': 'date',
        'rating': 'rating',
        'body': 'review_text'
    }
    
    # Apply column renaming where columns exist
    existing_columns = set(df.columns) & set(column_mapping.keys())
    rename_dict = {col: column_mapping[col] for col in existing_columns}
    df = df.rename(columns=rename_dict)
    
    # Convert rating to numeric if it's not already
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    
    # Combine all text fields for text analysis
    text_columns = ['title', 'review_text'] if 'review_text' in df.columns else ['title']
    df['full_text'] = df[text_columns].apply(lambda x: ' '.join(str(val) for val in x if pd.notna(val)), axis=1)
    
    return df

# Step 2: Basic Statistics and Distributions
def generate_basic_statistics(df):
    """Generate basic statistics about the dataset."""
    stats = {}
    
    # Count of reviews
    stats['total_reviews'] = len(df)
    
    # Average rating
    if 'rating' in df.columns:
        stats['avg_rating'] = df['rating'].mean()
        stats['rating_distribution'] = df['rating'].value_counts().sort_index().to_dict()
    
    # Count by product
    if 'product' in df.columns:
        stats['product_counts'] = df['product'].value_counts().to_dict()
    
    return stats

# Step 3: Sentiment Analysis
def analyze_sentiment(df):
    """Analyze sentiment of reviews using VADER and TextBlob."""
    # Initialize sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    
    # VADER sentiment analysis
    df['vader_scores'] = df['full_text'].apply(lambda x: sid.polarity_scores(str(x)))
    df['vader_compound'] = df['vader_scores'].apply(lambda x: x['compound'])
    df['vader_pos'] = df['vader_scores'].apply(lambda x: x['pos'])
    df['vader_neu'] = df['vader_scores'].apply(lambda x: x['neu'])
    df['vader_neg'] = df['vader_scores'].apply(lambda x: x['neg'])
    
    # TextBlob sentiment analysis
    df['textblob_polarity'] = df['full_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['textblob_subjectivity'] = df['full_text'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)
    
    # Categorize sentiment
    df['sentiment_category'] = pd.cut(
        df['vader_compound'],
        bins=[-1, -0.5, 0.5, 1],
        labels=['Negative', 'Neutral', 'Positive']
    )
    
    return df

# Step 4: Emotion Analysis
def analyze_emotions(df):
    """Analyze emotions in reviews using lexicon-based approach."""
    # Emotion lexicons
    emotion_lexicon = {
        'joy': ['happy', 'joy', 'love', 'glad', 'pleased', 'delighted', 'satisfied', 'excellent', 'great', 'awesome'],
        'anger': ['angry', 'annoyed', 'frustrat', 'irritat', 'upset', 'hate', 'furious'],
        'sadness': ['sad', 'disappoint', 'regret', 'unfortunate', 'unhappy', 'sorry'],
        'fear': ['fear', 'afraid', 'worry', 'concern', 'anxious', 'nervous'],
        'surprise': ['surprise', 'shock', 'amazed', 'astonished', 'unexpected'],
        'disgust': ['disgust', 'repulsed', 'awful', 'horrible', 'terrible']
    }
    
    # Function to count emotion words
    def count_emotions(text):
        text = str(text).lower()
        emotions = {emotion: 0 for emotion in emotion_lexicon}
        
        for emotion, words in emotion_lexicon.items():
            for word in words:
                emotions[emotion] += len(re.findall(r'\b' + word + r'[a-z]*\b', text))
        
        return emotions
    
    # Apply emotion analysis
    emotions_df = df['full_text'].apply(count_emotions).apply(pd.Series)
    df = pd.concat([df, emotions_df], axis=1)
    
    # Add dominant emotion column
    emotion_columns = list(emotion_lexicon.keys())
    df['dominant_emotion'] = df[emotion_columns].idxmax(axis=1)
    
    # For cases where all emotions are 0, set as 'neutral'
    mask = (df[emotion_columns].sum(axis=1) == 0)
    df.loc[mask, 'dominant_emotion'] = 'neutral'
    
    return df

# Step 5: Aspect-Based Sentiment Analysis with Transformers
def analyze_aspects(df):
    """Extract aspects (features) mentioned in reviews and analyze sentiment toward each aspect using both 
    rule-based and transformer-based approaches."""
    # Common laptop aspects with expanded keywords
    laptop_aspects = {
        'performance': ['performance', 'speed', 'fast', 'slow', 'lag', 'processor', 'cpu', 'core', 'responsive', 'snappy', 'freeze', 'hanging', 'loading', 'boot', 'startup'],
        'battery': ['battery', 'charge', 'power', 'hours', 'life', 'drain', 'last', 'lasting', 'charger', 'adapter', 'outlet', 'plug', 'energy', 'consumption'],
        'display': ['screen', 'display', 'resolution', 'retina', 'bezel', 'bright', 'color', 'contrast', 'graphic', 'hdr', 'pixel', 'nit', 'refresh', 'glossy', 'matte'],
        'design': ['design', 'light', 'weight', 'thin', 'sleek', 'portable', 'aluminum', 'metal', 'sturdy', 'premium', 'build', 'aesthetic', 'look', 'appearance'],
        'keyboard': ['keyboard', 'type', 'keys', 'trackpad', 'touch', 'butterfly', 'scissor', 'backlit', 'sticky', 'tactile', 'cursor', 'mouse', 'typing', 'click'],
        'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'worth', 'money', 'budget', 'investment', 'affordable', 'pricey', 'overpriced', 'bargain'],
        'software': ['software', 'os', 'mac', 'windows', 'ios', 'apps', 'program', 'system', 'update', 'macos', 'big sur', 'monterey', 'ventura', 'sonoma', 'catalina'],
        'support': ['support', 'service', 'help', 'apple care', 'warranty', 'genius', 'repair', 'customer', 'technical', 'assistant', 'replacement'],
        'ports': ['ports', 'usb', 'thunderbolt', 'hdmi', 'sd card', 'adapter', 'dongle', 'hub', 'connector', 'cable', 'connectivity'],
        'audio': ['audio', 'sound', 'speaker', 'microphone', 'headphone', 'jack', 'volume', 'quality', 'bass', 'treble', 'noise', 'recording'],
        'cooling': ['cooling', 'fan', 'heat', 'hot', 'temperature', 'thermal', 'loud', 'noise', 'throttle', 'vent', 'airflow']
    }
    
    # Initialize aspect columns
    for aspect in laptop_aspects:
        df[f'has_{aspect}'] = False
        df[f'{aspect}_sentiment'] = np.nan
        df[f'{aspect}_sentences'] = None
    
    # Rule-based aspect extraction and sentiment analysis
    def extract_aspect_sentiment_rule_based(text, sid):
        text = str(text).lower()
        results = {}
        sentences_by_aspect = {}
        
        # Split text into sentences for analysis
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for aspect, keywords in laptop_aspects.items():
            aspect_mentions = []
            aspect_sentences = []
            
            for sentence in sentences:
                if any(re.search(r'\b' + keyword + r'[a-z]*\b', sentence) for keyword in keywords):
                    sentiment = sid.polarity_scores(sentence)['compound']
                    aspect_mentions.append((sentence, sentiment))
                    aspect_sentences.append(sentence)
            
            if aspect_mentions:
                results[f'has_{aspect}'] = True
                results[f'{aspect}_sentiment'] = np.mean([s for _, s in aspect_mentions])
                sentences_by_aspect[f'{aspect}_sentences'] = aspect_sentences
        
        results.update({k: v for k, v in sentences_by_aspect.items()})
        return results
    
    # Apply rule-based analysis
    sid = SentimentIntensityAnalyzer()
    print("Applying rule-based aspect sentiment analysis...")
    aspect_results = df['full_text'].apply(lambda x: extract_aspect_sentiment_rule_based(x, sid))
    
    # Update dataframe with aspect results
    for i, result in enumerate(aspect_results):
        for key, value in result.items():
            df.loc[i, key] = value
    
    # Check if transformers are available for advanced analysis
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        import torch
        transformers_available = True
    except ImportError:
        print("Transformers library not available. Falling back to rule-based analysis only.")
        transformers_available = False
    
    # Function to handle long text with transformers by chunking
    def process_long_text(text, analyzer, max_length=512):
        """Process long text by chunking it into smaller pieces."""
        if not text or len(text) <= max_length:
            return analyzer(text)
        
        # Split into chunks
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        results = []
        
        # Process each chunk
        for chunk in chunks:
            result = analyzer(chunk)
            results.extend(result)
            
        return results
    
    # Apply transformer-based analysis if available
    if transformers_available:
        print("Initializing transformer models for advanced aspect-based sentiment analysis...")
        
        # Initialize tokenizer and model for aspect-based sentiment analysis
        # For this example we'll use the ABSA (Aspect-Based Sentiment Analysis) model
        try:
            # First, let's try to get a model specifically for ABSA if available
            absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
            absa_model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
            
            # Create a pipeline for zero-shot aspect classification
            classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Create a sentiment analysis pipeline
            sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Sample for performance (process a subset if dataset is large)
            sample_size = min(100, len(df))
            sampled_indices = np.random.choice(df.index, size=sample_size, replace=False)
            
            print(f"Applying transformer-based analysis on a sample of {sample_size} reviews...")
            
            # Apply transformer model to the sampled reviews
            for idx in sampled_indices:
                text = df.loc[idx, 'full_text']
                
                # Skip if text is missing or empty
                if pd.isna(text) or text == "":
                    continue
                
                for aspect in laptop_aspects:
                    # Skip if the rule-based method didn't detect this aspect
                    if not df.loc[idx, f'has_{aspect}']:
                        continue
                    
                    # Get sentences related to this aspect
                    aspect_sentences = df.loc[idx, f'{aspect}_sentences']
                    if not aspect_sentences or len(aspect_sentences) == 0:
                        continue
                    
                    # Join sentences for combined analysis
                    aspect_text = " ".join(aspect_sentences)
                    
                    # Process with transformer for more accurate sentiment
                    # Use chunking for long texts
                    try:
                        sentiment_result = process_long_text(aspect_text, sentiment_analyzer)
                        # Average the sentiment scores from all chunks
                        scores = [item['score'] if item['label'] == 'POSITIVE' else 1 - item['score'] for item in sentiment_result]
                        avg_score = sum(scores) / len(scores) if scores else 0.5
                        # Convert to -1 to 1 scale
                        transformer_sentiment = (avg_score * 2) - 1
                        
                        # Combine rule-based and transformer sentiment (weighted average)
                        rule_sentiment = df.loc[idx, f'{aspect}_sentiment']
                        if not pd.isna(rule_sentiment):
                            # Weighted average: 40% rule-based, 60% transformer
                            combined_sentiment = (0.4 * rule_sentiment) + (0.6 * transformer_sentiment)
                            df.loc[idx, f'{aspect}_sentiment'] = combined_sentiment
                    except Exception as e:
                        print(f"Error in transformer analysis: {e}")
                        # Keep the rule-based sentiment if transformer fails
                        pass
            
            print("Transformer-based analysis completed.")
            
        except Exception as e:
            print(f"Error loading transformer models: {e}")
            print("Continuing with rule-based analysis only.")
    
    # Add confidence scores for aspect detection
    for aspect in laptop_aspects:
        # Calculate confidence based on the number of mentions
        aspect_sentences_col = f'{aspect}_sentences'
        if aspect_sentences_col in df.columns:
            df[f'{aspect}_confidence'] = df[aspect_sentences_col].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
        else:
            df[f'{aspect}_confidence'] = 0
    
    # Create summary columns for aspect sentiment
    df['positive_aspects'] = df[[f'{aspect}_sentiment' for aspect in laptop_aspects]].apply(
        lambda x: [aspect for aspect, col in zip(laptop_aspects, x) if not pd.isna(col) and col > 0.2],
        axis=1
    )
    
    df['negative_aspects'] = df[[f'{aspect}_sentiment' for aspect in laptop_aspects]].apply(
        lambda x: [aspect for aspect, col in zip(laptop_aspects, x) if not pd.isna(col) and col < -0.2],
        axis=1
    )
    
    # Calculate normalized aspect importance
    aspect_counts = {aspect: df[f'has_{aspect}'].sum() for aspect in laptop_aspects}
    total_reviews = len(df)
    df['aspect_importance'] = {aspect: count/total_reviews for aspect, count in aspect_counts.items()}
    
    return df

# Step 6: Issue Identification
def identify_issues(df):
    """Identify common issues mentioned in negative reviews."""
    # Look at negative reviews
    negative_reviews = df[df['sentiment_category'] == 'Negative']
    
    # Common issue phrases
    issue_phrases = [
        'not working', 'doesn\'t work', 'stopped working', 'failed', 'problem',
        'issue', 'defect', 'broken', 'error', 'crash', 'slow', 'lag',
        'expensive', 'overpriced', 'not worth', 'disappointed', 'regret'
    ]
    
    # Find issues in reviews
    for phrase in issue_phrases:
        col_name = f'issue_{phrase.replace(" ", "_")}'
        df[col_name] = df['full_text'].str.lower().str.contains(phrase, regex=False)
    
    # Identify most common words in negative reviews
    if len(negative_reviews) > 0:
        stop_words = set(stopwords.words('english'))
        negative_words = []
        
        for text in negative_reviews['full_text']:
            words = word_tokenize(str(text).lower())
            words = [word for word in words if word.isalpha() and word not in stop_words]
            negative_words.extend(words)
        
        common_negative_words = Counter(negative_words).most_common(30)
        df.attrs['common_negative_words'] = common_negative_words
    
    return df

# Step 7: Customer Intent Analysis
def analyze_intent(df):
    """Analyze customer intent based on review content."""
    # Intent categories and their keywords
    intent_categories = {
        'recommendation': ['recommend', 'suggest', 'advice', 'should buy', 'worth'],
        'comparison': ['compare', 'versus', 'vs', 'better than', 'worse than', 'pc', 'windows', 'mac'],
        'experience_sharing': ['I used', 'I have', 'I bought', 'my experience', 'I owned'],
        'problem_reporting': ['problem', 'issue', 'doesn\'t work', 'broken', 'failed'],
        'feature_feedback': ['feature', 'missing', 'needs', 'should have', 'lack'],
        'purchase_decision': ['decided', 'purchase', 'buy', 'bought', 'ordered']
    }
    
    # Detect intent
    for intent, keywords in intent_categories.items():
        df[f'intent_{intent}'] = False
        for keyword in keywords:
            df[f'intent_{intent}'] = df[f'intent_{intent}'] | df['full_text'].str.lower().str.contains(r'\b' + keyword + r'\b', regex=True)
    
    # Determine primary intent
    intent_columns = [f'intent_{intent}' for intent in intent_categories]
    
    # Create a helper function for primary intent
    def get_primary_intent(row):
        for intent in intent_categories:
            if row[f'intent_{intent}']:
                return intent
        return 'unknown'
    
    df['primary_intent'] = df.apply(get_primary_intent, axis=1)
    
    return df

# Step 8: Visualization Functions
def create_visualizations(df, output_dir='./visualizations'):
    """Create visualizations from the analyzed data."""
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set style
    plt.style.use('ggplot')
    
    # 1. Rating Distribution
    if 'rating' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x='rating', data=df, palette='viridis')
        plt.title('Distribution of Ratings')
        plt.savefig(f'{output_dir}/rating_distribution.png')
        plt.close()
    
    # 2. Sentiment Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='sentiment_category', data=df, palette='RdYlGn')
    plt.title('Sentiment Distribution')
    plt.savefig(f'{output_dir}/sentiment_distribution.png')
    plt.close()
    
    # 3. Emotion Distribution
    emotion_columns = ['joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust', 'neutral']
    emotion_counts = df['dominant_emotion'].value_counts()
    
    plt.figure(figsize=(12, 7))
    emotion_counts.plot(kind='bar', color=sns.color_palette('viridis', len(emotion_counts)))
    plt.title('Dominant Emotions in Reviews')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.savefig(f'{output_dir}/emotion_distribution.png')
    plt.close()
    
    # 4. Aspect Sentiment
    aspects = ['performance', 'battery', 'display', 'design', 'keyboard', 'price', 'software', 'support']
    aspect_sentiments = []
    
    for aspect in aspects:
        mean_sentiment = df[df[f'has_{aspect}']][f'{aspect}_sentiment'].mean()
        if not np.isnan(mean_sentiment):
            aspect_sentiments.append((aspect, mean_sentiment))
    
    if aspect_sentiments:
        aspects, sentiments = zip(*aspect_sentiments)
        plt.figure(figsize=(12, 7))
        ax = sns.barplot(x=list(aspects), y=list(sentiments), palette='RdYlGn_r')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title('Average Sentiment by Aspect')
        plt.xlabel('Aspect')
        plt.ylabel('Average Sentiment (-1 to 1)')
        plt.xticks(rotation=45)
        
        # Add sentiment values on top of bars
        for i, v in enumerate(sentiments):
            ax.text(i, v + 0.05 if v >= 0 else v - 0.1, f'{v:.2f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/aspect_sentiment.png')
        plt.close()
    
    # 5. Intent Distribution
    plt.figure(figsize=(12, 7))
    sns.countplot(y='primary_intent', data=df, order=df['primary_intent'].value_counts().index, palette='Set2')
    plt.title('Primary Intent Distribution')
    plt.savefig(f'{output_dir}/intent_distribution.png')
    plt.close()
    
    # 6. Word Cloud of positive and negative reviews
    try:
        from wordcloud import WordCloud
        
        # Positive word cloud
        positive_text = ' '.join(df[df['sentiment_category'] == 'Positive']['full_text'])
        if positive_text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                max_words=100, contour_width=3, contour_color='steelblue').generate(positive_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud - Positive Reviews')
            plt.savefig(f'{output_dir}/positive_wordcloud.png')
            plt.close()
        
        # Negative word cloud
        negative_text = ' '.join(df[df['sentiment_category'] == 'Negative']['full_text'])
        if negative_text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                max_words=100, contour_width=3, contour_color='firebrick').generate(negative_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud - Negative Reviews')
            plt.savefig(f'{output_dir}/negative_wordcloud.png')
            plt.close()
    except ImportError:
        print("WordCloud not available. Skipping word cloud visualizations.")

# Step 9: Text Preprocessing for Large Reviews
def preprocess_long_text(df, max_tokens=1000):
    """Preprocess and optimize handling of long text reviews."""
    print("Optimizing long text reviews...")
    
    # Check text length
    df['text_length'] = df['full_text'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
    
    # Identify long reviews
    long_reviews = df['text_length'] > max_tokens
    print(f"Found {long_reviews.sum()} reviews with more than {max_tokens} tokens")
    
    if long_reviews.sum() > 0:
        try:
            # Try to use advanced NLP techniques if available
            import spacy
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            print("Using advanced NLP techniques for text summarization")
            
            # Load spaCy model
            nlp = spacy.load("en_core_web_sm")
            
            # Function to extract key sentences using TF-IDF
            def extract_key_sentences(text, n_sentences=5):
                if pd.isna(text) or text == "":
                    return text
                
                # Parse with spaCy
                doc = nlp(str(text))
                sentences = [sent.text for sent in doc.sents]
                
                if len(sentences) <= n_sentences:
                    return text
                
                # Use TF-IDF to find most important sentences
                vectorizer = TfidfVectorizer(stop_words='english')
                
                # Handle case where we have too few sentences
                if len(sentences) < 2:
                    return text
                
                # Compute TF-IDF matrix
                tfidf_matrix = vectorizer.fit_transform(sentences)
                sentence_scores = tfidf_matrix.sum(axis=1).A1
                
                # Get indices of top sentences
                top_indices = sentence_scores.argsort()[-n_sentences:]
                
                # Sort indices to maintain original order
                top_indices = sorted(top_indices)
                
                # Join top sentences
                summary = ' '.join([sentences[i] for i in top_indices])
                return summary
            
            # Apply summarization to long reviews
            df.loc[long_reviews, 'full_text_original'] = df.loc[long_reviews, 'full_text'].copy()
            df.loc[long_reviews, 'full_text'] = df.loc[long_reviews, 'full_text'].apply(extract_key_sentences)
            
            print("Text summarization completed")
            
        except ImportError:
            print("Advanced NLP libraries not available. Using simple truncation for long reviews.")
            # Simple approach - keep intro and conclusion
            df.loc[long_reviews, 'full_text_original'] = df.loc[long_reviews, 'full_text'].copy()
            
            def simple_optimize(text, max_tokens=max_tokens):
                if pd.isna(text) or text == "":
                    return text
                
                words = str(text).split()
                if len(words) <= max_tokens:
                    return text
                
                # Keep beginning and end portions
                beginning = ' '.join(words[:max_tokens//2])
                end = ' '.join(words[-max_tokens//2:])
                return beginning + " [...] " + end
            
            df.loc[long_reviews, 'full_text'] = df.loc[long_reviews, 'full_text'].apply(simple_optimize)
    
    return df

# Step 10: ML-Enhanced Model Implementation
def implement_ml_models(df):
    """Implement machine learning models for improved sentiment analysis."""
    print("Implementing ML-enhanced models...")
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report
        
        # Check if we have enough labeled data
        if len(df) >= 20 and 'sentiment_category' in df.columns:  # Need minimum samples for ML
            # Prepare features and target for sentiment prediction
            X = df['full_text'].fillna('')
            y = df['sentiment_category']
            
            # Create train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
            )
            
            # Create TF-IDF features
            vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)
            
            # Train a RandomForest classifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train_tfidf, y_train)
            
            # Make predictions
            y_pred = clf.predict(X_test_tfidf)
            
            # Output evaluation
            print("ML Model Performance:")
            print(classification_report(y_test, y_pred))
            
            # Apply model to improve aspect sentiment
            print("Enhancing aspect sentiment with ML predictions...")
            aspects = list(df.filter(regex='^has_').columns)
            aspects = [a.replace('has_', '') for a in aspects]
            
            for aspect in aspects:
                if df[f'has_{aspect}'].sum() >= 10:  # Need enough samples
                    # Get reviews mentioning this aspect
                    aspect_reviews = df[df[f'has_{aspect}']]
                    
                    if len(aspect_reviews) >= 10:  # Check again after filtering
                        # Create sentiment features
                        X_aspect = aspect_reviews[f'{aspect}_sentences'].apply(
                            lambda x: ' '.join(x) if isinstance(x, list) else '')
                        X_aspect = X_aspect.fillna('')
                        
                        # Create a binary sentiment target (positive vs negative)
                        y_aspect = aspect_reviews[f'{aspect}_sentiment'].apply(
                            lambda x: 'Positive' if x > 0 else 'Negative')
                        
                        # Create train/test split for this aspect
                        X_a_train, X_a_test, y_a_train, y_a_test = train_test_split(
                            X_aspect, y_aspect, test_size=0.2, random_state=42, 
                            stratify=y_aspect if len(set(y_aspect)) > 1 else None
                        )
                        
                        if len(set(y_a_train)) > 1:  # Need both classes
                            # Create TF-IDF features
                            vectorizer_aspect = TfidfVectorizer(max_features=1000)
                            X_a_train_tfidf = vectorizer_aspect.fit_transform(X_a_train)
                            
                            # Train a RandomForest for this aspect
                            clf_aspect = RandomForestClassifier(n_estimators=50, random_state=42)
                            clf_aspect.fit(X_a_train_tfidf, y_a_train)
                            
                            # Store model and vectorizer
                            df.attrs[f'model_{aspect}'] = clf_aspect
                            df.attrs[f'vectorizer_{aspect}'] = vectorizer_aspect
            
            print("ML enhancement completed")
    
    except ImportError:
        print("Machine learning libraries not available. Skipping ML enhancements.")
    
    return df

# Step 11: Main Analysis Function
def analyze_laptop_reviews(file_path, output_dir='./', use_ml=True, optimize_text=True):
    """Main function to analyze laptop reviews."""
    print("Starting laptop review sentiment analysis...")
    
    # Set up dependency installation if needed
    try:
        import pkg_resources
        pkg_resources.require(['transformers', 'torch', 'spacy', 'sklearn'])
    except (ImportError, pkg_resources.DistributionNotFound):
        print("Required packages not found. Installing dependencies...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                            "transformers", "torch", "spacy", "scikit-learn"])
        # Download spacy model
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

    # Prepare data
    print("Preparing data...")
    df = prepare_data(file_path)
    
    # Optimize long text if enabled
    if optimize_text:
        df = preprocess_long_text(df)
    
    # Basic statistics
    print("Generating basic statistics...")
    stats = generate_basic_statistics(df)
    
    # Sentiment analysis
    print("Analyzing sentiment...")
    df = analyze_sentiment(df)
    
    # Emotion analysis
    print("Analyzing emotions...")
    df = analyze_emotions(df)
    
    # Aspect-based analysis with transformers
    print("Analyzing aspects...")
    df = analyze_aspects(df)
    
    # Add ML models if enabled
    if use_ml:
        df = implement_ml_models(df)
    
    # Issue identification
    print("Identifying issues...")
    df = identify_issues(df)
    
    # Intent analysis
    print("Analyzing intent...")
    df = analyze_intent(df)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(df, output_dir)
    
    # Generate summary report
    print("Generating summary report...")
    generate_report(df, stats, output_dir)
    
    print("Analysis complete!")
    return df



# Execute the analysis function if this script is run directly
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else './'
        analyze_laptop_reviews(file_path, output_dir)
    else:
        print("Please provide the path to the laptop reviews data file.")
        print("Usage: python laptop_sentiment_analysis.py <file_path> [output_directory]")
