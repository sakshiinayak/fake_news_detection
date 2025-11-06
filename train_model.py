import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import re
import string

# Text preprocessing function
def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = " ".join(text.split())
    return text

# Load and prepare dataset
def load_data():
    """
    Load fake and real news datasets
    Expected files: Fake.csv and True.csv from Kaggle
    Download from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
    """
    try:
        fake_df = pd.read_csv("Fake.csv")
        true_df = pd.read_csv("True.csv")
        
        fake_df['label'] = 0  # Fake news
        true_df['label'] = 1  # Real news
        
        # Combine datasets
        df = pd.concat([fake_df, true_df], ignore_index=True)
        
        # Combine title and text
        df['content'] = df['title'].fillna('') + " " + df['text'].fillna('')
        
        print(f"Dataset loaded: {len(df)} articles")
        print(f"Fake news: {len(fake_df)}, Real news: {len(true_df)}")
        
        return df
    except FileNotFoundError:
        print("Error: Dataset files not found!")
        print("Please download 'Fake.csv' and 'True.csv' from Kaggle:")
        print("https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        return None

# Train model with improved parameters
def train_model(df):
    """Train the fake news detection model with optimized parameters"""
    
    # Preprocess text
    print("Preprocessing text...")
    df['clean_content'] = df['content'].apply(preprocess_text)
    
    # Remove empty texts
    df = df[df['clean_content'].str.len() > 0]
    
    # Split data
    X = df['clean_content']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # TF-IDF Vectorization with improved parameters
    print("Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=10000,  # Increased from 5000
        ngram_range=(1, 3),   # Include trigrams
        min_df=3,             # Reduced from 5
        max_df=0.7,           # Reduced from 0.8 to ignore very common words
        sublinear_tf=True,    # Apply sublinear tf scaling
        use_idf=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train Multiple Models and Compare
    print("\n" + "="*60)
    print("Training and comparing models...")
    print("="*60)
    
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            C=2.0,  # Increased regularization parameter
            class_weight='balanced',  # Handle any class imbalance
            random_state=42,
            n_jobs=-1,
            solver='saga'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=50,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_tfidf, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Test accuracy
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_name = name
    
    print("\n" + "="*60)
    print(f"Best Model: {best_name} with accuracy {best_score:.4f}")
    print("="*60)
    
    # Detailed evaluation of best model
    y_pred = best_model.predict(X_test_tfidf)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nTrue Negatives (Fake correctly identified): {cm[0][0]}")
    print(f"False Positives (Real incorrectly marked as Fake): {cm[1][0]}")
    print(f"False Negatives (Fake incorrectly marked as Real): {cm[0][1]}")
    print(f"True Positives (Real correctly identified): {cm[1][1]}")
    
    # Calculate specific metrics
    fake_precision = cm[0][0] / (cm[0][0] + cm[1][0]) if (cm[0][0] + cm[1][0]) > 0 else 0
    real_precision = cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
    
    print(f"\nFake News Precision: {fake_precision:.4f} ({fake_precision*100:.2f}%)")
    print(f"Real News Precision: {real_precision:.4f} ({real_precision*100:.2f}%)")
    
    # Save model and vectorizer
    print("\n" + "="*60)
    print("Saving model and vectorizer...")
    with open("fake_news_model_v2.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("tfidf_vectorizer_v2.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    # Save model metadata
    metadata = {
        'model_type': best_name,
        'accuracy': best_score,
        'fake_precision': fake_precision,
        'real_precision': real_precision,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    with open("model_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    print("✅ Files saved successfully!")
    print("   - fake_news_model_v2.pkl")
    print("   - tfidf_vectorizer_v2.pkl")
    print("   - model_metadata.pkl")
    print("="*60)
    
    return best_model, vectorizer, metadata

# Test with sample texts
def test_sample_predictions(model, vectorizer):
    """Test model with sample real and fake news"""
    print("\n" + "="*60)
    print("Testing with sample texts...")
    print("="*60)
    
    samples = [
        ("NASA's Lucy Mission Captures First Images of Jupiter's Trojan Asteroids. "
         "NASA's Lucy spacecraft has successfully captured its first close-up images of Jupiter's Trojan asteroids. "
         "The mission aims to study these ancient remnants to understand planetary formation.", "Real"),
        
        ("SHOCKING: Scientists discover aliens living among us! Share this before it gets deleted! "
         "You won't believe what happens next!", "Fake"),
        
        ("The Federal Reserve announced a quarter-point interest rate increase today, "
         "marking the fourth consecutive rate hike this year as officials continue efforts to combat inflation.", "Real"),
        
        ("BREAKING: Celebrity found dead in mysterious circumstances! "
         "Doctors hate this one weird trick! Click here now!", "Fake")
    ]
    
    for text, expected in samples:
        clean_text = preprocess_text(text)
        tfidf = vectorizer.transform([clean_text])
        pred = model.predict(tfidf)[0]
        proba = model.predict_proba(tfidf)[0]
        
        result = "REAL" if pred == 1 else "FAKE"
        confidence = max(proba) * 100
        
        print(f"\nExpected: {expected} | Predicted: {result} | Confidence: {confidence:.1f}%")
        print(f"Text preview: {text[:100]}...")
        
        if (expected == "Real" and pred == 1) or (expected == "Fake" and pred == 0):
            print("✅ Correct prediction")
        else:
            print("❌ Incorrect prediction")

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("Fake News Detection Model Training (Improved)")
    print("="*60 + "\n")
    
    # Load data
    df = load_data()
    
    if df is not None:
        # Train model
        model, vectorizer, metadata = train_model(df)
        
        # Test with samples
        test_sample_predictions(model, vectorizer)
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("Run: streamlit run app.py")
        print("="*60)