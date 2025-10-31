import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import re
import string

def preprocess_text(text):
    """text preprocessing function"""
    # convert to lowercase
    text = text.lower()
    # remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # remove numbers
    text = re.sub(r'\d+', '', text)
    # remove extra whitespace
    text = ' '.join(text.split())
    return text

def load_and_prepare_data():
    """Load and prepare dataset"""
    print("loading dataset")
    
    # read datasets
    fake_df = pd.read_csv('Fake.csv')
    true_df = pd.read_csv('True.csv')
    
    # add labels
    fake_df['label'] = 0  
    true_df['label'] = 1  
    
    # merge datasets
    df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)
    
    # combine title and text
    df['content'] = df['title'] + ' ' + df['text']
    
    # shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"total samples: {len(df)}")
    print(f"fake news samples: {len(df[df['label']==0])}")
    print(f"real news samples: {len(df[df['label']==1])}")
    
    return df

def train_model(df):
    """"Train and evaluate the model"""
    print("\npreprocessing text")
    df['processed_content'] = df['content'].apply(preprocess_text)
    
    # split dataset
    X = df['processed_content']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\ntraining set size: {len(X_train)}")
    print(f"training set size: {len(X_test)}")
    
    # TF-IDF向量化
    print("\nperforming TF-IDF vectorization")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.7,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # training logistic regression model
    print("\ntraining model")
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_tfidf, y_train)
    
    # evaluate model
    print("\n model evaluation:")
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nclassification report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['fake news', 'real news']))
    
    print("\nconfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # save model and vectorizer
    print("\nsaving model and vectorizer")
    with open('fake_news_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print("\nmodel saved as 'fake_news_model.pkl'")
    print("vectorizer saved as'tfidf_vectorizer.pkl'")
    
    return model, vectorizer, accuracy

if __name__ == "__main__":
    print("Fake News Detection Model Training")
    
    try:
        # 加载数据
        df = load_and_prepare_data()
        
        # 训练模型
        model, vectorizer, accuracy = train_model(df)
        

        print(f"training complete final accuracy: {accuracy*100:.2f}%")
        
    except FileNotFoundError:
        print("\nError: Dataset files not found!")
        print("Please make sure 'Fake.csv' and 'True.csv' are in the current directory.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise