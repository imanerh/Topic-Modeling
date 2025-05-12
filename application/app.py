from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import json
import os
import torch
from transformers import RobertaTokenizer, RobertaModel
import traceback
import warnings
import pandas as pd

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)

# Global variables to store loaded models
models = {}

# Define consistent colors for topics
TOPIC_COLORS = {
    "business": "#4e79a7",  # Blue
    "entertainment": "#f28e2c",  # Orange
    "politics": "#59a14f",  # Green
    "sport": "#e15759",  # Red
    "tech": "#af7aa1",  # Purple
    "Your Document": "#000" # Black
}

def load_models():
    """Load all required models and data"""
    print("Loading models...")
    
    model_dir = "topic_model_export"
    
    # Check for version info
    try:
        if os.path.exists(os.path.join(model_dir, "version_info.json")):
            with open(os.path.join(model_dir, "version_info.json"), "r") as f:
                version_info = json.load(f)
                print(f"Models were created with: {version_info}")
    except Exception as e:
        print(f"Failed to load version info: {e}")
    
    # Load tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    model.eval()  # Set to evaluation mode
    
    # Load topic metadata
    with open(os.path.join(model_dir, "topic_metadata.json"), "r") as f:
        topic_metadata = json.load(f)
    
    # Load cluster centroids
    cluster_centroids = np.load(os.path.join(model_dir, "cluster_centroids_umap.npy"))
    print(f"Loaded cluster centroids with shape: {cluster_centroids.shape}")
    
    # Load 2D UMAP embeddings for visualization
    umap_embeddings_2d = np.load(os.path.join(model_dir, "umap_embeddings_2d.npy"))

    # Initialize a variable to track if we're using a fallback method
    using_fallback = False
    
    # Load UMAP model for transforming new embeddings
    try:
        with open(os.path.join(model_dir, "umap_model.pkl"), "rb") as f:
            umap_model = pickle.load(f)
        print("Successfully loaded UMAP model")
        
        # Test that the UMAP model works by transforming a sample vector
        try:
            test_vector = np.random.random((1, 768))  # RoBERTa embedding size
            transformed = umap_model.transform(test_vector)
            print(f"UMAP transform test successful. Input: {test_vector.shape}, Output: {transformed.shape}")
        except Exception as e:
            print(f"UMAP transform test failed: {e}")
            print("Will use a fallback method for classification")
            umap_model = None
            using_fallback = True
    except Exception as e:
        print(f"Failed to load UMAP model: {e}")
        umap_model = None
        using_fallback = True
    
    # Fallback: Try to load original embeddings and cluster assignments
    # for a nearest-neighbor approach if UMAP model fails
    original_embeddings = None
    original_labels = None
    if using_fallback:
        try:
            if os.path.exists(os.path.join(model_dir, "roberta_embeddings.npy")):
                print("Loading original RoBERTa embeddings for fallback method...")
                original_embeddings = np.load(os.path.join(model_dir, "roberta_embeddings.npy"))
                print(f"Loaded {len(original_embeddings)} original embeddings")
                
                if os.path.exists(os.path.join(model_dir, "document_clusters.csv")):
                    print("Loading original cluster assignments...")
                    doc_clusters = pd.read_csv(os.path.join(model_dir, "document_clusters.csv"))
                    original_labels = doc_clusters['cluster'].values
                    print(f"Loaded cluster assignments for {len(original_labels)} documents")
        except Exception as e:
            print(f"Failed to load fallback data: {e}")
    
    # Define UMAP transform function with fallback options
    def transform_embeddings(X):
        """Transform RoBERTa embeddings to the dimensionality used for clustering"""
        # Convert to numpy array if needed
        if isinstance(X, list):
            X = np.array(X)
        
        # Reshape if needed
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        print(f"Input embedding shape: {X.shape}")
        
        # If UMAP model is available and working, use it
        if umap_model is not None:
            try:
                transformed = umap_model.transform(X)
                print(f"UMAP transform successful. Output shape: {transformed.shape}")
                return transformed
            except Exception as e:
                print(f"UMAP transform failed: {e}")
                print("Falling back to alternative method")
                # Continue to fallback methods
        
        # Fallback 1: If we have original embeddings, use nearest neighbors in original space
        if original_embeddings is not None and original_labels is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            
            print("Using nearest neighbors in original embedding space to assign cluster")
            # Find most similar original embeddings
            similarities = cosine_similarity(X, original_embeddings)[0]
            most_similar_idx = np.argmax(similarities)
            most_similar_cluster = original_labels[most_similar_idx]
            
            # Return the centroid of the most similar cluster
            return cluster_centroids[most_similar_cluster].reshape(1, -1)
        
        # Fallback 2: Simple truncation to match centroid dimensions
        target_dim = cluster_centroids.shape[1]
        print(f"Using simple truncation to {target_dim} dimensions")
        result = X[:, :target_dim]
        
        return result
    
    # Create topic coordinates for visualization
    topic_coordinates = []
    for topic in topic_metadata:
        cluster_id = topic["cluster_id"]
        category = topic["dominant_category"]

        # Assign color from our mapping (with fallback)
        color = TOPIC_COLORS.get(category.lower(), "#999999")

        # Find the centroid in 2D space from the 2D embeddings
        # Get all docs in this cluster
        doc_indices = []
        try:
            doc_clusters = pd.read_csv(os.path.join(model_dir, "document_clusters.csv"))
            doc_indices = doc_clusters[doc_clusters['cluster'] == cluster_id].index.tolist()
        except:
            # If we can't load the document clusters, just use a fixed position
            x = float(cluster_id * 3)
            y = float((cluster_id % 3) * 3)
            topic_coordinates.append({
                "id": cluster_id,
                "label": topic["dominant_category"],
                "x": x,
                "y": y,
                "color": color  
            })
            continue
        
        # If we have indices, calculate mean position from 2D embeddings
        if doc_indices:
            if len(doc_indices) > len(umap_embeddings_2d):
                # Safety check
                doc_indices = doc_indices[:len(umap_embeddings_2d)]
                
            # Calculate mean position
            positions = umap_embeddings_2d[doc_indices]
            mean_pos = positions.mean(axis=0)
            
            topic_coordinates.append({
                "id": cluster_id,
                "label": category,
                "x": float(mean_pos[0]),
                "y": float(mean_pos[1]),
                "color": color  
            })
        else:
            # Fallback to fixed position
            x = float(cluster_id * 3)
            y = float((cluster_id % 3) * 3)
            topic_coordinates.append({
                "id": cluster_id,
                "label": topic["dominant_category"],
                "x": x,
                "y": y,
                "color": color  
            })
    
    # Example documents
    example_documents = [
        {
            "title": "Presidential Debate Analysis",
            "text": "Presidential Debate Shows Divided Electorate. Last night's presidential debate revealed sharp policy differences as both candidates tried to appeal to undecided voters. The incumbent president defended his administration's economic record while the opposition candidate criticized government spending and foreign policy decisions. Recent polls show a tight race in key battleground states with just three weeks until election day. Political analysts noted that healthcare and climate policy emerged as major points of contention, with each candidate offering distinctly different approaches. The debate moderator struggled at times to maintain order as discussions became heated on topics of taxation and immigration reform."
        },
        {
            "title": "Electoral Reform Proposal",
            "text": "Congressional leaders unveiled a bipartisan electoral reform proposal today aimed at strengthening voting rights and election security. The comprehensive bill includes provisions for automatic voter registration, expanded early voting options, and increased funding for election infrastructure security. Supporters argue the legislation will increase voter participation and restore faith in democratic institutions, while critics have raised concerns about federal overreach into state election administration. The bill faces an uncertain future in the divided Senate, where it would need to overcome a potential filibuster to advance to a final vote. Political observers suggest this could become a defining issue in upcoming midterm campaigns."
        },
        {
            "title": "Market Analysis Report",
            "text": "Stock markets showed strong growth today as investors responded positively to new economic data. The banking sector led gains with major financial institutions reporting better than expected quarterly results. Market analysts attribute the rally to improving employment numbers and signals from the Federal Reserve suggesting interest rates will remain stable through the next quarter. Technology stocks also performed well, with several companies announcing expanded product lines and positive revenue forecasts. However, some economists caution that inflation pressures and ongoing supply chain constraints could impact growth in the manufacturing sector, potentially creating market volatility in coming months."
        },
        {
            "title": "Film Festival Winners",
            "text": "The International Film Festival concluded with an award ceremony celebrating this year's best films. The top prize went to an independent drama that critics praised for its powerful storytelling and innovative cinematography. The director, a first-time filmmaker, delivered an emotional acceptance speech highlighting the five-year journey to bring the project to screen. Other notable winners included a documentary examining climate change impacts on indigenous communities and an animated feature that blended traditional techniques with cutting-edge digital effects. Festival attendance reached record numbers this year, with organizers citing increased interest in international cinema and expanded virtual screening options."
        },
        {
            "title": "Championship Final",
            "text": "The team secured an impressive win in yesterday's championship match, with their star player scoring in the final minutes. Fans celebrated as the victory marks their third consecutive title in the national league. The closely contested match saw both teams demonstrating exceptional skill and determination, with momentum shifting multiple times throughout the game. The winning goal came after a perfectly executed set piece that had been practiced extensively during training sessions. The coach credited the victory to the team's mental resilience and strategic adjustments made at halftime. Meanwhile, the opposing team's manager acknowledged his players' effort but expressed disappointment at missed opportunities in critical moments."
        },
        {
            "title": "Quantum Computing Breakthrough",
            "text": "Researchers have announced a significant breakthrough in quantum computing architecture that could accelerate artificial intelligence systems exponentially. The new quantum processor features stable qubits and demonstrates unprecedented coherence times, addressing one of the field's most persistent challenges. Scientists at the university's advanced computing laboratory demonstrated the processor performing complex calculations that would take conventional supercomputers several years to complete. Industry experts suggest this development could have far-reaching implications for cryptography, materials science, and pharmaceutical development. The research team has published their findings in a peer-reviewed journal and made their technical specifications available to the broader scientific community to encourage collaborative advancements."
        },
        {
            "title": "AI Ethics Framework",
            "text": "A consortium of leading technology companies announced today a new ethical framework for artificial intelligence development and deployment. The voluntary guidelines address concerns about algorithmic bias, data privacy, and transparency in AI decision-making processes. The framework establishes independent review procedures for high-risk AI applications and sets standards for documenting training data sources. Tech policy experts have generally welcomed the initiative but note that without regulatory backing, compliance remains optional. Consumer advocacy groups are pushing for additional protections against automated decision systems in sensitive areas like employment, housing, and healthcare. The announcement comes as legislators in several countries are considering more stringent AI governance frameworks."
        }
    ]
    
    # Load K-means model
    try:
        with open(os.path.join(model_dir, "kmeans_model.pkl"), "rb") as f:
            kmeans_model = pickle.load(f)
        print("Successfully loaded K-means model")
        
        # Create a prediction function that works with our transform
        def kmeans_predict(X):
            # First transform to match dimensions
            X_transformed = transform_embeddings(X)
            # Then predict
            return kmeans_model.predict(X_transformed)
        
    except Exception as e:
        print(f"Failed to load K-means model: {e}")
        kmeans_model = None
        
        # Create a simple alternative with cosine similarity
        def kmeans_predict(X):
            # Transform to match dimensions
            X_transformed = transform_embeddings(X)
            
            # Calculate similarity to centroids
            similarities = []
            for i, centroid in enumerate(cluster_centroids):
                # Calculate cosine similarity
                similarity = np.dot(X_transformed[0], centroid) / (
                    np.linalg.norm(X_transformed[0]) * np.linalg.norm(centroid)
                )
                similarities.append((i, similarity))
            
            # Return index of most similar centroid
            most_similar = max(similarities, key=lambda x: x[1])[0]
            return [most_similar]
    
    print("Models loaded successfully.")
    return {
        "tokenizer": tokenizer,
        "model": model,
        "transform": transform_embeddings,
        "predict": kmeans_predict,
        "topic_metadata": topic_metadata,
        "topic_coordinates": topic_coordinates,
        "cluster_centroids": cluster_centroids,
        "example_documents": example_documents,
        "umap_model": umap_model,
        "original_embeddings": original_embeddings,
        "original_labels": original_labels,
        "using_fallback": using_fallback
    }

def get_roberta_embeddings(text, tokenizer, model):
    """Generate RoBERTa embeddings for input text"""
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Extract embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return embedding
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        # Return a dummy embedding
        return np.random.randn(768)

def extract_document_key_terms(text, n_terms=5):
    """Extract key terms specific to the document using TF-IDF approach."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    
    # Create a custom stopwords list
    import nltk
    from nltk.corpus import stopwords
    
    try:
        stop_words = set(stopwords.words('english'))
    except:
        # If NLTK data is not available, use a basic stopword list
        stop_words = {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'with', 'as', 'by', 'on', 'at', 'from', 'this', 'that', 'it', 'was', 'be', 'are', 'were'}
    
    # Add custom stopwords for news articles
    custom_stopwords = list(stop_words)
    custom_stopwords.extend(['said', 'says', 'told', 'according', 'mr', 'would', 'could', 'also', 'may', 'one', 'two', 'new', 'year', 'years', 'time', 'will', 'first', 'last'])
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words=custom_stopwords,
        min_df=1,  # Include terms that appear at least once
        ngram_range=(1, 2)  # Include unigrams and bigrams
    )
    
    # Fit the vectorizer on the text
    # We need a list of documents for the vectorizer
    try:
        X = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        
        # Get feature scores (TF-IDF weights)
        tfidf_scores = X.toarray()[0]
        
        # Sort the features by score
        indices = np.argsort(tfidf_scores)[::-1][:n_terms]
        top_terms = [(feature_names[i], tfidf_scores[i]) for i in indices]
        
        # Extract just the terms
        terms = [term for term, score in top_terms]
        return terms
    except Exception as e:
        print(f"Error extracting key terms: {str(e)}")
        # If TF-IDF fails, fall back to simple word counting
        import re
        from collections import Counter
        
        # Simple text cleaning
        cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
        words = cleaned_text.split()
        words = [word for word in words if word not in custom_stopwords and len(word) > 3]
        
        # Count words
        word_counts = Counter(words)
        terms = [word for word, count in word_counts.most_common(n_terms)]
        return terms
    
def classify_document(text):
    """Classify document based on RoBERTa embeddings and confidence scores"""
    try:
        print(f"Classifying text: {text[:100]}...")
        
        # Generate embedding
        embedding = get_roberta_embeddings(text, models["tokenizer"], models["model"])
        print(f"Generated embedding with shape: {embedding.shape}")
        
        # Define keyword dictionaries for topic mapping
        keywords = {
            "politics": ["president", "debate", "election", "candidate", "voter", "campaign",
                        "policy", "government", "congress", "senate", "party", "democrat", 
                        "republican", "administration", "ballot", "poll", "electorate"],
            "business": ["market", "economy", "stock", "finance", "investment", "company",
                        "corporate", "trade", "industry", "economic", "business", "bank",
                        "profit", "investor", "fiscal", "commercial"],
            "sport": ["game", "player", "team", "coach", "match", "tournament", "championship",
                     "stadium", "score", "athlete", "league", "fan", "win", "season", "sport"],
            "tech": ["technology", "computer", "software", "hardware", "digital", "internet",
                    "app", "device", "data", "code", "programming", "algorithm", "tech",
                    "innovation", "startup", "computing"],
            "entertainment": ["movie", "film", "music", "celebrity", "actor", "actress",
                            "director", "show", "television", "performance", "award",
                            "concert", "festival", "entertainment", "star", "theater"]
        }
        
        # Calculate keyword-based scores for all topics
        text_lower = text.lower()
        scores = []
        
        for topic in models["topic_metadata"]:
            topic_id = topic["cluster_id"]
            category = topic["dominant_category"]
            terms = topic.get("distinctive_terms", [])
            category_keywords = keywords.get(category.lower(), [])
            color = TOPIC_COLORS.get(category.lower(), "#999999") 
            
            # Count keyword matches from both distinctive terms and keywords
            term_matches = sum(text_lower.count(term.lower()) for term in terms)
            keyword_matches = sum(text_lower.count(keyword.lower()) for keyword in category_keywords)
            
            # Compute scores with higher weight to general keywords
            term_score = term_matches / max(1, len(terms))
            keyword_score = keyword_matches / max(1, len(category_keywords)) * 2
            combined_score = term_score + keyword_score
            
            scores.append({
                "topic": topic_id,
                "label": category,
                "score": combined_score,
                "terms": terms[:5] if terms else [],
                "color": color  
            })
        
        # Try to predict using the model
        try:
            # Transform embedding and predict cluster using K-means
            cluster_id = models["predict"](np.array([embedding]))[0]
            print(f"Model-based prediction: Cluster {cluster_id}")
            
            # Add a significant boost to the model-predicted cluster
            for s in scores:
                if s["topic"] == cluster_id:
                    s["score"] += 5  # Significant boost to ensure model prediction is prioritized
        except Exception as e:
            print(f"Model-based prediction failed: {e}")
            print("Relying on keyword-based classification")
            # We'll continue with just the keyword scores
        
        # Calculate probabilities
        total_score = sum(s["score"] for s in scores) or 1
        for s in scores:
            s["probability"] = s["score"] / total_score
        
        # Sort by probability (descending)
        scores.sort(key=lambda x: x["probability"], reverse=True)
        
        # Get the highest scoring topic
        top_cluster = scores[0]["topic"]
        print(f"Highest confidence cluster: {top_cluster} ({scores[0]['label']} - {scores[0]['probability']:.2f})")
        
        # For visualization, use the position of the top_cluster
        topic_coord = next(
            (t for t in models["topic_coordinates"] if t["id"] == top_cluster), 
            {"x": 0, "y": 0}
        )
        
        # Add small random offset for visualization
        position = {
            "x": float(topic_coord["x"] + (np.random.random() - 0.5)),
            "y": float(topic_coord["y"] + (np.random.random() - 0.5)),
            "label": "Your Document",
            "color": scores[0]["color"] 
        }

        # Extract document-specific key terms
        document_terms = extract_document_key_terms(text, n_terms=5)

        # Use the highest confidence topic as the top topic
        top_topic = scores[0].copy()  
        top_topic["terms"] = document_terms  

        result = {
            "topTopic": top_topic,  
            "scores": scores,
            "position": position
        }
        
        print(f"Final classification: {result['topTopic']['label']} with {result['topTopic']['probability']:.2f} confidence")
        return result
        
    except Exception as e:
        print(f"Error classifying document: {str(e)}")
        print(traceback.format_exc())
        raise e

@app.route('/')
def index():
    """Render the main page"""
    # Make topic metadata and coordinates available to the template
    topic_metadata = models.get("topic_metadata", [])
    topic_coords = models.get("topic_coordinates", [])
    example_docs = models.get("example_documents", [])
    
    return render_template('index.html', 
                          topic_metadata=topic_metadata,
                          topic_coordinates=topic_coords,
                          example_documents=example_docs,
                          topic_colors=TOPIC_COLORS)

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

@app.route('/classify', methods=['POST'])
def classify_api():
    """API endpoint for document classification"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text']
        
        # Print detailed information for debugging
        print(f"Got classification request for text of length {len(text)}")
        
        result = classify_document(text)
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in API: {str(e)}")
        print(traceback.format_exc())   
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Load models before starting the app
    models = load_models()
    print("Starting application...")
    app.run(debug=True)