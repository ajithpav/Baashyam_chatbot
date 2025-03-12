import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import threading
import queue
from functools import lru_cache
from pathlib import Path

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Constants
WEBSITE_TEXT_PATH = "output/bashyam_website_text.txt"
COMBINED_TEXT_PATH = "output/bashyam_combined_text.txt"
CLEANED_TEXT_PATH = "output/bashyam_cleaned_text.txt"

# Create output directory if it doesn't exist
Path("output").mkdir(parents=True, exist_ok=True)

# Load models and tokenizers
model_name = "gpt2"  # Replace with your fine-tuned model when available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Predefined responses - Update these with Bashyam Group specific information
predefined_responses = {
    "who is the ceo": "The CEO of Bashyam Group is [CEO Name].",
    "what does bashyam group do": "Bashyam Group provides services in [specific services from your website].",
    "contact": "You can reach Bashyam Group at the contact information provided on our website: https://www.bashyamgroup.com/contact",
    "location": "Bashyam Group's office is located at [Location from your website].",
    "services": "Bashyam Group offers [services from your website].",
    "projects": "Bashyam Group has worked on various projects including [projects from your website]."
}

# Irrelevant question response template
irrelevant_response = "I'm sorry, but I can only provide information about Bashyam Group and its services. If you have any questions about our company, products, or services, I'd be happy to help with those."

class TextProcessor:
    """Handle text processing and response generation"""
    def __init__(self):
        self.website_text = self._load_text_file(WEBSITE_TEXT_PATH)
        self.combined_text = self._load_text_file(COMBINED_TEXT_PATH)
        self.cleaned_text = self._load_text_file(CLEANED_TEXT_PATH)
        
        self.website_embeddings = None
        self.combined_embeddings = None
        self.cleaned_embeddings = None
        
        self.website_sentences = None
        self.combined_sentences = None
        self.cleaned_sentences = None
        
        # Initialize processing
        self._prepare_embeddings()
        self.response_cache = {}

    def _load_text_file(self, file_path):
        """Load and read text file"""
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        return ""

    def _prepare_embeddings(self):
        """Prepare text embeddings for faster processing"""
        self.website_sentences = self._process_text(self.website_text)
        self.combined_sentences = self._process_text(self.combined_text)
        self.cleaned_sentences = self._process_text(self.cleaned_text)
        
        if self.website_sentences:
            self.website_embeddings = sentence_model.encode(self.website_sentences)
        if self.combined_sentences:
            self.combined_embeddings = sentence_model.encode(self.combined_sentences)
        if self.cleaned_sentences:
            self.cleaned_embeddings = sentence_model.encode(self.cleaned_sentences)

    def _process_text(self, text):
        """Process text into clean sentences"""
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        cleaned_sentences = []
        
        for sentence in sentences:
            cleaned = self._clean_sentence(sentence)
            if cleaned and len(cleaned.split()) >= 5:
                cleaned_sentences.append(cleaned)
                
        return cleaned_sentences

    def _clean_sentence(self, sentence):
        """Clean individual sentence"""
        cleaned = re.sub(r'[^\w\s.,!?-]', '', sentence)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip().capitalize()
        if cleaned and not cleaned.endswith(('.', '!', '?')):
            cleaned += '.'
        return cleaned

    @lru_cache(maxsize=1000)
    def find_relevant_response(self, query, threshold=0.6):
        """Find relevant response with caching and parallel processing"""
        query_embedding = sentence_model.encode([query])
        best_response = None
        highest_similarity = threshold

        def process_source(embeddings, sentences):
            if embeddings is not None and len(sentences) > 0:
                similarities = cosine_similarity(query_embedding, embeddings)[0]
                max_sim = np.max(similarities)
                if max_sim > highest_similarity:
                    return (max_sim, sentences[np.argmax(similarities)])
            return (highest_similarity, None)

        # Process sources in parallel
        threads = []
        results = queue.Queue()
        
        sources = [
            (self.website_embeddings, self.website_sentences),
            (self.combined_embeddings, self.combined_sentences),
            (self.cleaned_embeddings, self.cleaned_sentences)
        ]

        for embeddings, sentences in sources:
            thread = threading.Thread(
                target=lambda q, emb, sent: q.put(process_source(emb, sent)),
                args=(results, embeddings, sentences)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        while not results.empty():
            sim, response = results.get()
            if sim > highest_similarity and response:
                highest_similarity = sim
                best_response = response

        return best_response
    
    def is_relevant_question(self, query):
        """Determine if a question is relevant to Bashyam Group"""
        # List of company-related keywords
        company_keywords = [
            "bashyam", "group", "company", "services", "products", "location", 
            "contact", "project", "client", "team", "about", "history", "ceo", 
            "management", "career", "job", "opportunity", "partner", "solution"
        ]
        
        # Check if any company keyword is in the query
        query_lower = query.lower()
        for keyword in company_keywords:
            if keyword in query_lower:
                return True
                
        # Check similarity with predefined responses
        for key in predefined_responses.keys():
            if cosine_similarity(
                sentence_model.encode([query_lower]), 
                sentence_model.encode([key])
            )[0][0] > 0.7:
                return True
                
        # Check if the query has any relevance to the website content
        relevant_response = self.find_relevant_response(query, threshold=0.5)
        if relevant_response:
            return True
            
        return False

def process_chat_input(text_processor, user_input):
    """Process text input and generate response"""
    # Check if the question is relevant
    if not text_processor.is_relevant_question(user_input):
        return irrelevant_response
    
    # Check predefined responses
    for key, response in predefined_responses.items():
        if key.lower() in user_input.lower():
            return clean_response(response)

    # Get relevant response
    relevant_response = text_processor.find_relevant_response(user_input)
    if relevant_response:
        if len(relevant_response.split()) > 40:
            relevant_response = summarizer(relevant_response, max_length=40, min_length=15, do_sample=False)[0]["summary_text"]
        return clean_response(relevant_response)

    # Use model as fallback
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=500,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        do_sample=True,
        num_return_sequences=1
    )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_text = clean_response(response_text)
    
    if len(response_text.split()) > 40:
        response_text = summarizer(response_text, max_length=40, min_length=15, do_sample=False)[0]["summary_text"]
    
    return response_text

def clean_response(text):
    """Clean and format response text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    text = text.strip().capitalize()
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    return text

# Initialize text processor
text_processor = TextProcessor()

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route("/chat", methods=["POST"])
def chat():
    """Handle text chat requests"""
    try:
        data = request.json
        user_input = data.get("message", "").strip()

        if not user_input:
            return jsonify({"response": "I didn't receive any input. How can I help you with information about Bashyam Group?"})

        response = process_chat_input(text_processor, user_input)
        return jsonify({"response": response})
    
    except Exception as e:
        print(f"Chat Error: {str(e)}")
        return jsonify({
            "response": "I apologize, but I'm having trouble processing your request. Could you please rephrase your question about Bashyam Group?"
        })

# Script to scrape website content
def scrape_website():
    """
    Scrape content from Bashyam Group website and save to files
    This function should be run once to populate the text files
    """
    import requests
    from bs4 import BeautifulSoup
    
    try:
        # Get website content
        response = requests.get("https://www.bashyamgroup.com/")
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text
        text_content = soup.get_text(separator=' ', strip=True)
        
        # Save website text
        with open(WEBSITE_TEXT_PATH, "w", encoding="utf-8") as file:
            file.write(text_content)
            
        # Clean the text
        cleaned_text = ' '.join(text_content.split())
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        # Save cleaned text
        with open(CLEANED_TEXT_PATH, "w", encoding="utf-8") as file:
            file.write(cleaned_text)
            
        # Combine text (you can add more sources here)
        combined_text = cleaned_text
        
        # Save combined text
        with open(COMBINED_TEXT_PATH, "w", encoding="utf-8") as file:
            file.write(combined_text)
            
        print("Website content scraped successfully.")
        
    except Exception as e:
        print(f"Error scraping website: {str(e)}")

if __name__ == "__main__":
    # Uncomment to scrape website content (run only once)
    # scrape_website()
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=5005, debug=True)