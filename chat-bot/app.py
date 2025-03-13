import os
import re
import pandas as pd
import numpy as np
import threading
import queue
from functools import lru_cache
from pathlib import Path
import json

# Flask imports
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# NLP and ML imports
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# PDF processing
import PyPDF2

# Excel processing is handled by pandas

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Constants
DATA_DIR = "data"
OUTPUT_DIR = "output"
WEBSITE_TEXT_PATH = f"{OUTPUT_DIR}/bashyam_website_text.txt"
COMBINED_TEXT_PATH = f"{OUTPUT_DIR}/bashyam_combined_text.txt"
CLEANED_TEXT_PATH = f"{OUTPUT_DIR}/bashyam_cleaned_text.txt"
TRAINING_DATA_PATH = f"{OUTPUT_DIR}/bashyam_training_data.json"

# Create necessary directories
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Load models and tokenizers (using smaller models for efficiency)
model_name = "distilgpt2"  # Replace with your fine-tuned model when available
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Improved predefined responses with clear structure
predefined_responses = {
    # Company information
    "about": "Bashyam Group is a well-established company that has been delivering quality services and products. We pride ourselves on our commitment to excellence and customer satisfaction.",
    "about bashyam": "Bashyam Group is a well-established company that has been delivering quality services and products. We pride ourselves on our commitment to excellence and customer satisfaction.",
    "bashyam group": "Bashyam Group is a well-established company that has been delivering quality services and products. We pride ourselves on our commitment to excellence and customer satisfaction.",
    "who is the ceo": "To know more about our leadership team, please visit our website at https://www.bashyamgroup.com/about-us or contact us directly.",
    
    # Contact information
    "contact": "You can contact Bashyam Group through our website contact form at https://www.bashyamgroup.com/contact or call us at our office. We'll be happy to assist you.",
    "contact us": "You can contact Bashyam Group through our website contact form at https://www.bashyamgroup.com/contact or call us at our office. We'll be happy to assist you.",
    "email": "You can reach out to us via email through our contact page at https://www.bashyamgroup.com/contact.",
    "phone": "For phone inquiries, please visit our contact page at https://www.bashyamgroup.com/contact to find the appropriate number for your needs.",
    
    # Services
    "services": "Bashyam Group offers a wide range of services. Please visit our website at https://www.bashyamgroup.com/services for more detailed information about what we offer.",
    "products": "For information about our products, please visit our website at https://www.bashyamgroup.com/products or contact us directly.",
    
    # Location
    "location": "To find Bashyam Group's office locations, please visit our website at https://www.bashyamgroup.com/contact.",
    "address": "For our office addresses, please check our contact page at https://www.bashyamgroup.com/contact.",
    
    # General
    "hi": "Hello! I'm the Bashyam Group virtual assistant. How can I help you today?",
    "hello": "Hello! I'm the Bashyam Group virtual assistant. How can I help you today?",
    "help": "I can provide information about Bashyam Group, our services, contact details, and more. How can I assist you today?"
}

# Improved irrelevant question response with helpful guidance
irrelevant_response = "I'm focused on providing information about Bashyam Group and our services. If you have questions about our company, products, services, or need assistance, I'd be happy to help. For other topics, you might want to try a general search engine."

# Prompt templates for different types of queries
prompt_templates = {
    "general_info": "Provide information about Bashyam Group's {topic}.",
    "contact": "Share contact details for Bashyam Group regarding {topic}.",
    "service_info": "Explain the {service} service offered by Bashyam Group.",
    "product_info": "Describe the {product} product from Bashyam Group.",
    "project_info": "Give details about Bashyam Group's {project} project.",
    "customer_info": "Provide details about customer {customer_id}.",
    "irrelevant": "The query is not directly related to Bashyam Group. Provide a polite response directing to relevant information."
}

class DataProcessor:
    """Process and manage various data sources"""
    
    def __init__(self):
        """Initialize data processor"""
        self.training_data = self._load_training_data()
    
    def _load_training_data(self):
        """Load existing training data if available"""
        if os.path.exists(TRAINING_DATA_PATH):
            try:
                with open(TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {"qa_pairs": [], "entities": {}}
        return {"qa_pairs": [], "entities": {}}
    
    def _save_training_data(self):
        """Save training data to file"""
        with open(TRAINING_DATA_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, indent=4)
    
    def add_qa_pair(self, question, answer):
        """Add a question-answer pair to training data"""
        self.training_data["qa_pairs"].append({"question": question, "answer": answer})
        self._save_training_data()
    
    def process_pdf(self, pdf_path):
        """Extract text from PDF file"""
        extracted_text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    extracted_text += page.extract_text() + "\n\n"
            
            # Clean and save the extracted text
            cleaned_text = self._clean_text(extracted_text)
            output_path = f"{OUTPUT_DIR}/{os.path.basename(pdf_path)}.txt"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            # Update combined text
            self._append_to_combined_text(cleaned_text)
            
            return True, f"Successfully processed {pdf_path}"
        except Exception as e:
            return False, f"Error processing PDF: {str(e)}"
    
    def process_excel(self, excel_path):
        """Extract data from Excel/CSV file with improved customer data handling"""
        try:
            # Read the file (works for both CSV and Excel)
            if excel_path.endswith('.csv'):
                df = pd.read_csv(excel_path)
            else:
                df = pd.read_excel(excel_path)
            
            # Display column names for debugging
            print(f"CSV columns found: {df.columns.tolist()}")
            
            # Extract Q&A pairs if columns match expected format
            qa_added = False
            entity_added = False
            
            # Check for question-answer columns
            if 'question' in df.columns and 'answer' in df.columns:
                for _, row in df.iterrows():
                    if not pd.isna(row['question']) and not pd.isna(row['answer']):
                        self.add_qa_pair(row['question'], row['answer'])
                        qa_added = True
            
            # Check for entity information (like products, services, etc.)
            if 'entity_type' in df.columns and 'entity_name' in df.columns and 'description' in df.columns:
                for _, row in df.iterrows():
                    if not pd.isna(row['entity_type']) and not pd.isna(row['entity_name']) and not pd.isna(row['description']):
                        entity_type = row['entity_type'].lower()
                        if entity_type not in self.training_data["entities"]:
                            self.training_data["entities"][entity_type] = {}
                        
                        self.training_data["entities"][entity_type][row['entity_name']] = row['description']
                        entity_added = True
            
            # Handle customer data specifically - try to identify relevant columns
            customer_id_col = None
            customer_name_col = None
            due_date_col = None
            total_interest_col = None
            
            # Try to identify columns by common names
            for col in df.columns:
                col_lower = str(col).lower()
                if any(term in col_lower for term in ['id', 'cust id', 'customerid', 'customer_id']):
                    customer_id_col = col
                elif any(term in col_lower for term in ['name', 'customer name', 'customername']):
                    customer_name_col = col
                elif any(term in col_lower for term in ['due date', 'duedate']):
                    due_date_col = col
                elif any(term in col_lower for term in ['interest', 'total interest']):
                    total_interest_col = col
            
            # If customer ID column is found, process customer data
            if customer_id_col:
                # Initialize customer entity type if not exists
                if "customer" not in self.training_data["entities"]:
                    self.training_data["entities"]["customer"] = {}
                
                # Process each row
                for _, row in df.iterrows():
                    # Ensure customer ID is treated as string
                    cust_id = str(row[customer_id_col]) if not pd.isna(row[customer_id_col]) else None
                    
                    if cust_id:
                        # Create description based on available data
                        details = []
                        
                        if customer_name_col and not pd.isna(row[customer_name_col]):
                            details.append(f"Name: {row[customer_name_col]}")
                        
                        if due_date_col and not pd.isna(row[due_date_col]):
                            details.append(f"Due Date: {row[due_date_col]}")
                        
                        if total_interest_col and not pd.isna(row[total_interest_col]):
                            details.append(f"Total Interest: {row[total_interest_col]}")
                        
                        # Add any other columns as details
                        for col in df.columns:
                            if col not in [customer_id_col, customer_name_col, due_date_col, total_interest_col] and not pd.isna(row[col]):
                                details.append(f"{col}: {row[col]}")
                        
                        # Create description
                        description = f"Customer ID: {cust_id}. " + ". ".join(details)
                        
                        # Add to entities
                        self.training_data["entities"]["customer"][cust_id] = description
                        
                        # Also add explicit QA pairs for common questions about this customer
                        if customer_name_col and not pd.isna(row[customer_name_col]):
                            name = row[customer_name_col]
                            self.add_qa_pair(f"What is the customer name for {cust_id}?", 
                                            f"The customer name for {cust_id} is {name}.")
                            self.add_qa_pair(f"{cust_id} customer name", 
                                            f"The customer name for {cust_id} is {name}.")
                            self.add_qa_pair(f"Who is {cust_id}?", 
                                            f"{cust_id} refers to customer {name}.")
                            
                            # New QA pairs for name-based lookup
                            
                            self.add_qa_pair(f"customer named {name}", f"Customer {name} has ID {cust_id}. {description}")
                            self.add_qa_pair(f"information for customer {name}", f"Customer {name} (ID: {cust_id}) details: {description}")
                            self.add_qa_pair(f"interest for {name}", 
                                             f"The total interest for customer {name} is {row[total_interest_col] if total_interest_col and not pd.isna(row[total_interest_col]) else 'not available'}.")
                        
                        if due_date_col and not pd.isna(row[due_date_col]):
                            due_date = row[due_date_col]
                            self.add_qa_pair(f"What is the due date for {cust_id}?", 
                                            f"The due date for customer {cust_id} is {due_date}.")
                            self.add_qa_pair(f"{cust_id} due date", 
                                            f"The due date for customer {cust_id} is {due_date}.")
                            self.add_qa_pair(f"When is {cust_id} due?", 
                                            f"The due date for customer {cust_id} is {due_date}.")
                        
                        if total_interest_col and not pd.isna(row[total_interest_col]):
                            interest = row[total_interest_col]
                            self.add_qa_pair(f"What is the total interest for {cust_id}?", 
                                            f"The total interest for customer {cust_id} is {interest}.")
                            self.add_qa_pair(f"{cust_id} total interest", 
                                            f"The total interest for customer {cust_id} is {interest}.")
                
                # Add general questions about total interest across all customers
                if total_interest_col:
                    total_interest_sum = df[total_interest_col].sum() if pd.api.types.is_numeric_dtype(df[total_interest_col]) else "not calculable"
                    self.add_qa_pair("What is the total interest calculated?", 
                                    f"The total interest calculated across all customers is {total_interest_sum}.")
                    self.add_qa_pair("Show me the total interest", 
                                    f"The total interest calculated across all customers is {total_interest_sum}.")
                    self.add_qa_pair("Sum of all interest", 
                                    f"The sum of all interest payments is {total_interest_sum}.")
                
                entity_added = True
            
            # Save any changes
            if qa_added or entity_added:
                self._save_training_data()
                
            # Create a text representation to add to combined text
            text_content = df.to_string(index=False)
            output_path = f"{OUTPUT_DIR}/{os.path.basename(excel_path)}.txt"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            # Update combined text
            self._append_to_combined_text(text_content)
            
            return True, f"Successfully processed {excel_path}"
        except Exception as e:
            print(f"Error processing Excel/CSV: {str(e)}")
            return False, f"Error processing Excel/CSV: {str(e)}"
    
    def _clean_text(self, text):
        """Clean extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters except punctuation
        text = re.sub(r'[^\w\s.,!?:;()\-\'"]', '', text)
        return text.strip()
    
    def _append_to_combined_text(self, text):
        """Append new text to combined text file"""
        # Read existing combined text
        existing_text = ""
        if os.path.exists(COMBINED_TEXT_PATH):
            with open(COMBINED_TEXT_PATH, 'r', encoding='utf-8') as f:
                existing_text = f.read()
        
        # Append new text
        with open(COMBINED_TEXT_PATH, 'w', encoding='utf-8') as f:
            if existing_text:
                f.write(f"{existing_text}\n\n{text}")
            else:
                f.write(text)

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
        
        # Load training data
        self.data_processor = DataProcessor()
        
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

    def reload_embeddings(self):
        """Reload text and embeddings when data changes"""
        self.combined_text = self._load_text_file(COMBINED_TEXT_PATH)
        self._prepare_embeddings()

    def _process_text(self, text):
        """Process text into clean sentences"""
        if not text:
            return []
            
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 20]
        cleaned_sentences = []
        
        for sentence in sentences:
            cleaned = self._clean_sentence(sentence)
            if cleaned and len(cleaned.split()) >= 5:
                cleaned_sentences.append(cleaned)
                
        return cleaned_sentences

    def _clean_sentence(self, sentence):
        """Clean individual sentence"""
        cleaned = re.sub(r'[^\w\s.,!?:;()\-\'"]', '', sentence)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip().capitalize()
        if cleaned and not cleaned.endswith(('.', '!', '?')):
            cleaned += '.'
        return cleaned

    def get_query_type(self, query):
        """Determine the type of query to select appropriate template"""
        query_lower = query.lower()
        
        # Check for customer ID pattern
        customer_id_match = re.search(r'\b[a-z]{3,6}\d{5,8}\b', query_lower)
        if customer_id_match or any(term in query_lower for term in ['customer', 'cust', 'client', 'cros']):
            return "customer_info"
        
        # Check for contact related queries
        if any(word in query_lower for word in ['contact', 'email', 'phone', 'call', 'reach']):
            return "contact"
            
        # Check for service related queries
        if any(word in query_lower for word in ['service', 'offer', 'provide', 'solution']):
            return "service_info"
            
        # Check for product related queries
        if any(word in query_lower for word in ['product', 'item', 'sell', 'buy', 'purchase']):
            return "product_info"
            
        # Check for project related queries
        if any(word in query_lower for word in ['project', 'work', 'portfolio', 'client', 'case study']):
            return "project_info"
            
        # Check for interest-related queries
        if any(word in query_lower for word in ['interest', 'due', 'payment', 'amount']):
            return "customer_info"
            
        # Check if it's irrelevant
        if not self.is_relevant_question(query):
            return "irrelevant"
            
        # Default to general info
        return "general_info"

    def search_training_data(self, query):
        """Search for answer in training data"""
        query_lower = query.lower()
    
    # Exact match in QA pairs
        for qa_pair in self.data_processor.training_data["qa_pairs"]:
            if query_lower == qa_pair["question"].lower():
                return qa_pair["answer"]
    
    # Extract customer ID if present
        customer_id_match = re.search(r'\b[a-z]{3,6}\d{5,8}\b', query_lower)
        customer_id = customer_id_match.group(0) if customer_id_match else None
    
    # Extract customer name if present
        customer_name_match = re.search(r'(?:name|customer|client)\s+(?:is|for|of|named)\s+([a-zA-Z\s\.]+)', query_lower)
        customer_name = customer_name_match.group(1).strip() if customer_name_match else None
    
    # Check for name in entity descriptions if name query
        if customer_name and "customer" in self.data_processor.training_data["entities"]:
            for cust_id, description in self.data_processor.training_data["entities"]["customer"].items():
                if customer_name.lower() in description.lower():
                # Extract specific details based on query
                    if 'due date' in query_lower:
                        due_date_match = re.search(r'Due Date: ([^.]+)', description)
                        if due_date_match:
                            return f"The due date for customer {customer_name} is {due_date_match.group(1)}."
                elif 'interest' in query_lower:
                    interest_match = re.search(r'Total Interest: ([^.]+)', description) 
                    if interest_match:
                        return f"The total interest for customer {customer_name} is {interest_match.group(1)}."
                else:
                    # Return full customer info
                    return f"Customer information for {customer_name}: {description}"
    
    # If query mentions customer ID, look for specific customer entity
        if customer_id and "customer" in self.data_processor.training_data["entities"]:
        # Try exact match
            if customer_id in self.data_processor.training_data["entities"]["customer"]:
            # Check if asking about specific attribute
                if 'name' in query_lower or 'who is' in query_lower:
                    customer_info = self.data_processor.training_data["entities"]["customer"][customer_id]
                    name_match = re.search(r'Name: ([^.]+)', customer_info)
                if name_match:
                    return f"The customer name for {customer_id} is {name_match.group(1)}."
            elif 'due date' in query_lower or 'when' in query_lower:
                customer_info = self.data_processor.training_data["entities"]["customer"][customer_id]
                due_date_match = re.search(r'Due Date: ([^.]+)', customer_info)
                if due_date_match:
                    return f"The due date for customer {customer_id} is {due_date_match.group(1)}."
            elif 'interest' in query_lower:
                customer_info = self.data_processor.training_data["entities"]["customer"][customer_id]
                interest_match = re.search(r'Total Interest: ([^.]+)', customer_info)
                if interest_match:
                    return f"The total interest for customer {customer_id} is {interest_match.group(1)}."
            else:
                # Return full customer info
                return self.data_processor.training_data["entities"]["customer"][customer_id]
        
        # Check for total interest calculation query
        if 'total interest' in query_lower or 'interest calculated' in query_lower:
            for qa_pair in self.data_processor.training_data["qa_pairs"]:
                if 'total interest calculated' in qa_pair["question"].lower():
                    return qa_pair["answer"]
        
        # Semantic similarity in QA pairs
        query_embedding = sentence_model.encode([query])
        best_score = 0.7  # Lowered threshold for better matches
        best_answer = None
        
        for qa_pair in self.data_processor.training_data["qa_pairs"]:
            question_embedding = sentence_model.encode([qa_pair["question"]])
            similarity = cosine_similarity(query_embedding, question_embedding)[0][0]
            
            if similarity > best_score:
                best_score = similarity
                best_answer = qa_pair["answer"]
        
        if best_answer:
            return best_answer
            
        # General entity search
        for entity_type, entities in self.data_processor.training_data["entities"].items():
            for entity_name, description in entities.items():
                if entity_name.lower() in query_lower:
                    return f"{entity_name}: {description}"
        
        return None

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
        """Determine if a question is relevant to Bashyam Group or customer data"""
        query_lower = query.lower()
        
        # Check for customer ID pattern
        customer_id_match = re.search(r'\b[a-z]{3,6}\d{5,8}\b', query_lower)
        # Enhanced customer name matching
        customer_name_match = re.search(r'(?:simon|kurian|a\.c\.|customer named|client named)', query_lower)
        if customer_id_match or customer_name_match:
            return True
        
        # Check for customer-related keywords
        customer_keywords = ["customer", "cust", "client", "cros", "interest", "due date", "payment"]
        for keyword in customer_keywords:
            if keyword in query_lower:
                return True
        
        # List of company-related keywords
        company_keywords = [
            "bashyam", "baashyam", "group", "company", "services", "products", "location", 
            "contact", "project", "client", "team", "about", "history", "ceo", 
            "management", "career", "job", "opportunity", "partner", "solution", "office"
        ]
        
        # Check if any company keyword is in the query
        for keyword in company_keywords:
            if keyword in query_lower:
                return True
                
        # Check similarity with predefined responses
        for key in predefined_responses.keys():
            if key in query_lower:
                return True
                
        # Check if the query has any relevance to the website content
        relevant_response = self.find_relevant_response(query, threshold=0.4)
        if relevant_response:
            return True
            
        return False

def process_chat_input(text_processor, user_input):
    """Process text input and generate response with improved handling"""
    user_input = user_input.strip()
    
    if not user_input:
        return "How can I help you with information about Bashyam Group today?"
    
    # Check for customer queries first
    customer_id_match = re.search(r'\b[a-z]{3,6}\d{5,8}\b', user_input.lower())
    # Add customer name matching
    customer_name_match = re.search(r'(?:name|customer|client)\s+(?:is|for|of|named)\s+([a-zA-Z\s\.]+)', user_input.lower())
    
    if customer_id_match or customer_name_match:
        #Try id first march
        if customer_id_match:
            customer_id = customer_id_match.group(0)
            training_response = text_processor.search_training_data(user_input)
        if training_response:
            return clean_response(training_response)
        
         # Then try name match
        if customer_name_match:
            customer_name = customer_name_match.group(1).strip()
            # Create a query that searches for the customer by name
            name_query = f"customer name {customer_name}"
            training_response = text_processor.search_training_data(name_query)
            if training_response:
                return clean_response(training_response)
            
            # If no match found
        search_term = customer_id_match.group(0) if customer_id_match else customer_name_match.group(1)
        return f"I don't have specific information about customer {search_term}. Please provide more details or ask about a different customer."
    
    # Check for interest-related queries
    interest_keywords = ["interest", "total interest", "interest calculated", "interest amount"]
    if any(keyword in user_input.lower() for keyword in interest_keywords):
        training_response = text_processor.search_training_data(user_input)
        if training_response:
            return clean_response(training_response)
    
    # Check training data next
    training_response = text_processor.search_training_data(user_input)
    if training_response:
        return clean_response(training_response)
    
    # Check for exact matches in predefined responses
    for key, response in predefined_responses.items():
        if user_input.lower() == key.lower():
            return clean_response(response)
    
    # Check for partial matches in predefined responses
    for key, response in predefined_responses.items():
        if key.lower() in user_input.lower() or user_input.lower() in key.lower():
            return clean_response(response)
    
    # Check if query is irrelevant
    if not text_processor.is_relevant_question(user_input):
        return irrelevant_response
    
    # Get relevant response
    relevant_response = text_processor.find_relevant_response(user_input)
    if relevant_response:
        if len(relevant_response.split()) > 40:
            relevant_response = summarizer(relevant_response, max_length=40, min_length=15, do_sample=False)[0]["summary_text"]
        return clean_response(relevant_response)

    # Use model as fallback with appropriate prompt template
    query_type = text_processor.get_query_type(user_input)
    prompt_template = prompt_templates.get(query_type, prompt_templates["general_info"])
    
    # Extract topic from user query
    # Simple extraction: take the last few words
    topic = " ".join(user_input.split()[-3:]) if len(user_input.split()) > 3 else user_input
    
    # Format the prompt
    formatted_prompt = prompt_template.format(
        topic=topic,
        service=topic,
        product=topic,
        project=topic,
        customer_id=customer_id_match.group(0) if customer_id_match else "unknown"
    )
    
    # Generate response
    combined_input = f"{formatted_prompt} Query: {user_input}"
    inputs = tokenizer(combined_input, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        temperature=0.3,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        do_sample=True,
        num_return_sequences=1
    )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the response if it appears
    response_text = response_text.replace(formatted_prompt, "").replace(f"Query: {user_input}", "")
    response_text = clean_response(response_text)
    
    # If response is too long, summarize it
    if len(response_text.split()) > 40:
        response_text = summarizer(response_text, max_length=40, min_length=15, do_sample=False)[0]["summary_text"]
    
    # Fall back to a safe response if generated text is inappropriate or too short
    # if len(response_text.split
    
    # Fall back to a safe response if generated text is inappropriate or too short
    if len(response_text.split()) < 3:
        response_text = "I'd be happy to help with your inquiry about Bashyam Group. Could you please provide more details about what you'd like to know?"
    
    return clean_response(response_text)

def clean_response(text):
    """Clean and format response text"""
    if not text:
        return "I'd be happy to help with information about Bashyam Group. Could you please provide more details?"
        
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = text.strip()
    
    # Ensure first letter is capitalized
    if text and len(text) > 0:
        text = text[0].upper() + text[1:]
    
    # Ensure there's proper punctuation at the end
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

@app.route("/upload-training-data", methods=["POST"])
def upload_training_data():
    """Handle file uploads for training data"""
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file part"})
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"})
            
        # Save the file
        file_path = os.path.join(DATA_DIR, file.filename)
        file.save(file_path)
        
        # Process the file based on its type
        success = False
        message = ""
        
        if file.filename.endswith('.pdf'):
            success, message = text_processor.data_processor.process_pdf(file_path)
        elif file.filename.endswith(('.xls', '.xlsx', '.csv')):
            success, message = text_processor.data_processor.process_excel(file_path)
        else:
            message = "Unsupported file type. Please upload PDF, Excel or CSV files."
            
        # Reload embeddings if successful
        if success:
            text_processor.reload_embeddings()
            
        return jsonify({
            "status": "success" if success else "error",
            "message": message
        })
        
    except Exception as e:
        print(f"Upload Error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error processing file: {str(e)}"
        })

@app.route("/add-qa", methods=["POST"])
def add_qa():
    """Add a question-answer pair to training data"""
    try:
        data = request.json
        question = data.get("question", "").strip()
        answer = data.get("answer", "").strip()
        
        if not question or not answer:
            return jsonify({
                "status": "error", 
                "message": "Both question and answer are required"
            })
            
        text_processor.data_processor.add_qa_pair(question, answer)
        
        return jsonify({
            "status": "success",
            "message": "Question-answer pair added successfully"
        })
        
    except Exception as e:
        print(f"Add QA Error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error adding QA pair: {str(e)}"
        })


@app.route("/check-interest-data", methods=["GET"])
def check_interest_data():
    """Check if interest report data is loaded"""
    try:
        # Check if we have entities or QA pairs related to interest
        entity_count = 0
        qa_count = 0
        
        # Count entities related to interest
        for entity_type, entities in text_processor.data_processor.training_data["entities"].items():
            entity_count += len(entities)
        
        # Count QA pairs (you might want to filter only interest-related ones)
        qa_count = len(text_processor.data_processor.training_data["qa_pairs"])
        
        return jsonify({
            "status": "success",
            "data": {
                "entity_count": entity_count,
                "qa_count": qa_count,
                "interest_data_loaded": entity_count > 0 or qa_count > 0
            }
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error checking interest data: {str(e)}"
        })



def load_interest_report_csv():
    """Load and process the interest report CSV file"""
    CSV_FILE_PATH = r"C:\Users\Ajithkumar.p\OneDrive - Droidal.com\Desktop\Baashyam-AI\chat-bot\csv\interst_report 1.csv"
    
    try:
        # Check if file exists
        if not os.path.exists(CSV_FILE_PATH):
            print(f"Error: CSV file not found at {CSV_FILE_PATH}")
            return False, "CSV file not found"
            
        # Process the CSV file using the existing method
        success, message = text_processor.data_processor.process_excel(CSV_FILE_PATH)
        
        # Reload embeddings if successful
        if success:
            text_processor.reload_embeddings()
            print("Successfully loaded interest report CSV and updated embeddings")
        else:
            print(f"Error processing CSV: {message}")
            
        return success, message
        
    except Exception as e:
        print(f"Error loading interest report CSV: {str(e)}")
        return False, f"Error: {str(e)}"

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
        # Load interest report CSV at startup
    print("Loading interest report CSV...")
    load_interest_report_csv()
    # Uncomment to scrape website content (run only once)
    # scrape_website()
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=5005, debug=True)
