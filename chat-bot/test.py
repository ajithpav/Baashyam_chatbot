from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import traceback
import logging
from logging.handlers import RotatingFileHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up file logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging to file with rotation
log_file = os.path.join(log_dir, "chatbot.log")
file_handler = RotatingFileHandler(log_file, maxBytes=10485760, backupCount=10)  # 10MB per file, keep 10 files max
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
file_handler.setLevel(logging.INFO)

# Configure root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Add console logging for development
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logger.addHandler(console_handler)

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.error("GOOGLE_API_KEY not found in environment variables")
    raise ValueError("GOOGLE_API_KEY is required")
    
genai.configure(api_key=api_key)

# Test API key validity and check available models
try:
    # Get available models to ensure we're using valid model names
    models = genai.list_models()
    available_models = [model.name for model in models]
    logger.info(f"Available models: {available_models}")
    logger.info("API key validation successful")
except Exception as e:
    logger.error(f"API key validation failed: {e}")
    raise ValueError(f"Invalid or inactive GOOGLE_API_KEY: {str(e)}")

# Model configuration
EMBEDDING_MODEL = "models/embedding-001"
CHAT_MODEL = "gemini-1.5-pro"

# Path to your CSV data - UPDATED to use absolute path
CSV_PATH = r"C:\Users\Ajithkumar.p\OneDrive - Droidal.com\Desktop\Baashyam-AI\chat-bot\csv\interst_report 1.csv"

# Path to your CSV data - using a relative path instead of absolute
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# CSV_PATH = os.path.join(DATA_DIR, "interst_report 1/.csv")

# Create data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    logger.info(f"Created data directory at {DATA_DIR}")

# Initialize vector store
vector_store = None

def load_and_process_csv():
    """Loads CSV data and creates a vector store with detailed error handling."""
    try:
        # Use the global CSV_PATH but don't try to modify it
        csv_file_path = CSV_PATH
        
        # Check if file exists
        if not os.path.exists(csv_file_path):
            logger.error(f"CSV file not found at path: {csv_file_path}")
            logger.info(f"Looking for CSV files in data directory: {DATA_DIR}")
            
            # List available CSV files in the data directory
            csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
            if csv_files:
                logger.info(f"Found CSV files: {csv_files}")
                # Use the first CSV file found
                alternative_path = os.path.join(DATA_DIR, csv_files[0])
                logger.info(f"Using alternative CSV file: {alternative_path}")
                csv_file_path = alternative_path
            else:
                logger.error("No CSV files found in data directory")
                return None
            
        # Load CSV with better error handling
        logger.info(f"Loading CSV from {csv_file_path}")
        try:
            df = pd.read_csv(csv_file_path, encoding='utf-8')
        except UnicodeDecodeError:
            # Try with a different encoding if utf-8 fails
            logger.warning("UTF-8 encoding failed, trying with latin-1")
            df = pd.read_csv(csv_file_path, encoding='latin-1')
        
        if df.empty:
            logger.error("CSV file is empty")
            return None
            
        # Log data shape for debugging
        logger.info(f"Loaded DataFrame with shape: {df.shape}")
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        
        # Convert DataFrame to text
        text = df.to_string(index=False)
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_text(text)
        logger.info(f"Created {len(chunks)} text chunks")
        
        # Create vector store
        logger.info("Creating embeddings and vector store...")
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        logger.info("Vector store created successfully")
        
        return vector_store
    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty or corrupted")
        return None
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing CSV data: {e}")
        logger.error(traceback.format_exc())
        return None

def get_conversational_chain():
    """Creates a conversational chain for question answering with improved prompt."""
    prompt_template = """
    You are a helpful assistant for Baashyaam Group, a real estate developer based in Chennai.
    
    Answer the question based on the context provided below. Be specific, clear, and concise.
    If the exact answer is not in the context, say "I don't have specific information about that" 
    and then provide general information about Baashyam Group that might be relevant.
    
    Format your answer professionally, using bullet points where appropriate.
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """

    try:
        # Fixed model name format and proper initialization parameters
        model = ChatGoogleGenerativeAI(
            model=CHAT_MODEL, 
            temperature=0.3,
            max_output_tokens=1024,
            convert_system_message_to_human=True  # Added to ensure prompt template is properly handled
        )
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        logger.error(f"Error creating conversation chain: {e}")
        logger.error(traceback.format_exc())
        return None

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests with improved error handling and debugging."""
    global vector_store
    
    # Log request details
    request_data = request.json
    request_id = id(request)
    logger.info(f"Request {request_id}: New chat request received")
    
    # Load vector store if not already loaded
    if vector_store is None:
        logger.info(f"Request {request_id}: Vector store not initialized, loading now...")
        vector_store = load_and_process_csv()
        if vector_store is None:
            logger.error(f"Request {request_id}: Failed to initialize vector store")
            return jsonify({
                "response": "I'm having trouble accessing the information database. Please try again later or contact support.",
                "status": "error"
            })
    
    # Get the user query from the request
    user_question = request_data.get('message', '')
    
    if not user_question:
        logger.warning(f"Request {request_id}: Empty question received")
        return jsonify({
            "response": "I didn't receive your question. Please try again.",
            "status": "error"
        })
    
    logger.info(f"Request {request_id}: Processing question: {user_question}")
    
    try:
        # Query the vector store
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        logger.info(f"Request {request_id}: Searching vector store")
        docs = vector_store.similarity_search(user_question, k=4)  # Retrieve top 4 documents
        
        if not docs:
            logger.warning(f"Request {request_id}: No relevant documents found")
            return jsonify({
                "response": "I couldn't find specific information related to your question. Please try asking something else about Baashyam Group.",
                "status": "no_docs"
            })
            
        logger.info(f"Request {request_id}: Found {len(docs)} relevant documents")
        
        # Log the document content for debugging (truncated for log readability)
        for i, doc in enumerate(docs):
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            logger.debug(f"Request {request_id}: Doc {i+1} preview: {content_preview}")
        
        # Get response from the conversational chain
        logger.info(f"Request {request_id}: Creating conversation chain")
        chain = get_conversational_chain()
        if not chain:
            logger.error(f"Request {request_id}: Failed to create conversation chain")
            return jsonify({
                "response": "I'm experiencing a technical issue. Please try again later.",
                "status": "error"
            })
        
        logger.info(f"Request {request_id}: Generating response")
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        # Log the response for debugging (truncated for log readability)
        response_text = response.get("output_text", "No response generated")
        response_preview = response_text[:100] + "..." if len(response_text) > 100 else response_text
        logger.info(f"Request {request_id}: Generated response: {response_preview}")
        
        return jsonify({
            "response": response_text,
            "status": "success"
        })
    except Exception as e:
        error_message = str(e)
        logger.error(f"Request {request_id}: Error processing query: {error_message}")
        logger.error(traceback.format_exc())
        
        # Provide a more helpful error message
        if "API" in error_message or "key" in error_message.lower():
            return jsonify({
                "response": "I'm having trouble connecting to my knowledge service. Please try again later.",
                "status": "api_error"
            })
        elif "timeout" in error_message.lower():
            return jsonify({
                "response": "The request timed out. Please try asking a simpler question.",
                "status": "timeout"
            })
        else:
            return jsonify({
                "response": "I encountered an error while processing your question. Please try again with a different question.",
                "status": "error"
            })

@app.route('/reset', methods=['POST'])
def reset():
    """Reset the vector store (useful for updating data)."""
    global vector_store
    vector_store = None
    logger.info("Vector store has been reset")
    return jsonify({"response": "Knowledge base has been reset.", "status": "success"})

# Common responses for predefined queries
@app.route('/predefined', methods=['POST'])
def predefined():
    """Handle predefined responses."""
    data = request.json
    query = data.get('message', '')
    request_id = id(request)
    logger.info(f"Request {request_id}: Predefined query received: {query}")
    
    predefined_responses = {
        "Contact Sales": "You can contact our sales team at sales@baashyam.com or call us at +91-44-2345-6789.",
        "About Baashyam": "Baashyaam Group, established over three decades ago, is a prominent real estate developer based in Chennai, Tamil Nadu. The company has a diverse portfolio that includes affordable housing, independent villas, premium living spaces, luxury residences, townships, and commercial buildings.",
        "Contact us": "You can reach us at info@baashyam.com or visit our office at 123 Anna Salai, Chennai, Tamil Nadu. Our customer service number is +91-44-2345-6789.",
        "Projects": "Baashyaam Group has several ongoing projects in Chennai including Baashyaam Pinnacle, Baashyaam Harmony, and Baashyaam Towers. Our projects range from affordable housing to luxury apartments.",
        "Offers": "We currently have special festive season offers including 5% discount on booking amount and free modular kitchen for selected properties. Contact our sales team for more details."
    }
    
    response = predefined_responses.get(query, "I don't have a predefined answer for that query. Please ask something else.")
    logger.info(f"Request {request_id}: Returned predefined response for '{query}'")
    return jsonify({"response": response, "status": "success"})

# Add a new route to check application health and debug CSV loading
@app.route('/health', methods=['GET'])
def health_check():
    """Check if the application is running and the vector store is loaded."""
    global vector_store
    vector_store_status = "loaded" if vector_store is not None else "not_loaded"
    
    # Additional diagnostic information
    csv_exists = os.path.exists(CSV_PATH)
    csv_size = os.path.getsize(CSV_PATH) if csv_exists else 0
    csv_info = {
        "path": CSV_PATH,
        "exists": csv_exists,
        "size_bytes": csv_size
    }
    
    # Test API key (limited check)
    api_status = "valid"
    available_models = []
    try:
        models = genai.list_models()
        available_models = [model.name for model in models]
    except Exception as e:
        api_status = f"error: {str(e)}"
    
    logger.info(f"Health check: Application is running, vector store is {vector_store_status}")
    return jsonify({
        "status": "healthy",
        "vector_store": vector_store_status,
        "csv": csv_info,
        "api_key": api_status,
        "models": {
            "embedding": EMBEDDING_MODEL,
            "chat": CHAT_MODEL,
            "available_models": available_models
        }
    })

# New debug endpoint to test CSV loading directly
@app.route('/debug/load-csv', methods=['GET'])
def debug_load_csv():
    """Attempt to load CSV and return diagnostic information."""
    try:
        # Check file existence
        csv_exists = os.path.exists(CSV_PATH)
        if not csv_exists:
            return jsonify({
                "status": "error",
                "message": f"CSV file not found at {CSV_PATH}",
                "data_dir_exists": os.path.exists(DATA_DIR),
                "available_files": os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else []
            })
            
        # Try loading with pandas
        try:
            df = pd.read_csv(CSV_PATH, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(CSV_PATH, encoding='latin-1')
        
        return jsonify({
            "status": "success",
            "message": "CSV loaded successfully",
            "rows": len(df),
            "columns": df.columns.tolist(),
            "preview": df.head(3).to_dict(orient="records")
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error loading CSV: {str(e)}",
            "traceback": traceback.format_exc()
        })

if __name__ == '__main__':
    try:
        # Create required directories
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Print startup information
        logger.info(f"Starting application with:")
        logger.info(f"- CSV path: {CSV_PATH}")
        logger.info(f"- Data directory: {DATA_DIR}")
        logger.info(f"- Embedding model: {EMBEDDING_MODEL}")
        logger.info(f"- Chat model: {CHAT_MODEL}")
        
        # Initialize the vector store on startup
        logger.info("Initializing vector store on startup...")
        vector_store = load_and_process_csv()
        if vector_store is None:
            logger.warning("Vector store could not be initialized on startup")
        else:
            logger.info("Vector store initialized successfully")
            
        app.run(debug=True, port=5005)
    except Exception as e:
        logger.critical(f"Failed to start application: {e}")
        logger.critical(traceback.format_exc())
