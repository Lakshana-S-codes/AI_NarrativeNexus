import streamlit as st
import pandas as pd
import os
import json
import uuid
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    st.warning("⚠️ python-docx not installed. Word documents (.docx) won't be supported.")

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("⚠️ pdfplumber not installed. PDF files won't be supported.")

try:
    import praw
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False
    st.warning("⚠️ praw not installed. Reddit functionality won't be supported.")


load_dotenv()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
DATA_FILE = os.path.join(DATA_DIR, "data_store.json")


reddit = None
if REDDIT_AVAILABLE and REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT:
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
    except Exception as e:
        st.error(f"❌ Error initializing Reddit client: {e}")


def load_data():
    """Load existing data from JSON file"""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.warning("⚠️ Data file corrupted. Starting fresh.")
            return []
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return []
    return []


def save_single_data(new_record):
    """Save new record to JSON file"""
    try:
        data = load_data()
        data.append(new_record)
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        st.error(f"Error saving data: {e}")
        return False


def save_data_export(records, format_choice):
    """Save collected data as CSV or JSON for export"""
    try:
        if format_choice == "CSV":
            df = pd.json_normalize(records)
            df.to_csv("output_data.csv", index=False, encoding="utf-8")
            return "✅ Data exported to output_data.csv"
        else:
            with open("output_data.json", "w", encoding="utf-8") as f:
                json.dump(records, f, indent=4, ensure_ascii=False)
            return "✅ Data exported to output_data.json"
    except Exception as e:
        return f"❌ Error exporting data: {e}"


def read_txt(file):
    """Read text file"""
    try:
        return file.read().decode("utf-8")
    except UnicodeDecodeError:
        file.seek(0)
        try:
            return file.read().decode("latin-1")
        except Exception as e:
            raise Exception(f"Could not decode text file: {e}")


def read_csv(file):
    """Read CSV file and convert to string"""
    try:
        df = pd.read_csv(file)
        return df.to_string()
    except Exception as e:
        raise Exception(f"Error reading CSV: {e}")


def read_docx(file):
    """Read Word document"""
    if not DOCX_AVAILABLE:
        raise Exception("python-docx not installed. Install with: pip install python-docx")
    
    try:
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        raise Exception(f"Error reading Word document: {e}")


def read_pdf(file):
    """Read PDF file"""
    if not PDF_AVAILABLE:
        raise Exception("pdfplumber not installed. Install with: pip install pdfplumber")
    
    try:
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text if text.strip() else "No text could be extracted from this PDF."
    except Exception as e:
        raise Exception(f"Error reading PDF: {e}")


def fetch_reddit_post(url):
    """Fetch a single Reddit post given its URL"""
    if not reddit:
        raise Exception("Reddit client not initialized. Check your API credentials.")
    
    try:
        submission = reddit.submission(url=url)
        record = {
            "id": str(uuid.uuid4()),
            "source": "reddit",
            "author": submission.author.name if submission.author else "unknown",
            "timestamp": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).isoformat(),
            "text": (submission.title or "") + "\n" + (submission.selftext or ""),
            "metadata": {
                "language": "en",
                "likes": submission.score,
                "rating": None,
                "url": url,
                "subreddit": submission.subreddit.display_name,
                "num_comments": submission.num_comments
            }
        }
        return record
    except Exception as e:
        raise Exception(f"Error fetching Reddit post: {e}")


def fetch_news(query):
    """Fetch the first News article from NewsAPI matching a query"""
    if not NEWS_API_KEY:
        raise Exception("News API key not found. Check your .env file.")
    
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&sortBy=publishedAt&pageSize=1"
        response = requests.get(url).json()

        if "articles" not in response or len(response["articles"]) == 0:
            return None

        article = response["articles"][0]
        record = {
            "id": str(uuid.uuid4()),
            "source": "news",
            "author": article.get("author") or "unknown",
            "timestamp": article.get("publishedAt"),
            "text": (article.get("title") or "") + "\n" + (article.get("description") or ""),
            "metadata": {
                "language": article.get("language", "en"),
                "likes": None,
                "rating": None,
                "url": article.get("url"),
                "source_name": article.get("source", {}).get("name", "unknown")
            }
        }
        return record
    except Exception as e:
        raise Exception(f"Error fetching news: {e}")


def create_file_record(filename, source_type, file_type, content):
    """Create a data record for file uploads"""
    return {
        "id": str(uuid.uuid4()),
        "source": "file",
        "author": "user_upload",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": content,
        "metadata": {
            "filename": filename,
            "source_type": source_type,
            "file_type": file_type,
            "content_length": len(content),
            "language": "en"
        }
    }


st.title("📊 NarrativeNexus Data Collector")
st.write("Comprehensive data collection from files, Reddit posts, and news articles.")


tab1, tab2, tab3 = st.tabs(["📄 File Upload", "🔗 Reddit Posts", "📰 News Articles"])


with tab1:
    st.header("📄 File Upload")
    st.write("Upload text files for analysis and storage.")
    

    supported_types = ["txt", "csv"]
    if DOCX_AVAILABLE:
        supported_types.append("docx")
    if PDF_AVAILABLE:
        supported_types.append("pdf")

    uploaded_file = st.file_uploader(
        f"Upload a file ({', '.join(['.' + t for t in supported_types])})",
        type=supported_types
    )

    text_input = st.text_area("Or paste text directly here:", height=150)

    if st.button("Process File/Text", key="file_process"):
        try:
            content = None
            filename = None
            file_type = None
            
        
            if uploaded_file is not None:
                filename = uploaded_file.name
                file_extension = os.path.splitext(uploaded_file.name)[1].lower().replace(".", "")
                file_type = file_extension
                
                if file_extension == "txt":
                    content = read_txt(uploaded_file)
                elif file_extension == "csv":
                    content = read_csv(uploaded_file)
                elif file_extension == "docx":
                    content = read_docx(uploaded_file)
                elif file_extension == "pdf":
                    content = read_pdf(uploaded_file)
                    
     
            elif text_input.strip():
                content = text_input
                filename = "pasted_text"
                file_type = "text"

            if content and content.strip():
                record = create_file_record(filename, "file_upload", file_type, content)
                if save_single_data(record):
                    st.success(f"✅ Content saved successfully! (ID: {record['id']})")
                    
           
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Content Length", f"{len(content):,} chars")
                    with col2:
                        st.metric("Type", file_type.upper())
                    with col3:
                        st.metric("Record ID", record['id'][:8] + "...")
                    
           
                    with st.expander("Preview Content"):
                        preview_length = 500
                        if len(content) > preview_length:
                            st.write(content[:preview_length] + "...")
                            st.info(f"Showing first {preview_length} characters of {len(content):,} total")
                        else:
                            st.write(content)
            else:
                st.warning("⚠️ Please upload a file or paste some text.")
                
        except Exception as e:
            st.error(f"❌ Error processing content: {e}")


with tab2:
    st.header("🔗 Reddit Post Collector")
    st.write("Fetch Reddit posts by URL.")
    
    if not REDDIT_AVAILABLE:
        st.error("❌ Reddit functionality unavailable. Install praw: `pip install praw`")
    elif not reddit:
        st.error("❌ Reddit API not configured. Check your .env file for REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT")
    else:
        reddit_url = st.text_input("Enter Reddit post URL:", 
                                  placeholder="https://www.reddit.com/r/example/comments/...")
        
        if st.button("Fetch Reddit Post", key="reddit_fetch"):
            if reddit_url.strip():
                try:
                    with st.spinner("Fetching Reddit post..."):
                        record = fetch_reddit_post(reddit_url)
                        if save_single_data(record):
                            st.success(f"✅ Reddit post saved! (ID: {record['id']})")
                            
                    
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Score", record['metadata']['likes'])
                            with col2:
                                st.metric("Comments", record['metadata']['num_comments'])
                            with col3:
                                st.metric("Subreddit", f"r/{record['metadata']['subreddit']}")
                            
              
                            with st.expander("Preview Post"):
                                st.write(f"**Author:** {record['author']}")
                                st.write(f"**Time:** {record['timestamp'][:19]}")
                                st.write("**Content:**")
                                st.write(record['text'][:800] + "..." if len(record['text']) > 800 else record['text'])
                                
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            else:
                st.warning("⚠️ Please enter a Reddit URL.")


with tab3:
    st.header("📰 News Article Collector")
    st.write("Search and fetch news articles by keyword.")
    
    if not NEWS_API_KEY:
        st.error("❌ News API not configured. Add NEWS_API_KEY to your .env file")
    else:
        news_query = st.text_input("Enter search keywords:", 
                                  placeholder="e.g., artificial intelligence, climate change, technology")
        
        if st.button("Fetch News Article", key="news_fetch"):
            if news_query.strip():
                try:
                    with st.spinner("Searching for news articles..."):
                        record = fetch_news(news_query)
                        if record:
                            if save_single_data(record):
                                st.success(f"✅ News article saved! (ID: {record['id']})")
                                
              
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Source", record['metadata']['source_name'])
                                with col2:
                                    st.metric("Author", record['author'])
                   
                                with st.expander("Preview Article"):
                                    st.write(f"**Published:** {record['timestamp'][:19]}")
                                    st.write(f"**URL:** {record['metadata']['url']}")
                                    st.write("**Content:**")
                                    st.write(record['text'])
                        else:
                            st.warning(f"⚠️ No news articles found for '{news_query}'")
                            
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            else:
                st.warning("⚠️ Please enter search keywords.")


