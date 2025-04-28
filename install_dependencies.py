import sys
import subprocess
import os

def install_dependencies():
    print("Installing dependencies...")
    
    subprocess.check_call([sys.executable, "-m", "pip", "install", "keras==2.12.0", "numpy==1.24.3", 
                          "tensorflow==2.12.0", "nltk==3.8.1", "flask==2.3.3", 
                          "flask-cors==4.0.0", "gunicorn==21.2.0"])
    
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    
    subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy==3.5.0"])
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    
    subprocess.check_call([sys.executable, "-m", "pip", "install", "chatterbot==1.0.4", "chatterbot-corpus==1.2.0"])
    
    print("All dependencies installed successfully!")

if __name__ == "__main__":
    install_dependencies()
