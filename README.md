# Character-Level Shakespeare Text Generation

This project is about:

- **Whitespace vs. Smart Tokenization** (NLTK) and Zipf's Law analysis  
- **One-Hot LSTM Model** for next-character prediction  
- **Temperature Sampling** & **Beam Search**  
- **Dense Embedding + LSTM Model**  
- **Re-ranking** via perplexity  

## Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/<your-username>/shakespeare-gen.git
cd shakespeare-gen
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Data
```python
from keras.utils import get_file
url = 'https://www.gutenberg.org/cache/epub/100/pg100.txt'
get_file('pg100.txt', origin=url, cache_dir='./data')
```

### 4. Run Scripts
- `src/onehot_model.py` for one-hot experiments  
- `src/embedding_model.py` for embedding experiments  

## License
MIT
