## Sentiment Analysis of Financial Articles with FinBERT
This is a simple pipeline to read news articles from the provided web URL and detect their sentiment using the FinBERT model.

### Implementation 
Since this is a long document classification problem, the document is chunked into tokens of 512 and sent to the model for inference. This project does not perform any fine-tuning, rather it demonstrates an inference pipeline for the model by creating an easy-to-use CLI.

### Dependencies 
1. transformers
2. trafilatura
3. torch
4. rich
5. argparse

### How to run 
```
python finsent.py --fpath 'data.txt' --model_name 'ProsusAI/finbert' --chunksize 512
```