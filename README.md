# Transformers for Natural Language Processing
Performing various NLP tasks with the latest transformer libraries and engineering tools. In this repo, the following tasks will be demonstrated (each task will have its own CLI) :
1. Financial Sentiment Analysis from HTML news articles 
```
python finsent.py --fpath 'data.txt' --model_name 'ProsusAI/finbert' --chunksize 512
```

3. Named Entity Recognition
4. Open Domain Question Answering with Transformers (Elasticsearch & Haystack)
5. Masked Language Modelling 
6. Interpretability techniques on NLP Sentiment Analysis


## Dependencies 
1. Huggingface Transformers
2. PyTorch
3. Trafilatura
4. Haystack
5. Elasticsearch 
6. FAISS
