import trafilatura

import torch
from transformers import BertForSequenceClassification, BertTokenizer

import argparse
from rich.console import Console
from rich.table import Table

def extract_article_title(url):
    downloaded = trafilatura.fetch_url(url) 
    return trafilatura.bare_extraction(downloaded, include_links=True)['title']

def extract_article_text(url):
    downloaded = trafilatura.fetch_url(url) 
    return trafilatura.extract(downloaded)

def clean_article_text(text):
    return text.replace("\n", "")

def prepare_text(text, tokenizer):
    return tokenizer.encode_plus(text, 
                                add_special_tokens=False, 
                                return_tensors='pt')

def prepare_chunked_inputs(chunksize, tokens):
    input_id_chunks = list(tokens['input_ids'][0].split(chunksize - 2))
    mask_chunks = list(tokens['attention_mask'][0].split(chunksize - 2))
    
    #Loop through each chunk 
    for i in range(len(input_id_chunks)):
        #add CLS and SEP tokens to input IDs
        input_id_chunks[i] = torch.cat([
            torch.Tensor([101]), input_id_chunks[i], torch.Tensor([102])
        ])
        #add attention tokens to attention mask
        mask_chunks[i] = torch.cat([
            torch.Tensor([1]), mask_chunks[i], torch.Tensor([2])
        ])
        #calc padding length
        pad_len = chunksize - input_id_chunks[i].shape[0]
        #check if the length of tensor satisfies padding size requirement
        if pad_len > 0:
            #if padding length is more than 0, we need to add padding
            input_id_chunks[i] = torch.cat([
                input_id_chunks[i], torch.Tensor([0] * pad_len)
            ])
            mask_chunks[i] = torch.cat([
                mask_chunks[i], torch.Tensor([0] * pad_len)
            ])
    
    input_ids = torch.stack(input_id_chunks)
    attention_mask = torch.stack(mask_chunks)

    input_dict = {
        'input_ids':input_ids.long(),
        'attention_mask':attention_mask.int()
    }

    return input_dict


def pipeline(url, model_name, chunksize):
    ## Defining the model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    ## Defining the complete preprocessing pipeline
    article_title = extract_article_title(url)
    article_text = extract_article_text(url)
    cleaned_text = clean_article_text(article_text)
    tokens = prepare_text(cleaned_text, tokenizer)
    input_dict = prepare_chunked_inputs(chunksize, tokens)

    outputs = model(**input_dict)
    probs = torch.nn.functional.softmax(outputs[0], dim=-1)
    probs = probs.mean(dim=0)
    return article_title, torch.argmax(probs).item()

def get_output_label(index):
    labels = {
        0:'positive',
        1:'neutral',
        2:'negative'
    }
    return labels[index]

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    ## User can pass the article text, model name and chunksize as arguments
    parser.add_argument('--fpath', type=str, help='Path of the file containing URLS to parse', required=False, default='')
    parser.add_argument('--model_name', type=str, help='Name of the pre-trained model', required=False, default='')
    parser.add_argument('--chunksize', type=int, help='Chunksize depends on the max length of tokens the model can accept',
                        required=True)
    args = parser.parse_args()
    
    table = Table(title="Financial Sentiment Analysis")

    table.add_column("Article URL", justify="right", style="cyan", no_wrap=True)
    table.add_column("Sentiment", style="magenta")

    with open(args.fpath, 'r') as f:
        urls = f.read().splitlines()
        for url in urls:
            title, label = pipeline(url, args.model_name, args.chunksize)
            label_name = get_output_label(label)
            table.add_row(title, label_name)
    
    console = Console()
    console.print(table)


