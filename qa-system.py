import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import pandas as pd
import tiktoken
import openai
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
from ast import literal_eval

# Define OpenAI api_key -> throw error if none is found
api_key = os.environ.get("OPENAI_API_KEY")

if api_key is None:
    raise EnvironmentError("OpenAI API key not found in your system's environment variables.")

# Load the cl100k_base tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Read the CSV file
df = pd.read_csv('processed/scraped.csv', index_col=0)
df.columns = ['title', 'text']

# Tokenize the text and save the number of tokens to a new column
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

# FOR TESTING PURPOSES | Visualize the distribution of the number of tokens per row using a histogram
# df.n_tokens.hist()

# Define the maximum number of tokens | defined by the models we're using (text-embedding-ada-002 & gpt-3.5-turbo)
# max_tokens = 500
max_tokens = 3000

# Function to split the text into chunks of a maximum number of tokens
def split_into_chunks(text, max_tokens=max_tokens):
    # Split the text into sentences
    sentences = text.split('. ')

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and calculate the number of tokens for each sentence
    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(" " + sentence))  # Calculate tokens for the current sentence

        # If adding the current sentence would exceed the maximum tokens, start a new chunk
        if tokens_so_far + sentence_tokens > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # Add the sentence to the current chunk
        chunk.append(sentence)
        tokens_so_far += sentence_tokens + 1  # Add 1 for the period after the sentence

    # Add the last chunk to the list of chunks
    if chunk:
        chunks.append(". ".join(chunk) + ".")

    return chunks

# Create a list to store shortened text chunks
shortened = []

# Loop through the dataframe
for row in df.iterrows():
    # If the text is None, go to the next row
    if row[1]['text'] is None:
        continue

    # If the number of tokens is greater than the max number of tokens, split the text into chunks
    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_chunks(row[1]['text'])
    # Otherwise, add the text to the list of shortened texts
    else:
        shortened.append(row[1]['text'])

# Create a dataframe from the list of shortened text chunks
df = pd.DataFrame(shortened, columns=['text'])

# Calculate the number of tokens for each shortened text
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

# FOR TESTING PURPOSES | plot the distribution
# df.n_tokens.hist()

# Function to create a context for a question based on the most similar context from the dataframe
def create_context(question, df, max_len=3000, size="ada"):
    # Get the embeddings for the question
    #q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    #df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x[:3000], engine='text-embedding-ada-002')['data'][0]['embedding'])

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

# Function to answer a question based on the most similar context from the dataframe texts
def answer_question(
    df,
    # model="text-davinci-003",
    model="gpt-3.5-turbo-instruct",
    # question="Am I allowed to publish model outputs without human review?",
    question="What is the Goldman School of Public Policy/GSPP? What do they do? What research do they engage in? How do they help the world be a better place?",
    # max_len=1800,
    # max_len=8192,
    max_len=3000,
    size="ada",
    debug=False,
    # max_tokens=150,
    max_tokens=500,
    stop_sequence=None
):
    # Create a context for the question
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )

    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create completions using the question and context
        response = openai.Completion.create(
            # prompt=f"Answer the question based on the context.",
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"Hm, I'm not sure\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

# Example usage of the `answer_question` function
# print(answer_question(df, question="What day is it?", debug=False))
print("\n\n")

print(answer_question(df, question="What does GSPP do?"))
print("\n")

print(answer_question(df, question="Can you tell me about GSPP's research impact?"))
print("\n")

print(answer_question(df, question="How can I learn more about GSPP?"))
print("\n")

print(answer_question(df, question="Who is the GSPP's Dean?"))
print("\n")
