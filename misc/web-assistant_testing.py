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

# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]{0,1}://.+$'

# Define root domain to crawl
domain = "gspp.berkeley.edu"
full_url = "https://gspp.berkeley.edu/"

# Parse the URL and get the domain
local_domain = urlparse(full_url).netloc

# Create a directory to store the text files (sorted by domain)
if not os.path.exists("text/"):
    os.mkdir("text/")

if not os.path.exists(f"text/{local_domain}/"):
    os.mkdir(f"text/{local_domain}/")

# Create a directory to store the CSV files
if not os.path.exists("processed"):
    os.mkdir("processed")

# Create a class to parse the HTML and get the hyperlinks
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # Create a list to store the hyperlinks
        self.hyperlinks = []

    # Override the HTMLParser's handle_starttag method to get the hyperlinks
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        # If the tag is an anchor tag and it has an href attribute, add the href attribute to the list of hyperlinks
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])

# Function to get the hyperlinks from a URL
def get_hyperlinks(url):
    try:
        # Open the URL and read the HTML
        with urllib.request.urlopen(url) as response:

            # If the response is not HTML, return an empty list
            if not response.info().get('Content-Type').startswith("text/html"):
                return []

            # Decode the HTML
            html = response.read().decode('utf-8')
    except Exception as e:
        print(e)
        return []

    # Create the HTML Parser and then Parse the HTML to get hyperlinks
    parser = HyperlinkParser()
    parser.feed(html)

    return parser.hyperlinks

# Function to get the hyperlinks from a URL that are within the same domain
def get_domain_hyperlinks(local_domain, url):
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None

        # If the link is a URL, check if it is within the same domain
        if re.search(HTTP_URL_PATTERN, link):
            # Parse the URL and check if the domain is the same
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link

        # If the link is not a URL, check if it is a relative link
        else:
            if link.startswith("/"):
                link = link[1:]
            elif (
                link.startswith("#")
                or link.startswith("mailto:")
                or link.startswith("tel:")
            ):
                continue
            clean_link = "https://" + local_domain + "/" + link

        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
            clean_links.append(clean_link)

    # Return the list of hyperlinks that are within the same domain
    return list(set(clean_links))

# Function to crawl and scrape specified website
def crawl(url):
    # Parse the URL and get the domain
    local_domain = urlparse(url).netloc

    # Create a queue to store the URLs to crawl
    queue = deque([url])

    # Create a set to store the URLs that have already been seen (no duplicates)
    seen = set([url])

    # Create a directory to store the text files -> I think this is redundant? putting it here just in case
    if not os.path.exists("text/"):
        os.mkdir("text/")

    if not os.path.exists("text/"+local_domain+"/"):
        os.mkdir("text/" + local_domain + "/")

    # Create a directory to store the CSV files
    if not os.path.exists("processed"):
        os.mkdir("processed")

    # Counter to generate unique identifiers for file names
    file_counter = 1

    # While the queue is not empty, continue crawling
    while queue:

        # Get the next URL from the queue
        url = queue.pop()
        print(url) # for debugging and to see the progress | +1 for realtime updates :)

        # Try extracting the text from the link, if failed proceed with the next item in the queue
        try:
            # Generate a unique identifier for the file name (maximum of 10 characters)
            file_identifier = f"page_{file_counter:04}"[:10]

            # Increment the file counter
            file_counter += 1

            # Save text from the URL to a <file_identifier>.txt file
            with open(f'text/{local_domain}/{file_identifier}.txt', "w", encoding="UTF-8") as f:
                # Get the text from the URL using BeautifulSoup
                soup = BeautifulSoup(requests.get(url).text, "html.parser")

                # Get the text but remove the tags
                text = soup.get_text()

                # If the crawler gets to a page that requires JavaScript, it will stop the crawl
                if "You need to enable JavaScript to run this app." in text:
                    print(f"Unable to parse page {url} due to JavaScript being required")

                # Otherwise, write the text to the file in the text directory
                f.write(text)
        except Exception as e:
            print(f"Unable to parse page {url}")

        # Get the hyperlinks from the URL and add them to the queue
        for link in get_domain_hyperlinks(local_domain, url):
            if link not in seen:
                queue.append(link)
                seen.add(link)

crawl(full_url)

# Function to remove newlines from a Series
def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

# Create a list to store the text files
texts = []

# Get all the text files in the text directory
for file in os.listdir(f"text/{domain}/"):
    # Open the file and read the text
    with open(f"text/{domain}/{file}", "r", encoding="UTF-8") as f:
        text = f.read()
        # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces
        texts.append((file[11:-4].replace('-', ' ').replace('_', ' ').replace('#update', ''), text))

# Create a dataframe from the list of texts
df = pd.DataFrame(texts, columns=['fname', 'text'])

# Set the text column to be the raw text with the newlines removed
df['text'] = df.fname + ". " + remove_newlines(df.text)

# Save the DataFrame to a CSV file
df.to_csv('processed/scraped.csv', escapechar='\\')

# Load the cl100k_base tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Read the CSV file
df = pd.read_csv('processed/scraped.csv', index_col=0)
df.columns = ['title', 'text']

# Tokenize the text and save the number of tokens to a new column
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

# Visualize the distribution of the number of tokens per row using a histogram
df.n_tokens.hist()

# Define the maximum number of tokens
# max_tokens = 500
max_tokens = 50

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

# Calculate the number of tokens for each shortened text and plot the distribution
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
df.n_tokens.hist()

# Function to create a context for a question based on the most similar context from the dataframe
def create_context(question, df, max_len=450, size="ada"):
    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])

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
    model="text-davinci-003",
    # question="Am I allowed to publish model outputs without human review?",
    question="What is OpenAI?",
    # max_len=1800,
    # max_len=8192,
    max_len=450,
    size="ada",
    debug=False,
    # max_tokens=150,
    max_tokens=50,
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
            # prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            prompt=f"Answer the question based on the context.",
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
# print(answer_question(df, question="What is our newest embeddings model?"))
print(answer_question(df, question="How can I enroll at GSPP?"))
