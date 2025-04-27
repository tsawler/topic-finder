# Topic Finder

A web application that helps identify common topics or themes from a set of related words using natural language processing techniques.

## Overview

Topic Finder uses semantic analysis to find a representative word that best captures the common theme among a set of words. For example, given the words "apple", "banana", and "orange", the application would suggest "fruit" as the common topic.

This tool is particularly useful for:
- Content organization and tagging
- SEO keyword analysis
- Document categorization
- Educational applications in linguistics

## How It Works

### Technical Flow

1. **User Input**: Users enter a comma-separated list of words through a web interface
2. **Word Processing**: The application analyzes the semantic relationships between these words using WordNet (a lexical database)
3. **Topic Analysis**: The system identifies a word that represents the common theme or category
4. **Result Display**: The suggested topic word is presented to the user

### Core NLP Concepts

#### WordNet
WordNet is a large lexical database of English words. It groups nouns, verbs, adjectives, and adverbs into sets of cognitive synonyms (synsets), each expressing a distinct concept. WordNet includes various semantic relationships between these concepts, such as:

- **Synonyms**: Words with similar meanings
- **Hypernyms**: More general terms (e.g., "fruit" is a hypernym of "apple")
- **Hyponyms**: More specific terms (e.g., "apple" is a hyponym of "fruit")

#### Word Sense Disambiguation (WSD)
Words often have multiple meanings (senses). For example, "bank" can refer to a financial institution or the side of a river. WSD is the process of identifying which sense of a word is being used in a particular context.

In this application, WSD is performed using Wu-Palmer similarity, which:
1. Compares each possible meaning of each word with possible meanings of other words
2. Calculates similarity scores based on their positions in the WordNet hierarchy
3. Selects the sense (meaning) of each word that has the highest average similarity to the meanings of other words

#### Hypernym Finding
After identifying the most likely sense of each word, the application:
1. Traces up the WordNet hierarchy to find common "ancestor" terms (hypernyms)
2. Ranks these common hypernyms based on:
   - How many of the input words share this hypernym
   - How specific the hypernym is (deeper terms in the hierarchy are more specific)
   - How directly related the hypernym is to the original words

The system then selects the most appropriate common term based on this ranking to suggest as the topic word.

## Installation

### Prerequisites
- Docker and Docker Compose
- Git

### Setup

1. Clone the repository:
   ```
   git clone [repository-url]
   cd topic-finder
   ```

2. Start the application in development mode:
   ```
   task up
   ```
   or
   ```
   docker compose up -d
   ```

3. For production deployment:
   ```
   task up_prod
   ```
   or
   ```
   docker compose -f docker-compose-prod.yml up -d
   ```

4. Access the application in your browser:
   ```
   http://localhost:4001
   ```

## Usage

1. Enter a list of related words in the text area, separated by commas (e.g., "apple, banana, orange")
2. Click the "Find Topic" button
3. View the suggested topic word in the results section

## Project Structure

- `app.py`: Main Flask application with the topic-finding algorithm
- `index.html`: Web interface for the application
- `requirements.txt`: Python dependencies
- `docker-compose.yml`: Configuration for development deployment
- `docker-compose-prod.yml`: Configuration for production deployment
- `Taskfile.yml`: Task runner configuration for common commands

## Technical Details

### NLTK and WordNet

The application uses the Natural Language Toolkit (NLTK), specifically its WordNet integration, to analyze word relationships. The first time the application runs, it automatically downloads the required WordNet data.

### The Algorithm

The core algorithm in `find_meaningful_topic_word_improved()` performs these steps:

1. **Word Validation**: Checks if the provided words exist in WordNet as nouns
2. **Word Sense Disambiguation**:
   - For each word, retrieves all possible noun meanings from WordNet
   - Calculates similarity scores between all word senses using Wu-Palmer similarity
   - Selects the most likely meaning of each word based on highest similarity to other words
3. **Common Hypernym Identification**:
   - Maps the selected word senses to their positions in the WordNet hierarchy
   - Identifies common ancestor terms (hypernyms) shared by multiple words
   - Filters out overly general terms (like "entity" or "object")
4. **Ranking and Selection**:
   - Ranks candidate hypernyms based on how many words they represent, their specificity, and their proximity to the original words
   - Returns the most appropriate common term as the suggested topic

## Docker Deployment

The application is containerized using Docker for consistent deployment:

- **Development Mode**: Uses Flask's built-in development server for quick testing and debugging
- **Production Mode**: Uses Gunicorn as the WSGI server for improved performance and reliability

## API Reference

### Endpoint: `/find-topic`
- **Method**: POST
- **Content-Type**: application/json
- **Request Body**:
  ```json
  {
    "words": ["word1", "word2", "word3"]
  }
  ```
- **Response**:
  ```json
  {
    "topic_word": "suggested_topic"
  }
  ```

## Limitations

- Works best with concrete nouns that have clear hierarchical relationships
- Limited to words available in the WordNet database
- Performance depends on the specificity and relatedness of the input words
- Currently supports English language only

## Future Enhancements

- Support for additional languages
- Integration with other NLP techniques like word embeddings
- Ability to process phrases and not just individual words
- User feedback mechanism to improve suggestions over time