# Topic Finder API

A Flask-based web application that uses natural language processing (NLP) techniques to analyze words and their relationships. The application offers two main features:

1. **Find Common Topics** - Identify the most relevant common topics or categories for a group of words.
2. **Categorize a Single Word** - Determine which topic from a provided list best fits a single input word.

## Table of Contents

- [Overview](#overview)
- [Technical Details](#technical-details)
  - [Key Concepts](#key-concepts)
  - [Architecture](#architecture)
- [API Reference](#api-reference)
  - [Find Common Topics](#find-common-topics)
  - [Categorize a Single Word](#categorize-a-single-word)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Development](#development)
- [How It Works](#how-it-works)
  - [Find Common Topics Process](#find-common-topics-process)
  - [Word Categorization Process](#word-categorization-process)
- [Attribution](#attribution)
- [License](#license)

## Overview

This application leverages the power of Princeton University's WordNet database to analyze semantic relationships between words. Through a user-friendly web interface or API endpoints, users can:

1. Find common topics or categories that best describe a group of related words
2. Determine which topic from a list best fits a single input word

These capabilities are particularly useful for:
- Content tagging and categorization
- Search optimization
- Text analysis
- Knowledge organization

## Technical Details

### Key Concepts

For those unfamiliar with natural language processing (NLP), here are the key concepts used in this application:

- **NLP (Natural Language Processing)**: A field of computer science focused on enabling computers to understand, interpret, and manipulate human language.

- **WordNet**: A large lexical database of English words. Nouns, verbs, adjectives, and adverbs are grouped into sets of cognitive synonyms (synsets), each expressing a distinct concept.

- **Synset**: A set of synonyms that share a common meaning.

- **Hypernym**: A word with a broader meaning that includes more specific words. For example, "fruit" is a hypernym of "apple."

- **Wu-Palmer Similarity**: A method to calculate how similar two words are, based on the depth of their synsets in the WordNet hierarchy and their Least Common Subsumer (LCS).

- **Word Sense Disambiguation (WSD)**: The process of identifying which sense of a word is being used in a particular context.

### Architecture

The application consists of:

1. **Flask Web Application**: Serves both the HTML frontend and API endpoints.
2. **NLTK (Natural Language Toolkit)**: Provides the NLP functionality, particularly access to WordNet.
3. **Docker Container**: Ensures consistent deployment across environments.

## API Reference

### Find Common Topics

Finds the most meaningful common topics for a group of words.

**Endpoint:** `POST /find-topic`

**Request Format:**
```json
{
  "words": ["word1", "word2", "word3", ...]
}
```

**Response Format:**
```json
{
  "topic_words": ["topic1", "topic2", "topic3", ...]
}
```

**Example:**
```json
// Request
{
  "words": ["apple", "banana", "orange"]
}

// Response
{
  "topic_words": ["fruit", "produce", "food"]
}
```

### Categorize a Single Word

Identifies which topic from a provided list best fits a single word.

**Endpoint:** `POST /categorize-word`

**Request Format:**
```json
{
  "word": "word_to_categorize",
  "topics": ["topic1", "topic2", "topic3", ...]
}
```

**Response Format:**
```json
{
  "best_topic": "matching_topic"
}
```

Or, if no match is found:
```json
{
  "best_topic": null,
  "message": "Could not find a significantly similar topic."
}
```

**Example:**
```json
// Request
{
  "word": "grape",
  "topics": ["fruit", "vehicle", "sport"]
}

// Response
{
  "best_topic": "fruit"
}
```

## Getting Started

### Prerequisites

- Docker (recommended) or Python 3.8+
- [Task](https://taskfile.dev/) (optional but recommended; alternative to Make)

### Installation

#### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/topic-finder.git
   cd topic-finder
   ```

2. Build and start the Docker container:
   ```bash
   # Using Taskfile (recommended)
   task up_build
   
   # Or using Docker Compose directly
   docker compose up -d --build
   ```

#### Manual Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/topic-finder.git
   cd topic-finder
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download WordNet data:
   ```bash
   python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
   ```

4. Run the Flask application:
   ```bash
   flask run
   ```

### Running the Application

After starting the application:

1. Open your browser and navigate to `http://localhost:4001`
2. Use the web interface to interact with the application, or
3. Make API requests to `http://localhost:4001/find-topic` or `http://localhost:4001/categorize-word`

## Development

The application uses two Docker Compose files:

1. `docker-compose.yml` - For development with hot-reloading
2. `docker-compose-prod.yml` - For production with Gunicorn WSGI server

You can use the included Taskfile for common operations:

```bash
# Start development environment
task up_dev

# Rebuild and start development environment
task up_build

# Stop the application
task down

# Start production environment
task up_prod

# Rebuild and start production environment
task up_build_prod
```

## How It Works

### Find Common Topics Process

When you submit a list of words to find common topics, the application:

1. **Synset Identification**: Finds all possible meanings (synsets) for each word in WordNet.
2. **Word Sense Disambiguation (WSD)**: Determines the most likely sense of each word based on Wu-Palmer similarity to other words in the list.
3. **Common Hypernym Detection**: Identifies common broader terms (hypernyms) that encompass the selected word senses.
4. **Ranking and Selection**: Ranks these hypernyms based on specificity and relevance, then returns the top 3.

#### Example

Input: ["apple", "banana", "orange"]

1. The system identifies that all three words have senses related to fruit.
2. It finds common hypernyms like "fruit", "produce", and "food".
3. It ranks these hypernyms based on how specific they are and how strongly they relate to the input words.
4. The final result would be these ranked hypernyms: ["fruit", "produce", "food"].

### Word Categorization Process

When categorizing a single word against a list of topics:

1. **Synset Identification**: Finds all possible meanings for the input word and topic words.
2. **Similarity Calculation**: Calculates Wu-Palmer similarity between the input word and each topic word.
3. **Best Match Selection**: Selects the topic with the highest similarity score that exceeds a minimum threshold.

#### Example

Input word: "grape"
Topic list: ["fruit", "vehicle", "sport"]

1. The system identifies synsets for "grape" and for each topic word.
2. It calculates similarities: grape-fruit (high), grape-vehicle (low), grape-sport (low).
3. Since "fruit" has the highest similarity score above the threshold, it's selected as the best topic.

## Attribution

This application uses **WordNet®**, a large lexical database of English developed at Princeton University under the direction of Professor George A. Miller. WordNet® is a registered trademark of Princeton University.

**Citation:**
- Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010.

## License

This project is licensed under the MIT License - see below for details:
