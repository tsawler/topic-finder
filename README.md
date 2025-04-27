# Topic Finder and Categorizer

A Flask-based web application that leverages Natural Language Processing (NLP) to identify common topics from a group of words and categorize words into topics using semantic similarity analysis with WordNet.

## Table of Contents
- [Overview](#overview)
- [NLP Concepts](#nlp-concepts)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Docker Development](#docker-development)
  - [Docker Production](#docker-production)
- [API Endpoints](#api-endpoints)
  - [Find Topic](#find-topic)
  - [Categorize Word](#categorize-word)
  - [Analyze Context](#analyze-context)
  - [Clear Cache](#clear-cache)
- [Web Interface](#web-interface)
- [Technical Implementation](#technical-implementation)
- [Attribution](#attribution)
- [License](#license)

## Overview

This application helps users analyze semantic relationships between words using Princeton University's WordNet database. The system can:

1. Find common topics among a set of words (e.g., "apple", "banana", "orange" → "fruit")
2. Categorize a single word into the most appropriate topic from a given list
3. Analyze the context of a group of words to determine their dominant part of speech

The application employs sophisticated NLP techniques to understand word relationships and hierarchies.

## NLP Concepts

### Natural Language Processing (NLP)
NLP is a field of artificial intelligence that focuses on the interaction between computers and human language. It involves enabling computers to process, understand, and generate natural language in useful ways.

### WordNet
WordNet is a large lexical database of English words. Nouns, verbs, adjectives, and adverbs are grouped into sets of cognitive synonyms (synsets), each expressing a distinct concept. Synsets are interlinked by means of conceptual-semantic and lexical relations.

Key WordNet concepts used in this application:

- **Synsets**: Groups of synonymous words that represent a specific concept.
- **Hypernyms**: Words with a broader meaning that encompass more specific words (e.g., "furniture" is a hypernym of "chair").
- **Hyponyms**: Words with a more specific meaning (e.g., "apple" is a hyponym of "fruit").
- **Lemmatization**: The process of reducing words to their base or dictionary form (e.g., "running" → "run").
- **Parts of Speech (POS)**: Categories of words with similar grammatical properties:
  - `n`: nouns (objects, concepts)
  - `v`: verbs (actions, occurrences)
  - `a`: adjectives (descriptors)
  - `r`: adverbs (how actions are performed)

### Semantic Similarity
Measures how close two words are in meaning. This application uses multiple similarity measures:

- **Wu-Palmer Similarity**: Considers the depths of two synsets in the WordNet taxonomy, along with the depth of their Least Common Subsumer (most specific ancestor node).
- **Path Similarity**: Based on the shortest path that connects the senses in the taxonomy.
- **Leacock-Chodorow Similarity**: Calculates similarity based on the shortest path between two concepts and the maximum depth of the taxonomy.
- **Resnik Similarity**: Uses information content (IC) to find similarity, based on how much information concepts share.

## Features

- Identify common topics from a list of words
- Categorize words into predefined topics
- Context-aware analysis of word groups
- Web interface with simple forms
- RESTful API for programmatic access
- Docker support for easy deployment
- Performance optimization with similarity caching

## Installation

### Prerequisites
- Docker and Docker Compose
- [Task](https://taskfile.dev) (optional, for running task commands)

Clone the repository:
```bash
git clone <repository-url>
cd topic-finder
```

## Usage

### Docker Development

Start the development server:
```bash
# Using Task
task up_dev
# OR using Docker Compose directly
docker compose up -d
```

Rebuild containers when code changes:
```bash
task up_build
```

Stop the containers:
```bash
task down
```

### Docker Production

Start the production server with Gunicorn:
```bash
# Using Task
task up_prod
# OR using Docker Compose directly
docker compose -f docker-compose-prod.yml up -d
```

Rebuild production containers:
```bash
task up_build_prod
```

## API Endpoints

### Find Topic
Finds common topics among a set of words.

**Endpoint:** `POST /find-topic`

**Request:**
```json
{
  "words": ["apple", "banana", "orange", "grape"],
  "context_pos": "n"  // Optional: preferred part of speech (n, v, a, or r)
}
```

**Response:**
```json
{
  "topic_words": ["fruit", "edible_fruit", "produce"]
}
```

### Categorize Word
Finds which topic from a list best fits a single word.

**Endpoint:** `POST /categorize-word`

**Request:**
```json
{
  "word": "laptop",
  "topics": ["furniture", "electronics", "food", "clothing"],
  "context_pos": "n"  // Optional: preferred part of speech
}
```

**Response:**
```json
{
  "best_topic": "electronics"
}
```

If no suitable topic is found:
```json
{
  "best_topic": null,
  "message": "Could not find a significantly similar topic."
}
```

### Analyze Context
Analyzes words to determine their dominant part of speech and suggests topics.

**Endpoint:** `POST /analyze-context`

**Request:**
```json
{
  "words": ["run", "sprint", "jog", "dash"]
}
```

**Response:**
```json
{
  "dominant_pos": "v",
  "pos_distribution": {"n": 1, "v": 3, "a": 0, "r": 0},
  "suggested_topics": ["run", "move", "locomote"],
  "word_pos_mapping": {
    "run": "v",
    "sprint": "v",
    "jog": "v",
    "dash": "n"
  }
}
```

### Clear Cache
Clears the similarity calculation cache.

**Endpoint:** `POST /clear-cache`

**Response:**
```json
{
  "message": "Similarity cache cleared",
  "previous_size": 250
}
```

## Web Interface

The application provides a simple web interface at the root URL (`/`) with two forms:

1. **Find Common Topics**: Enter comma-separated words to find their common topics
2. **Categorize a Single Word**: Enter a word and a list of possible topics to find the best match

## Technical Implementation

### Key Components

1. **NLTK Integration**: Uses the Natural Language Toolkit to access and work with WordNet
2. **WordNet Similarity Metrics**: Implements multiple similarity metrics with weights for better results
3. **POS-Aware Processing**: Handles different parts of speech appropriately
4. **Context-Sensitive Analysis**: Considers context when determining relationships
5. **Caching**: Implements similarity result caching for better performance
6. **Word Sense Disambiguation**: Uses multiple metrics to find the most appropriate meaning of a word
7. **Hypernym Tree Traversal**: Navigates word hierarchies to find common ancestors

### Advanced Features

- **Cross-POS Comparisons**: Can compare words across different parts of speech with appropriate penalties
- **Fallback Mechanisms**: Multiple strategies to ensure results even with difficult inputs
- **POS-Specific Relations**: Handles verb troponyms, adjective similarities, etc.
- **Weighted Scoring System**: Uses a sophisticated scoring system to rank potential matches

## Attribution

This project uses Princeton University's WordNet, a large lexical database of English:

**WordNet** © Princeton University 2010.
- George A. Miller (1995). WordNet: A Lexical Database for English. Communications of the ACM Vol. 38, No. 11: 39-41.
- Christiane Fellbaum (1998, ed.) *WordNet: An Electronic Lexical Database*. Cambridge, MA: MIT Press.

WordNet License: https://wordnet.princeton.edu/license-and-commercial-use

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.