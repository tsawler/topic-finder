import nltk
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from flask import Flask, request, jsonify, render_template
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- NLTK Data Configuration ---
# Check if NLTK_DATA environment variable is set and use it
nltk_data_path = os.environ.get('NLTK_DATA')
if nltk_data_path:
    logger.info(f"Using NLTK data path from environment: {nltk_data_path}")
    nltk.data.path.insert(0, nltk_data_path)

# --- NLTK Data Download ---
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')  # For lemmatization
    nltk.data.find('corpora/omw-1.4')   # Open Multilingual WordNet
    logger.info("All required NLTK resources found.")
except LookupError as e:
    logger.info(f"NLTK data not found: {e}. Downloading required resources...")
    try:
        # Use NLTK_DATA path if set, otherwise default download location
        download_dir = nltk_data_path if nltk_data_path else None
        nltk.download('wordnet', download_dir=download_dir)
        nltk.download('omw-1.4', download_dir=download_dir)
        nltk.download('punkt', download_dir=download_dir)  # For lemmatization
        logger.info("Download complete.")
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {e}")
        logger.error("Please ensure you have an internet connection and try again.")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# --- Initialize Information Content (IC) for Resnik similarity ---
ic = None
try:
    # Try to find the IC corpus
    nltk.data.find('corpora/wordnet_ic')
    ic = wordnet_ic.ic('ic-brown.dat')
    logger.info("Information Content corpus loaded successfully.")
except LookupError:
    # Download if not available
    logger.info("Information Content corpus not found. Downloading...")
    try:
        nltk.download('wordnet_ic', download_dir=nltk_data_path)
        ic = wordnet_ic.ic('ic-brown.dat')
        logger.info("Information Content corpus loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load Information Content corpus: {e}")
        # Fallback to None, which will disable Resnik similarity

# Define constants
# Higher threshold for more confident matching
WUP_SIMILARITY_THRESHOLD = 0.35
# Allow more flexibility in hypernym depth
MIN_MEANINGFUL_HYPERNYM_DEPTH = 2

# New constants based on analysis
MIN_VALID_WORDS_FOR_TOPIC = 2 # Minimum number of words with noun synsets to attempt finding a common topic
MIN_LENIENT_HYPERNYM_DEPTH = 1 # Lower depth threshold for hypernyms in fallback ranking

# Ranking score constants (used in compute_rank_score)
SHARED_COUNT_WEIGHT = 100
DEPTH_WEIGHT = 10
PATH_SCORE_BASE = 50
PATH_SCORE_MULTIPLIER = 5

# Similarity calculation constants
CROSS_POS_SIMILARITY_PENALTY = 0.8 # Penalty applied to cross-POS similarity scores

# Enhanced similarity methods
SIMILARITY_METHODS = {
    'wup': lambda s1, s2: s1.wup_similarity(s2),
    'path': lambda s1, s2: s1.path_similarity(s2),
    'lch': lambda s1, s2: s1.lch_similarity(s2) if s1.pos() == s2.pos() else None,
    'res': lambda s1, s2: s1.res_similarity(s2, ic) if ic and s1.pos() == s2.pos() else None
}

# Weights for different similarity methods
SIMILARITY_WEIGHTS = {
    'wup': 1.0,  # Wu-Palmer gives good results for hierarchical relationships
    'path': 0.8,  # Path similarity is simpler but still useful
    'lch': 0.9,  # Leacock-Chodorow works well for specific domain comparisons
    'res': 1.1   # Resnik uses information content, good for disambiguating word senses
}

# POS weighting - give more importance to certain parts of speech based on context
POS_WEIGHTS = {
    'n': 1.0,  # Nouns (default weight)
    'v': 0.9,  # Verbs
    'a': 0.8,  # Adjectives
    'r': 0.7   # Adverbs
}

# Caching mechanism for similarity calculations
similarity_cache = {}

def get_cached_similarity(method, synset1, synset2):
    """
    Get or calculate and cache similarity between two synsets using specified method.
    
    Args:
        method (str): The similarity method to use ('wup', 'path', 'lch', 'res')
        synset1: First WordNet synset
        synset2: Second WordNet synset
        
    Returns:
        float: Similarity score or None if calculation fails
    """
    # Create a cache key from the method and synset names
    cache_key = (method, synset1.name(), synset2.name())
    
    # Check if result is already in cache
    if cache_key in similarity_cache:
        return similarity_cache[cache_key]
    
    # Calculate similarity
    sim_func = SIMILARITY_METHODS.get(method)
    if not sim_func:
        return None
    
    try:
        similarity = sim_func(synset1, synset2)
        # Cache the result
        similarity_cache[cache_key] = similarity
        return similarity
    except Exception as e:
        logger.debug(f"Error calculating {method} similarity: {e}")
        similarity_cache[cache_key] = None
        return None

def calculate_weighted_similarity(synset1, synset2):
    """
    Calculate a weighted similarity score using multiple similarity measures.
    
    Args:
        synset1: First WordNet synset
        synset2: Second WordNet synset
        
    Returns:
        float: Weighted similarity score or None if no valid similarities
    """
    scores = []
    weights = []
    
    # Try each similarity method
    for method_name in SIMILARITY_METHODS:
        sim = get_cached_similarity(method_name, synset1, synset2)
        if sim is not None:
            scores.append(sim)
            weights.append(SIMILARITY_WEIGHTS[method_name])
    
    # Return weighted average if we have scores
    if scores:
        return sum(s * w for s, w in zip(scores, weights)) / sum(weights)
    return None

# --- Helper Functions ---
def get_synsets_with_fallbacks(word, pos=None):
    """
    Get WordNet synsets for a word with fallbacks to alternate forms.

    Args:
        word (str): The word to look up
        pos (str, optional): Part of speech. Defaults to None (all).

    Returns:
        dict: Dictionary of synsets by POS
    """
    result = {'n': [], 'v': [], 'a': [], 'r': []}

    # Normalize the word
    word = word.lower().strip()

    # Try to get synsets for the original word
    if pos is None or pos == 'n':
        result['n'] = wordnet.synsets(word, pos=wordnet.NOUN)

        # Try singular form if word might be plural
        if word.endswith('s') and not result['n']:
            singular = lemmatizer.lemmatize(word, pos='n')
            if singular != word:
                result['n'].extend(wordnet.synsets(singular, pos=wordnet.NOUN))

    if pos is None or pos == 'v':
        result['v'] = wordnet.synsets(word, pos=wordnet.VERB)
        
        # Try lemmatized verb form
        lemma_verb = lemmatizer.lemmatize(word, pos='v')
        if lemma_verb != word:
            additional_verb_synsets = wordnet.synsets(lemma_verb, pos=wordnet.VERB)
            result['v'].extend([s for s in additional_verb_synsets if s not in result['v']])

    if pos is None or pos == 'a':
        result['a'] = wordnet.synsets(word, pos=wordnet.ADJ)
        result['a'].extend(wordnet.synsets(word, pos='s'))  # Include satellite adjectives

    if pos is None or pos == 'r':
        result['r'] = wordnet.synsets(word, pos=wordnet.ADV)

    return result

def get_pos_specific_relations(synset):
    """
    Get POS-specific semantic relations for a synset.
    
    Args:
        synset: A WordNet synset
        
    Returns:
        list: Related synsets through POS-specific relations
    """
    related = []
    
    try:
        pos = synset.pos()
        
        if pos == 'v':
            # For verbs: collect troponyms (more specific ways to do the verb)
            for troponym in synset.troponyms():
                related.append(troponym)
                
            # Also collect entailments (actions that this verb entails)
            for entailment in synset.entailments():
                related.append(entailment)
                
        elif pos == 'a':
            # For adjectives: collect similar adjectives
            for similar in synset.similar_tos():
                related.append(similar)
                
            # Also collect pertainyms (words this adjective is derived from)
            for lemma in synset.lemmas():
                for pertainym in lemma.pertainyms():
                    if pertainym.synset() not in related:
                        related.append(pertainym.synset())
    
    except Exception as e:
        logger.debug(f"Error getting POS-specific relations: {e}")
    
    return related

# --- Topic Finding Function ---
def find_top_topic_words(topic_words, top_n=3, context_pos=None):
    """
    Finds the top n meaningful descriptive words for a list of topic words.
    Enhanced to handle all parts of speech with context-sensitive weighting.

    Args:
        topic_words (list): A list of strings, where each string is a word from the topic.
        top_n (int): Number of top topic words to return (default: 3)
        context_pos (str, optional): Context-based part of speech preference ('n', 'v', 'a', or 'r')

    Returns:
        list: A list of the top n most representative common hypernyms found,
              or the original word if only one is provided,
              or empty list if no meaningful common hypernyms are identified.
        dict: Optional debug info explaining why no results were found if the list is empty
    """
    # Handle edge cases
    if not topic_words:
        return [], {"reason": "No input words provided"}

    if len(topic_words) == 1:
        return [topic_words[0]], None

    # Apply POS weighting based on context
    pos_weights = dict(POS_WEIGHTS)
    if context_pos and context_pos in pos_weights:
        pos_weights[context_pos] *= 1.2

    # Preprocess words: lowercase and lemmatize for all relevant POS
    processed_words = []
    for word in topic_words:
        word_lower = word.lower().strip()
        processed_words.append(word_lower)

    # Get synsets for each word by POS
    word_synsets_by_pos = {}
    valid_words = set()

    for word in processed_words:
        word_synsets_by_pos[word] = get_synsets_with_fallbacks(word)
        
        # Check if word has any synsets
        has_synsets = False
        for pos_synsets in word_synsets_by_pos[word].values():
            if pos_synsets:
                has_synsets = True
                valid_words.add(word)
                break
        
        if not has_synsets:
            logger.debug(f"'{word}' not found in WordNet. Skipping.")

    # Check if we have enough valid words to proceed
    if len(valid_words) < MIN_VALID_WORDS_FOR_TOPIC:
        return [], {"reason": f"Not enough words with synsets found (found {len(valid_words)})"}

    # --- Improved Word Sense Disambiguation with multiple similarity methods ---
    # For each word, find the synset with the highest weighted similarity to other words
    synset_scores = defaultdict(float)
    synset_to_word_map = {}
    synset_pos_map = {}

    for word in valid_words:
        for pos, synsets in word_synsets_by_pos[word].items():
            for syn in synsets:
                synset_to_word_map[syn] = word
                synset_pos_map[syn] = pos

                # Calculate average weighted similarity to most similar sense of other words
                similarities = []
                pos_multiplier = pos_weights.get(pos, 1.0)
                
                for other_word in valid_words:
                    if word == other_word:
                        continue

                    max_sim_to_other_word = 0.0
                    for other_pos, other_synsets in word_synsets_by_pos[other_word].items():
                        # Apply cross-POS penalty if different POS
                        other_pos_multiplier = pos_weights.get(other_pos, 1.0)
                        cross_pos_penalty = 1.0 if pos == other_pos else CROSS_POS_SIMILARITY_PENALTY
                        
                        for other_syn in other_synsets:
                            weighted_sim = calculate_weighted_similarity(syn, other_syn)
                            if weighted_sim is not None:
                                # Apply POS weights and cross-POS penalty
                                adjusted_sim = weighted_sim * pos_multiplier * other_pos_multiplier * cross_pos_penalty
                                max_sim_to_other_word = max(max_sim_to_other_word, adjusted_sim)

                    if max_sim_to_other_word > 0:
                        similarities.append(max_sim_to_other_word)

                # Use average similarity
                if similarities:
                    synset_scores[syn] = sum(similarities) / len(similarities)

    # Select the synset with the highest score for each word
    word_best_synset = {}
    word_best_synset_score = {}

    for syn, score in synset_scores.items():
        word = synset_to_word_map[syn]
        if word not in word_best_synset_score or score > word_best_synset_score[word]:
            word_best_synset_score[word] = score
            word_best_synset[word] = syn

    # Collect the selected synsets
    selected_synsets_list = list(word_best_synset.values())

    if len(selected_synsets_list) < MIN_VALID_WORDS_FOR_TOPIC:
        return [], {"reason": f"Not enough synsets ({len(selected_synsets_list)}) found with good disambiguation scores"}

    # --- Find Common Hypernyms/Ancestors with Improved Algorithm ---
    # Track hypernym info and the synsets they're connected to
    common_ancestors_info = defaultdict(lambda: {
        'synsets': set(),
        'min_path_length': float('inf'),
        'synset': None,
        'pos': None
    })

    # For each selected synset, gather all its hypernyms and POS-specific relations
    for selected_syn in selected_synsets_list:
        pos = selected_syn.pos()
        
        # Process hypernyms for all POS types
        hypernym_paths = selected_syn.hypernym_paths()
        for path in hypernym_paths:
            for distance, hypernym_syn in enumerate(reversed(path)):
                hypernym_name = hypernym_syn.name()

                # Record the selected synset that is connected to this hypernym
                common_ancestors_info[hypernym_name]['synsets'].add(selected_syn)
                # Track the minimum path length
                common_ancestors_info[hypernym_name]['min_path_length'] = min(
                    common_ancestors_info[hypernym_name]['min_path_length'],
                    distance
                )
                # Store the synset object and POS
                common_ancestors_info[hypernym_name]['synset'] = hypernym_syn
                common_ancestors_info[hypernym_name]['pos'] = pos

        # For verbs and adjectives, add POS-specific relationships
        if pos in ['v', 'a']:
            related_synsets = get_pos_specific_relations(selected_syn)
            
            for related_syn in related_synsets:
                related_name = related_syn.name()
                
                # Add this as a potential common concept with a standard path length of 1
                # (similar to a direct hypernym)
                common_ancestors_info[related_name]['synsets'].add(selected_syn)
                common_ancestors_info[related_name]['min_path_length'] = min(
                    common_ancestors_info[related_name]['min_path_length'],
                    1
                )
                common_ancestors_info[related_name]['synset'] = related_syn
                common_ancestors_info[related_name]['pos'] = related_syn.pos()

    # Filter for meaningful hypernyms/ancestors
    meaningful_candidates = {}

    for synset_name, info in common_ancestors_info.items():
        shared_synsets_set = info['synsets']
        min_path_length = info['min_path_length']
        syn = info.get('synset')
        pos = info.get('pos')

        # Check if this ancestor is shared by enough selected synsets
        if len(shared_synsets_set) >= MIN_VALID_WORDS_FOR_TOPIC and syn:
            try:
                # Get the depth of this hypernym in the WordNet hierarchy
                # For verbs and adjectives, use a different approach to assess specificity
                if pos == 'n':
                    # For nouns: use max_depth() which tells how specific the concept is
                    syn_depth = syn.max_depth()
                elif pos == 'v':
                    # For verbs: count how many troponyms (specific ways to do the action)
                    troponyms = syn.closure(lambda s: s.troponyms())
                    syn_depth = len(troponyms) // 5 + 1  # Scale to be similar to noun depths
                elif pos in ['a', 'r']:
                    # For adjectives and adverbs: use number of similar terms as a proxy for specificity
                    similar_terms = syn.similar_tos()
                    syn_depth = len(similar_terms) + 1
                else:
                    syn_depth = 1  # Default depth

                # Skip abstract concepts while being more flexible with depth requirement
                # For nouns, avoid 'entity.n' which is too general
                # For verbs, avoid 'change.v' which is too general
                too_abstract = (
                    (pos == 'n' and synset_name.startswith('entity.n')) or
                    (pos == 'v' and synset_name in ['change.v.01', 'act.v.01'])
                )
                
                if syn_depth >= MIN_MEANINGFUL_HYPERNYM_DEPTH and not too_abstract:
                    meaningful_candidates[synset_name] = {
                        'shared_count': len(shared_synsets_set),
                        'min_path_length': min_path_length,
                        'depth': syn_depth,
                        'synset': syn,
                        'pos': pos
                    }
            except Exception as e:
                logger.debug(f"Error processing {synset_name}: {e}")
                continue

    if not meaningful_candidates:
        # Try again with a lower depth threshold before giving up
        for synset_name, info in common_ancestors_info.items():
            shared_synsets_set = info['synsets']
            min_path_length = info['min_path_length']
            syn = info.get('synset')
            pos = info.get('pos')

            if len(shared_synsets_set) >= MIN_VALID_WORDS_FOR_TOPIC and syn:
                try:
                    # Determine depth based on POS
                    if pos == 'n':
                        syn_depth = syn.max_depth()
                    elif pos == 'v':
                        troponyms = syn.closure(lambda s: s.troponyms())
                        syn_depth = len(troponyms) // 5 + 1
                    elif pos in ['a', 'r']:
                        similar_terms = syn.similar_tos()
                        syn_depth = len(similar_terms) + 1
                    else:
                        syn_depth = 1

                    # More lenient depth check
                    too_abstract = (
                        (pos == 'n' and synset_name.startswith('entity.n')) or
                        (pos == 'v' and synset_name in ['change.v.01', 'act.v.01'])
                    )
                    
                    if syn_depth >= MIN_LENIENT_HYPERNYM_DEPTH and not too_abstract:
                        meaningful_candidates[synset_name] = {
                            'shared_count': len(shared_synsets_set),
                            'min_path_length': min_path_length,
                            'depth': syn_depth,
                            'synset': syn,
                            'pos': pos
                        }
                except Exception as e:
                    logger.debug(f"Error in fallback processing for {synset_name}: {e}")
                    continue

    if not meaningful_candidates:
        return [], {"reason": "No meaningful common ancestors found that meet the criteria"}

    # --- Improved Ranking with POS sensitivity ---
    def compute_rank_score(candidate):
        # Base score from number of shared synsets (most important)
        shared_score = candidate['shared_count'] * SHARED_COUNT_WEIGHT

        # Add depth score (moderate importance)
        depth_score = candidate['depth'] * DEPTH_WEIGHT

        # Adjust for path length (least importance, shorter is better)
        path_score = max(0, PATH_SCORE_BASE - candidate['min_path_length'] * PATH_SCORE_MULTIPLIER)

        # Add POS preference if specified
        pos_bonus = 0
        if context_pos and candidate.get('pos') == context_pos:
            pos_bonus = 15  # Bonus for matching preferred POS

        return shared_score + depth_score + path_score + pos_bonus

    # Sort candidates using the scoring function
    sorted_candidates = sorted(
        meaningful_candidates.values(),
        key=compute_rank_score,
        reverse=True  # Higher scores first
    )

    # Get top candidates
    top_candidates = sorted_candidates[:top_n]

    # Extract names, replacing underscores with spaces for readability
    top_words = []
    for candidate in top_candidates:
        synset = candidate['synset']
        # Use the first lemma name (most common form)
        if synset.lemmas():
            top_words.append(synset.lemmas()[0].name().replace('_', ' '))

    return top_words, None

# --- Find Best Fitting Topic for a Single Word ---
def find_best_fitting_topic(single_word, topic_list, context_pos=None):
    """
    Finds which word in a list of topics is most semantically similar
    to a single input word using multiple WordNet similarity metrics.
    Enhanced to handle all parts of speech with context-sensitive weighting.

    Args:
        single_word (str): The word to categorize.
        topic_list (list): A list of topic words (strings).
        context_pos (str, optional): Context-based part of speech preference ('n', 'v', 'a', or 'r')

    Returns:
        str or None: The topic word that best fits the single word, or None if no match found
        dict: Optional debug info explaining why no results were found if return is None
    """
    if not single_word or not topic_list:
        return None, {"reason": "Empty input word or topic list"}

    # Preprocess the input word
    single_word = single_word.lower().strip()

    # Get synsets for the input word with fallbacks
    word_synsets_dict = get_synsets_with_fallbacks(single_word)

    # No synsets found
    if not any(word_synsets_dict.values()):
        return None, {"reason": f"Word '{single_word}' not found in WordNet"}

    # Apply POS preference if context suggests it
    pos_weights = dict(POS_WEIGHTS)
    if context_pos and context_pos in pos_weights:
        # Boost the preferred POS
        pos_weights[context_pos] *= 1.2

    best_topic = None
    highest_score = -1.0
    debug_info = {}

    # Dictionary to hold processed topics for efficient reuse
    processed_topics = {}

    for topic_word in topic_list:
        topic_clean = topic_word.lower().strip()
        if not topic_clean:
            continue

        # Get topic synsets with fallbacks
        if topic_clean in processed_topics:
            topic_synsets_dict = processed_topics[topic_clean]
        else:
            topic_synsets_dict = get_synsets_with_fallbacks(topic_clean)
            processed_topics[topic_clean] = topic_synsets_dict

        # Skip if no synsets found for this topic
        if not any(topic_synsets_dict.values()):
            logger.debug(f"Warning: '{topic_word}' not found in WordNet. Skipping.")
            continue

        # Track best match for this topic
        topic_best_score = 0.0
        topic_best_details = {}

        # Compare synsets across all POS combinations
        for word_pos, word_synsets in word_synsets_dict.items():
            for topic_pos, topic_synsets in topic_synsets_dict.items():
                # Skip if both lists are empty
                if not word_synsets or not topic_synsets:
                    continue
                
                # Calculate POS-specific multiplier
                # Same POS comparisons get full weight, cross-POS are reduced
                pos_multiplier = pos_weights.get(word_pos, 1.0) * (1.0 if word_pos == topic_pos else CROSS_POS_SIMILARITY_PENALTY)
                
                for ws in word_synsets:
                    for ts in topic_synsets:
                        # Get weighted similarity score
                        similarity = calculate_weighted_similarity(ws, ts)
                        
                        if similarity is not None:
                            # Apply POS weighting
                            adjusted_score = similarity * pos_multiplier
                            
                            if adjusted_score > topic_best_score:
                                topic_best_score = adjusted_score
                                topic_best_details = {
                                    "word_synset": ws.name(),
                                    "topic_synset": ts.name(),
                                    "word_pos": word_pos,
                                    "topic_pos": topic_pos,
                                    "raw_similarity": similarity,
                                    "adjusted_score": adjusted_score
                                }

        # Update best overall topic
        if topic_best_score > highest_score:
            highest_score = topic_best_score
            best_topic = topic_word
            debug_info = topic_best_details

    # Return best match if it meets threshold
    if highest_score >= WUP_SIMILARITY_THRESHOLD:
        return best_topic, debug_info
    else:
        return None, {"reason": "No topic with significant similarity found", "highest_score": highest_score}

# --- Flask Application ---
app = Flask(__name__)

# --- Route: Web UI---
@app.route('/')
def index():
    """
    Basic route to serve the HTML form page.
    """
    return render_template('index.html')

# --- Route: Find a topic---
@app.route('/find-topic', methods=['POST'])
def find_topic():
    """
    POST route to find top 3 meaningful topic words from a list of words.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    data = request.get_json()

    if 'words' not in data or not isinstance(data['words'], list):
        return jsonify({"error": "JSON payload must contain a 'words' key with a list of strings"}), 400

    topic_words = data['words']

    # Validate all items are strings
    if not all(isinstance(word, str) for word in topic_words):
        return jsonify({"error": "'words' list must contain only strings"}), 400

    # Filter out empty strings
    topic_words = [word.strip() for word in topic_words if word.strip()]

    if not topic_words:
        return jsonify({"error": "No valid words provided after filtering empty strings"}), 400

    # Get context POS if specified
    context_pos = data.get('context_pos')
    if context_pos and context_pos not in ['n', 'v', 'a', 'r']:
        context_pos = None

    # Get top topics
    suggested_words, debug_info = find_top_topic_words(topic_words, top_n=3, context_pos=context_pos)

    # Include debug info in development mode
    response = {"topic_words": suggested_words}
    if debug_info and app.debug:
        response["debug_info"] = debug_info

    return jsonify(response)

# --- Route: Categorize a Single Word ---
@app.route('/categorize-word', methods=['POST'])
def categorize_word():
    """
    POST route to find the best fitting topic for a single word.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    data = request.get_json()

    # Validate input
    if 'word' not in data or not isinstance(data['word'], str):
        return jsonify({"error": "JSON payload must contain a 'word' key with a string"}), 400

    if 'topics' not in data or not isinstance(data['topics'], list):
        return jsonify({"error": "JSON payload must contain a 'topics' key with a list of strings"}), 400

    if not all(isinstance(topic, str) for topic in data['topics']):
        return jsonify({"error": "'topics' list must contain only strings"}), 400

    single_word = data['word'].strip()
    topic_list = [topic.strip() for topic in data['topics'] if topic.strip()]

    if not single_word:
        return jsonify({"error": "Input word is empty after stripping whitespace"}), 400

    if not topic_list:
        return jsonify({"error": "No valid topics provided after filtering empty strings"}), 400

    # Get context POS if specified
    context_pos = data.get('context_pos')
    if context_pos and context_pos not in ['n', 'v', 'a', 'r']:
        context_pos = None

    # Call the function with improved error handling
    best_topic, debug_info = find_best_fitting_topic(single_word, topic_list, context_pos=context_pos)

    # Build the response
    response = {"best_topic": best_topic}

    # Include debug information in development mode
    if debug_info and app.debug:
        response["debug_info"] = debug_info

    if best_topic is None:
        response["message"] = "Could not find a significantly similar topic."
        return jsonify(response), 204  # No Content - clearer HTTP status
    
    return jsonify(response), 200

# --- New Route: Context-Based Analysis ---
@app.route('/analyze-context', methods=['POST'])
def analyze_context():
    """
    POST route to analyze words in a contextual manner, determining the dominant POS
    and finding appropriate topics based on that context.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    data = request.get_json()

    # Validate input
    if 'words' not in data or not isinstance(data['words'], list):
        return jsonify({"error": "JSON payload must contain a 'words' key with a list of strings"}), 400

    words = [word.strip() for word in data['words'] if isinstance(word, str) and word.strip()]

    if not words:
        return jsonify({"error": "No valid words provided after filtering"}), 400

    # Determine dominant POS in the word set
    pos_counts = {'n': 0, 'v': 0, 'a': 0, 'r': 0}
    word_pos_mapping = {}
    
    for word in words:
        synsets_by_pos = get_synsets_with_fallbacks(word)
        word_pos = None
        max_synsets = 0
        
        for pos, synsets in synsets_by_pos.items():
            if len(synsets) > max_synsets:
                max_synsets = len(synsets)
                word_pos = pos
        
        if word_pos:
            pos_counts[word_pos] += 1
            word_pos_mapping[word] = word_pos
    
    # Determine dominant POS
    dominant_pos = max(pos_counts.items(), key=lambda x: x[1])[0] if any(pos_counts.values()) else None
    
    # Find topics with context awareness
    suggested_topics, debug_info = find_top_topic_words(words, top_n=3, context_pos=dominant_pos)
    
    # Build response
    response = {
        "dominant_pos": dominant_pos,
        "pos_distribution": pos_counts,
        "suggested_topics": suggested_topics,
        "word_pos_mapping": word_pos_mapping
    }
    
    if debug_info and app.debug:
        response["debug_info"] = debug_info
    
    return jsonify(response), 200

# --- Route: Clear Cache ---
@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """
    POST route to clear the similarity calculation cache.
    Useful for benchmarking or if memory usage becomes a concern.
    """
    global similarity_cache
    
    # Record stats before clearing
    cache_size = len(similarity_cache)
    similarity_cache = {}
    
    return jsonify({
        "message": "Similarity cache cleared",
        "previous_size": cache_size
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')