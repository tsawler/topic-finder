import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from flask import Flask, request, jsonify, render_template
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- NLTK Data Download ---
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')  # For lemmatization
except LookupError:
    logger.info("NLTK data not found. Downloading required resources...")
    try:
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('punkt')  # For lemmatization
        logger.info("Download complete.")
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {e}")
        logger.error("Please ensure you have an internet connection and try again.")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define constants
# Higher threshold for more confident matching
WUP_SIMILARITY_THRESHOLD = 0.35
# Allow more flexibility in hypernym depth
MIN_MEANINGFUL_HYPERNYM_DEPTH = 2

# Common food-related terms and categories
FOOD_TERMS = ['food', 'nutrient', 'aliment', 'nourishment', 'sustenance', 'edible', 'comestible', 'produce']
KNOWN_FOOD_ITEMS = ['fruit', 'vegetable', 'meat', 'grain', 'dairy', 'bread', 'cereal', 'soup', 'juice']

# --- Helper Functions ---
def check_in_food_taxonomy(synset):
    """
    Check if a synset falls under the food taxonomy in WordNet.
    
    Args:
        synset: The WordNet synset to check
        
    Returns:
        bool: True if the synset is food-related, False otherwise
    """
    # Quick initial check using lemma names
    for lemma in synset.lemmas():
        if lemma.name().lower() in FOOD_TERMS:
            return True
    
    # Check hypernym paths
    for hypernym_path in synset.hypernym_paths():
        for hypernym in hypernym_path:
            for lemma in hypernym.lemmas():
                if lemma.name().lower() in FOOD_TERMS:
                    return True
    
    return False

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
        
    if pos is None or pos == 'a':
        result['a'] = wordnet.synsets(word, pos=wordnet.ADJ)
        result['a'].extend(wordnet.synsets(word, pos='s'))  # Include satellite adjectives
        
    if pos is None or pos == 'r':
        result['r'] = wordnet.synsets(word, pos=wordnet.ADV)
    
    return result

# --- Topic Finding Function ---
def find_top_topic_words(topic_words, top_n=3, min_synsets_for_common=2):
    """
    Finds the top n meaningful descriptive words for a list of topic words.
    
    Args:
        topic_words (list): A list of strings, where each string is a word from the topic.
        top_n (int): Number of top topic words to return (default: 3)
        min_synsets_for_common (int): Minimum number of synsets that must share a hypernym
                                     to consider it a common topic (default: 2)

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
    
    # Preprocess words: lowercase and lemmatize
    processed_words = []
    for word in topic_words:
        word_lower = word.lower().strip()
        # Lemmatize as noun (default)
        lemma = lemmatizer.lemmatize(word_lower, pos='n')
        processed_words.append(lemma)
    
    # Get all noun synsets for each word first
    word_synsets_map = {}
    valid_words = []
    
    for word in processed_words:
        synsets = wordnet.synsets(word, pos=wordnet.NOUN)
        
        # Try singular form if no synsets found and word might be plural
        if not synsets and word.endswith('s'):
            singular = lemmatizer.lemmatize(word, pos='n')
            if singular != word:
                synsets = wordnet.synsets(singular, pos=wordnet.NOUN)
        
        if synsets:
            word_synsets_map[word] = synsets
            valid_words.append(word)
        else:
            logger.debug(f"'{word}' not found as a noun in WordNet. Skipping for WSD.")
    
    # Check if we have enough valid words to proceed
    if len(valid_words) < 2:
        return [], {"reason": f"Not enough words with noun synsets found (found {len(valid_words)})"}
    
    # --- Improved Word Sense Disambiguation ---
    # Instead of just summing similarity scores, we'll use a context-based approach
    # that considers the highest average similarity for each sense
    synset_scores = defaultdict(float)
    synset_to_word_map = {}
    
    for word in valid_words:
        synsets = word_synsets_map[word]
        for syn in synsets:
            synset_to_word_map[syn] = word
            
            # Calculate average similarity to most similar sense of other words
            similarities = []
            for other_word in valid_words:
                if word == other_word:
                    continue
                
                other_synsets = word_synsets_map[other_word]
                max_sim_to_other_word = 0.0
                
                for other_syn in other_synsets:
                    # Only compare synsets with same POS
                    if syn.pos() == other_syn.pos():
                        similarity = syn.wup_similarity(other_syn)
                        if similarity is not None:
                            max_sim_to_other_word = max(max_sim_to_other_word, similarity)
                
                if max_sim_to_other_word > 0:  # Only count if there was some similarity
                    similarities.append(max_sim_to_other_word)
            
            # Use average similarity instead of sum (more stable with varying input sizes)
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
    
    if len(selected_synsets_list) < min_synsets_for_common:
        return [], {"reason": f"Not enough synsets ({len(selected_synsets_list)}) found with good disambiguation scores"}
    
    # --- Find Common Hypernyms with Improved Algorithm ---
    # Track hypernym info and the synsets they're connected to
    common_ancestors_info = defaultdict(lambda: {
        'synsets': set(), 
        'min_path_length': float('inf'),
        'synset': None
    })
    
    # For each selected synset, gather all its hypernyms
    for selected_syn in selected_synsets_list:
        # Get paths for this synset
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
                # Store the synset object
                common_ancestors_info[hypernym_name]['synset'] = hypernym_syn
    
    # Filter for meaningful hypernyms
    meaningful_candidates = {}
    
    for synset_name, info in common_ancestors_info.items():
        shared_synsets_set = info['synsets']
        min_path_length = info['min_path_length']
        syn = info.get('synset')
        
        # Check if this hypernym is shared by enough selected synsets
        if len(shared_synsets_set) >= min_synsets_for_common and syn:
            try:
                # Get the depth of this hypernym in the WordNet hierarchy
                syn_depth = syn.max_depth()
                
                # Skip abstract concepts while being more flexible with depth requirement
                if syn_depth >= MIN_MEANINGFUL_HYPERNYM_DEPTH and not synset_name.startswith('entity.n'):
                    meaningful_candidates[synset_name] = {
                        'shared_count': len(shared_synsets_set),
                        'min_path_length': min_path_length,
                        'depth': syn_depth,
                        'synset': syn
                    }
            except Exception as e:
                logger.debug(f"Error getting depth for {synset_name}: {e}")
                continue
    
    if not meaningful_candidates:
        # Try again with a lower depth threshold before giving up
        for synset_name, info in common_ancestors_info.items():
            shared_synsets_set = info['synsets']
            min_path_length = info['min_path_length']
            syn = info.get('synset')
            
            if len(shared_synsets_set) >= min_synsets_for_common and syn:
                try:
                    syn_depth = syn.max_depth()
                    
                    # More lenient depth check, but still avoid 'entity'
                    if syn_depth >= 1 and not synset_name.startswith('entity.n'):
                        meaningful_candidates[synset_name] = {
                            'shared_count': len(shared_synsets_set),
                            'min_path_length': min_path_length,
                            'depth': syn_depth,
                            'synset': syn
                        }
                except Exception:
                    continue
    
    if not meaningful_candidates:
        return [], {"reason": "No meaningful common hypernyms found that meet the criteria"}
    
    # --- Improved Ranking ---
    # Weight the factors: Shared count (highest importance), then depth, then path length
    # Use a scoring formula that better balances these factors
    
    def compute_rank_score(candidate):
        # Base score from number of shared synsets (most important)
        shared_score = candidate['shared_count'] * 100
        
        # Add depth score (moderate importance)
        depth_score = candidate['depth'] * 10
        
        # Adjust for path length (least important, shorter is better)
        # Convert to a positive factor (smaller path length = higher score)
        path_score = max(0, 50 - candidate['min_path_length'] * 5)
        
        return shared_score + depth_score + path_score
    
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
def find_best_fitting_topic(single_word, topic_list):
    """
    Finds which word in a list of topics is most semantically similar
    to a single input word using WordNet similarity.
    
    Args:
        single_word (str): The word to categorize.
        topic_list (list): A list of topic words (strings).

    Returns:
        str or None: The topic word that best fits the single word, or None if no match found
        dict: Optional debug info explaining why no results were found if return is None
    """
    if not single_word or not topic_list:
        return None, {"reason": "Empty input word or topic list"}
    
    # Preprocess the input word
    single_word = single_word.lower().strip()
    lemmatized_word = lemmatizer.lemmatize(single_word, pos='n')
    
    # Check if this is a known food item
    is_known_food = lemmatized_word in KNOWN_FOOD_ITEMS
    
    # Get synsets for the input word with fallbacks
    word_synsets_dict = get_synsets_with_fallbacks(lemmatized_word)
    
    # Flatten the synsets for processing
    single_word_synsets = []
    for pos_synsets in word_synsets_dict.values():
        single_word_synsets.extend(pos_synsets)
    
    if not single_word_synsets:
        return None, {"reason": f"Word '{single_word}' not found in WordNet"}
    
    # Track if any input word synset is food-related
    word_is_food_related = any(check_in_food_taxonomy(syn) for syn in word_synsets_dict['n'])
    
    best_topic = None
    highest_similarity = -1.0
    best_topic_pos = None
    word_pos = None
    debug_info = {}
    
    # Dictionary to hold processed topics for efficient reuse
    processed_topics = {}
    
    for topic_word in topic_list:
        topic_clean = topic_word.lower().strip()
        if not topic_clean:
            continue
            
        # Check if this is a food-related topic
        is_food_topic = topic_clean in FOOD_TERMS
        
        # Adjust threshold for food categories
        if is_food_topic and (word_is_food_related or is_known_food):
            # Lower threshold for food categories when comparing with food items
            similarity_threshold = WUP_SIMILARITY_THRESHOLD * 0.8
            logger.debug(f"Using lower threshold ({similarity_threshold}) for food topic: {topic_clean}")
        else:
            similarity_threshold = WUP_SIMILARITY_THRESHOLD
        
        # Get topic synsets with fallbacks
        if topic_clean in processed_topics:
            topic_synsets_dict = processed_topics[topic_clean]
        else:
            topic_synsets_dict = get_synsets_with_fallbacks(topic_clean)
            processed_topics[topic_clean] = topic_synsets_dict
        
        # Flatten topic synsets for easier reference
        topic_synsets = []
        for pos_synsets in topic_synsets_dict.values():
            topic_synsets.extend(pos_synsets)
            
        # If no synsets found for this topic, skip it
        if not topic_synsets:
            logger.debug(f"Warning: '{topic_word}' not found in WordNet. Skipping.")
            continue
        
        current_max_similarity = 0.0
        current_pos_match = None
        current_word_pos = None
        
        # Track food-related match bonus
        food_match_bonus = 0.0
        
        # Check food taxonomy match if applicable
        if is_food_topic and word_synsets_dict['n']:
            # Check if any of the word's noun synsets fall under food taxonomy
            for word_syn in word_synsets_dict['n']:
                if check_in_food_taxonomy(word_syn):
                    # Apply a bonus for food-related matches
                    food_match_bonus = 0.1
                    logger.debug(f"Applied food match bonus for '{single_word}' and '{topic_clean}'")
                    break
        
        # Compare synsets with same POS first (more accurate)
        for pos_key, word_pos_synsets in word_synsets_dict.items():
            topic_pos_synsets = topic_synsets_dict.get(pos_key, [])
            
            for ws in word_pos_synsets:
                for ts in topic_pos_synsets:
                    # Ensure valid comparison within same POS
                    similarity = ws.wup_similarity(ts)
                    
                    if similarity is not None:
                        # Apply food match bonus if applicable
                        adjusted_similarity = similarity + food_match_bonus
                        
                        if adjusted_similarity > current_max_similarity:
                            current_max_similarity = adjusted_similarity
                            current_pos_match = "same"
                            current_word_pos = pos_key
        
        # Try cross-POS comparisons if needed
        if current_max_similarity < similarity_threshold:
            for word_pos_key, word_pos_synsets in word_synsets_dict.items():
                for topic_pos_key, topic_pos_synsets in topic_synsets_dict.items():
                    if word_pos_key != topic_pos_key:  # Only cross-POS
                        for ws in word_pos_synsets:
                            for ts in topic_pos_synsets:
                                # Use path_similarity for cross-POS (more general)
                                similarity = ws.path_similarity(ts)
                                
                                if similarity is not None:
                                    # Penalize cross-POS matches slightly
                                    adjusted_similarity = (similarity * 0.8) + food_match_bonus
                                    
                                    if adjusted_similarity > current_max_similarity:
                                        current_max_similarity = adjusted_similarity
                                        current_pos_match = "cross"
                                        current_word_pos = f"{word_pos_key}->{topic_pos_key}"
        
        # Special case for known food items
        if is_known_food and is_food_topic and current_max_similarity > 0:
            # Further boost known food items matching food topics
            current_max_similarity += 0.1
            logger.debug(f"Applied known food item boost for '{single_word}' and '{topic_clean}'")
        
        # Update best topic if this one is more similar
        if current_max_similarity > highest_similarity:
            highest_similarity = current_max_similarity
            best_topic = topic_word
            best_topic_pos = current_pos_match
            word_pos = current_word_pos
            
            # Update debug info
            debug_info = {
                "similarity_score": highest_similarity,
                "pos_match_type": best_topic_pos,
                "word_pos": word_pos,
                "food_related": word_is_food_related or is_known_food
            }
    
    # Return the best topic if significant similarity found
    if highest_similarity >= WUP_SIMILARITY_THRESHOLD:
        return best_topic, debug_info
    else:
        # Special case: If the word is a known food item and "food" is among topics, return it
        for topic in topic_list:
            if topic.lower() in FOOD_TERMS and is_known_food:
                logger.debug(f"Forced match of known food item '{single_word}' to '{topic}'")
                return topic, {
                    "reason": "Matched based on known food item list",
                    "forced_match": True,
                    "highest_score": highest_similarity
                }
        
        return None, {"reason": "No topic found with significant similarity", "highest_score": highest_similarity}

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
    
    # Get top topics
    suggested_words, debug_info = find_top_topic_words(topic_words, top_n=3)
    
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
    
    # Call the function with improved error handling
    best_topic, debug_info = find_best_fitting_topic(single_word, topic_list)
    
    # Build the response
    response = {"best_topic": best_topic}
    
    # Include debug information in development mode
    if debug_info and app.debug:
        response["debug_info"] = debug_info
    
    if best_topic is None:
        response["message"] = "Could not find a significantly similar topic."
    
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')