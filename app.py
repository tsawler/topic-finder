import nltk
from nltk.corpus import wordnet
from collections import defaultdict
from flask import Flask, request, jsonify, render_template

# --- NLTK Data Download ---
# WordNet is a lexical database of English words that groups nouns, verbs, adjectives, and adverbs into sets of
# cognitive synonyms (synsets), each expressing a distinct concept. It records semantic relationships between these synsets.
# OMW-1.4 (Open Multilingual WordNet) provides WordNet data for languages other than English.
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("NLTK data not found. Downloading wordnet and omw-1.4...")
    try:
        nltk.download('wordnet')
        nltk.download('omw-1.4') # Often used in conjunction with WordNet for multilingual support
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        print("Please ensure you have an internet connection and try again.")

# --- Topic Finding Function ---
def find_top_topic_words(topic_words, top_n=3):
    """
    Finds the top n meaningful descriptive words for a list of topic words
    by performing Word Sense Disambiguation (WSD) using Wu-Palmer similarity
    and then identifying common, relatively specific hypernyms using WordNet.
    
    This function works by:
    1. Finding the most likely meaning (synset) for each word based on how similar it is to other words
    2. Identifying common ancestor concepts (hypernyms) shared by these words
    3. Selecting the most specific and relevant common concepts
    
    For example, given ["apple", "banana", "orange"], it might return ["fruit", "produce", "food"]

    Args:
        topic_words (list): A list of strings, where each string is a word
                            from the topic.
        top_n (int): Number of top topic words to return (default: 3)

    Returns:
        list: A list of the top n most representative common hypernyms found,
              or the original word if only one is provided,
              or empty list if no meaningful common hypernyms are identified.
    """
    if not topic_words:
        return []
    if len(topic_words) == 1:
        # If only one word, return the word itself.
        return [topic_words[0]]

    # --- Step 1: Word Sense Disambiguation (WSD) ---
    # WSD is the process of identifying which sense (meaning) of a word is used in a given context
    # Here we select the most likely synset for each word based on its similarity to the synsets of other words
    # A synset is a set of cognitive synonyms (words that express the same concept)
    selected_synsets = {} # {word: selected_synset}
    word_synsets_map = {} # {word: list of noun synsets}

    # Get all noun synsets for each word first
    # A noun synset represents a group of nouns that express the same concept
    valid_words = []
    for word in topic_words:
        synsets = wordnet.synsets(word.lower(), pos=wordnet.NOUN)
        if synsets:
            word_synsets_map[word.lower()] = synsets
            valid_words.append(word.lower())
        # else:
            # print(f"Warning: '{word}' not found as a noun in WordNet. Skipping for WSD.")

    if len(valid_words) < 2:
        # Need at least two words with noun synsets to find common hypernyms based on similarity
        # If we have only one word, the initial check handles it. If multiple words but only one
        # has noun synsets, we can't perform pairwise WSD.
        return []

    # Calculate scores for each synset of each valid word based on Wu-Palmer similarity
    # Wu-Palmer similarity measures how similar two word senses are, based on:
    # - The depth of their lowest common ancestor in the WordNet hierarchy
    # - The depths of the individual synsets
    # Higher Wu-Palmer similarity (closer to 1.0) means words are more semantically related
    synset_scores = defaultdict(float) # {synset_object: score}
    synset_to_word_map = {} # {synset_object: word} for easy lookup later

    for word in valid_words:
        synsets = word_synsets_map[word]
        for syn in synsets:
            synset_to_word_map[syn] = word # Store mapping

            current_syn_score = 0.0
            for other_word in valid_words:
                if word == other_word:
                    continue # Don't compare a word's synsets to itself

                other_synsets = word_synsets_map[other_word]
                max_sim_to_other_word = 0.0

                for other_syn in other_synsets:
                    # Use Wu-Palmer similarity, which balances depth and path distance
                    # This measure gives a value between 0 (not similar) and 1 (identical)
                    similarity = syn.wup_similarity(other_syn)
                    if similarity is not None: # wup_similarity returns None if no common subsumer
                        max_sim_to_other_word = max(max_sim_to_other_word, similarity)

                current_syn_score += max_sim_to_other_word

            synset_scores[syn] = current_syn_score


    # Select the synset with the highest score for each word
    # This is the sense that is, on average, most similar to the most related sense of other words
    # For example, "apple" could refer to a fruit or a technology company - we want to select
    # the meaning that's most similar to the other words in the list
    word_best_synset_score = {} # {word: max_score}
    word_best_synset = {} # {word: best_synset}

    for syn, score in synset_scores.items():
        word = synset_to_word_map[syn]
        # If this is the first synset for the word or has a higher score
        if word not in word_best_synset_score or score > word_best_synset_score[word]:
             word_best_synset_score[word] = score
             word_best_synset[word] = syn

    # Collect the selected synsets that will be used for finding common hypernyms
    selected_synsets_list = list(word_best_synset.values())

    if len(selected_synsets_list) < 2:
        # Need at least two selected senses to find a common ancestor
        return []

    # --- Step 2: Find Common Hypernyms of Selected Synsets ---
    # A hypernym is a word with a broader meaning that includes more specific words
    # For example, "fruit" is a hypernym of "apple", "vehicle" is a hypernym of "car"
    # Now, we only consider hypernyms that are ancestors of the *selected* synsets
    common_ancestors_info = defaultdict(lambda: {'synsets': set(), 'min_path_length': float('inf')})

    for selected_syn in selected_synsets_list:
        # Get all hypernyms (ancestors) for this selected synset up the tree
        # A hypernym_path is a list of synsets from the root of the taxonomy down to this synset
        hypernym_paths = selected_syn.hypernym_paths()

        for path in hypernym_paths:
            # Iterate through the path to find hypernyms and their distance from the selected synset
            # We reverse the path to start from the root (most general) and move down
            for distance, hypernym_syn in enumerate(reversed(path)):
                hypernym_name = hypernym_syn.name()

                # Record the selected synset that is an ancestor of this hypernym
                common_ancestors_info[hypernym_name]['synsets'].add(selected_syn)
                # Update the minimum path length from any of the selected synsets to this hypernym
                # Shorter path length means the hypernym is closer to the original word
                common_ancestors_info[hypernym_name]['min_path_length'] = min(
                    common_ancestors_info[hypernym_name]['min_path_length'],
                    distance
                )
                # Store the synset object for later use
                common_ancestors_info[hypernym_name]['synset'] = hypernym_syn

    # Filter for hypernyms that are common to more than one *selected* synset
    # and exclude the 'entity' root and very shallow terms which are too general
    # For example, we want to avoid returning "entity", "object" or other overly general concepts
    meaningful_candidates = {}
    # Define a minimum depth for the hypernym itself to exclude overly general terms
    # Depth in WordNet represents how specific a concept is:
    # - entity (depth 0) - extremely general
    # - physical_entity (depth 1) - still very general
    # - object (depth 2) - too general to be useful
    # - depths >= 3 start to become more meaningful for categorization
    MIN_MEANINGFUL_HYPERNYM_DEPTH = 3 # Excludes entity (0), physical_entity (1), object (2) - adjust as needed

    min_synsets_for_common = 2 # A common ancestor must be shared by at least two selected synsets

    for synset_name, info in common_ancestors_info.items():
        shared_synsets_set = info['synsets']
        min_path_length = info['min_path_length']
        syn = info.get('synset')

        # Check if shared by at least the minimum number of distinct selected synsets
        # And make sure the hypernym's depth is above the minimum threshold
        if len(shared_synsets_set) >= min_synsets_for_common and syn:
             try:
                 # max_depth returns the length of the longest hypernym path to the root
                 # Higher values indicate more specific concepts
                 syn_depth = syn.max_depth()
                 if syn_depth >= MIN_MEANINGFUL_HYPERNYM_DEPTH:
                     meaningful_candidates[synset_name] = {
                         'shared_count': len(shared_synsets_set), # Number of selected synsets it's common to
                         'min_path_length': min_path_length, # Minimum distance from a selected synset
                         'depth': syn_depth, # Depth of the hypernym itself
                         'synset': syn
                     }
             except ValueError:
                 # Handle cases where max_depth might fail (rare)
                 continue


    if not meaningful_candidates:
        # Fallback: if no meaningful common ancestors found for selected synsets, return empty list
        return []

    # --- Ranking ---
    # Rank candidates:
    # 1. Prioritize by the number of shared *selected synsets* (higher is better)
    #    This favors concepts that apply to more of our input words
    # 2. If counts are tied, prioritize by the hypernym's depth (higher depth is more specific, better)
    #    This helps avoid overly general concepts
    # 3. If still tied, prioritize by the minimum path length (shorter path from a selected synset is better)
    #    This favors concepts that are closer to our original words

    sorted_candidates = sorted(
        meaningful_candidates.values(),
        key=lambda x: (-x['shared_count'], -x['depth'], x['min_path_length']) # Use negative for descending sort
    )

    # Get the top n candidates or all if less than n are available
    top_candidates = sorted_candidates[:top_n]

    # Extract the name (lemma) of each top synset, replacing underscores with spaces
    # A lemma is a canonical form of a word - in WordNet, multi-word concepts use underscores
    top_words = [candidate['synset'].lemmas()[0].name().replace('_', ' ') for candidate in top_candidates]

    return top_words

# --- Find Best Fitting Topic for a Single Word ---
def find_best_fitting_topic(single_word, topic_list):
    """
    Finds which word in a list of topics is most semantically similar
    to a single input word using WordNet Wu-Palmer similarity.
    
    This function:
    1. Gets the possible meanings (synsets) of the input word
    2. Gets the possible meanings of each topic word
    3. Calculates the similarity between the input word and each topic
    4. Returns the topic with the highest similarity score (if above a threshold)
    
    For example, given "apple" as the word and ["fruit", "vehicle", "sport"] as topics,
    it would return "fruit" as the best matching topic.

    Args:
        single_word (str): The word to categorize.
        topic_list (list): A list of topic words (strings).

    Returns:
        str or None: The topic word from the list that best fits the single word,
                     or None if no meaningful comparison could be made or
                     no topic word was found to be significantly similar.
    """
    if not single_word or not topic_list:
        return None

    # Get noun synsets for the single word
    # Consider both noun and adjective senses might be relevant for categorization
    # For example, "red" could be categorized under "color" even though it's an adjective
    single_word_synsets = wordnet.synsets(single_word.lower(), pos=wordnet.NOUN) # Start with Noun
    if not single_word_synsets: # If no noun synsets, try adjectives
         single_word_synsets = wordnet.synsets(single_word.lower(), pos=wordnet.ADJ)

    if not single_word_synsets:
        # print(f"Warning: '{single_word}' not found as a noun or adjective in WordNet.")
        return None # Cannot find synsets for the main word

    best_topic = None
    highest_similarity = -1.0 # Initialize with a value lower than any possible similarity

    # Use a threshold to avoid returning a topic for very low similarity scores
    # Wu-Palmer similarity ranges from 0 to 1, where 1 means identical concepts
    # 0.2 is a relatively low threshold, but helps filter out completely unrelated topics
    SIGNIFICANCE_THRESHOLD = 0.2 # Example threshold, can be adjusted

    for topic_word in topic_list:
        # Consider both noun and adjective senses for topic words too
        topic_word_synsets = wordnet.synsets(topic_word.lower(), pos=wordnet.NOUN) # Start with Noun
        if not topic_word_synsets: # If no noun synsets, try adjectives
             topic_word_synsets = wordnet.synsets(topic_word.lower(), pos=wordnet.ADJ)


        if not topic_word_synsets:
            # print(f"Warning: '{topic_word}' not found as a noun or adjective in WordNet. Skipping.")
            continue # Cannot compare if topic word has no synsets

        current_max_similarity = 0.0

        # Calculate the maximum Wu-Palmer similarity between any synset pair
        # We're looking for the highest possible semantic similarity between the word and the topic
        for sw_syn in single_word_synsets:
            for tw_syn in topic_word_synsets:
                # Wu-Palmer similarity works best with nouns, but let's try other types if included
                # wup_similarity primarily defined for nouns/verbs. Let's check part of speech.
                # If synsets are of different POS, similarity might return None.
                if sw_syn.pos() == tw_syn.pos() and (sw_syn.pos() == wordnet.NOUN or sw_syn.pos() == wordnet.VERB):
                     similarity = sw_syn.wup_similarity(tw_syn)
                     if similarity is not None:
                          current_max_similarity = max(current_max_similarity, similarity)
                # For Adj/Adv, other metrics like path_similarity or lch_similarity might be used,
                # but wup_similarity is generally robust and common for Nouns/Verbs which
                # are typical topic words. Sticking primarily to WUP for simplicity and consistency.
                # We've already filtered to Noun/Adj senses initially. WUP is usually only between N/N or V/V.
                elif sw_syn.pos() == wordnet.NOUN and tw_syn.pos() == wordnet.NOUN:
                     similarity = sw_syn.wup_similarity(tw_syn)
                     if similarity is not None:
                          current_max_similarity = max(current_max_similarity, similarity)


        # Update best topic if current word is more similar
        if current_max_similarity > highest_similarity:
             highest_similarity = current_max_similarity
             best_topic = topic_word


    # Return the best topic only if the highest similarity is above a threshold
    # This prevents returning topics that are only very weakly related to the input word
    if highest_similarity > SIGNIFICANCE_THRESHOLD:
        return best_topic
    else:
        return None # No topic found with significant similarity

# --- Flask Application ---
app = Flask(__name__)

# --- Route: Web UI---
@app.route('/')
def index():
    """
    Basic route to serve the HTML form page.
    """
    return render_template('index.html') # Render the index.html template

# --- Route: Find a topic---
@app.route('/find-topic', methods=['POST'])
def find_topic():
    """
    POST route to find top 3 meaningful topic words from a list of words.
    This endpoint uses semantic analysis to find common themes among a set of words.
    
    Example use case: Analyzing keywords from a text to determine its main topics.
    
    Expects a JSON payload like: {"words": ["apple", "banana", "orange"]}
    Returns a JSON response like: {"topic_words": ["fruit", "produce", "food"]} or {"topic_words": []}
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415 # Unsupported Media Type

    data = request.get_json()

    if 'words' not in data or not isinstance(data['words'], list):
        return jsonify({"error": "JSON payload must contain a 'words' key with a list of strings"}), 400 # Bad Request

    topic_words = data['words']

    # Basic validation to ensure all items in the list are strings
    if not all(isinstance(word, str) for word in topic_words):
         return jsonify({"error": "'words' list must contain only strings"}), 400 # Bad Request

    # Use the updated function to get top 3 topics
    suggested_words = find_top_topic_words(topic_words, top_n=3)

    return jsonify({"topic_words": suggested_words})


# --- Route: Categorize a Single Word ---
@app.route('/categorize-word', methods=['POST'])
def categorize_word():
    """
    POST route to find the best fitting topic for a single word
    from a list of provided topics using WordNet similarity.
    
    Example use case: Determining which category a new word belongs to,
    given a set of predefined categories.
    
    Expects a JSON payload like: {"word": "apple", "topics": ["fruit", "vehicle", "sport"]}
    Returns a JSON response like: {"best_topic": "fruit"} or {"best_topic": null}
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    data = request.get_json()

    # Validate input structure and types
    if 'word' not in data or not isinstance(data['word'], str):
         return jsonify({"error": "JSON payload must contain a 'word' key with a single string"}), 400

    if 'topics' not in data or not isinstance(data['topics'], list) or not all(isinstance(topic, str) for topic in data['topics']):
         return jsonify({"error": "JSON payload must contain a 'topics' key with a list of strings"}), 400

    single_word = data['word'].strip()
    topic_list = [topic.strip() for topic in data['topics'] if topic.strip()] # Clean and filter empty topics

    # Basic check for empty word or topic list after cleaning
    if not single_word or not topic_list:
         # Depending on requirements, could return 400, but 200 with null indicates no match found for empty input
         return jsonify({"best_topic": None, "message": "Input word or valid topic list is empty"}), 200

    # Call the new function
    best_topic = find_best_fitting_topic(single_word, topic_list)

    # Return the result
    if best_topic:
        # Find the original casing from the input list if available
        # This preserves user-provided capitalization in the response
        original_topic_casing = next((topic for topic in data['topics'] if topic.lower() == best_topic.lower()), best_topic)
        return jsonify({"best_topic": original_topic_casing}), 200
    else:
        # If no significant topic was found
        return jsonify({"best_topic": None, "message": "Could not find a significantly similar topic."}), 200


if __name__ == '__main__':
    # Run the Flask app
    # debug=True allows for automatic reloading on code changes and shows detailed errors
    # In a production environment, set debug=False and use a production server like Gunicorn
    # This block is primarily for running the script directly for testing/debugging outside Docker.
    # The host='0.0.0.0' makes the server accessible externally, useful for Docker or other environments.
    app.run(debug=True, host='0.0.0.0')