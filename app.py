import nltk
from nltk.corpus import wordnet
from collections import defaultdict
from flask import Flask, request, jsonify, render_template

# --- NLTK Data Download ---
# Ensure WordNet and OMW are downloaded (run these lines once)
# This block will attempt to download the necessary data if not found.
# In a production environment, you might want to handle this differently
# (e.g., download during setup or deployment).
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("NLTK data not found. Downloading wordnet and omw-1.4...")
    try:
        nltk.download('wordnet')
        nltk.download('omw-1.4') # Often used in conjunction
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        print("Please ensure you have an internet connection and try again.")

# --- Topic Finding Function (original - finds common topic for a list of words) ---
# This function remains as it was.
def find_top_topic_words(topic_words, top_n=3):
    """
    Finds the top n meaningful descriptive words for a list of topic words
    by performing Word Sense Disambiguation (WSD) using Wu-Palmer similarity
    and then identifying common, relatively specific hypernyms using WordNet.

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
    # Select the most likely synset for each word based on its similarity to the synsets of other words.
    selected_synsets = {} # {word: selected_synset}
    word_synsets_map = {} # {word: list of noun synsets}

    # Get all noun synsets for each word first
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
    # A synset's score is the sum of its maximum similarities to any synset of each other word.
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
                    similarity = syn.wup_similarity(other_syn)
                    if similarity is not None: # wup_similarity returns None if no common subsumer
                        max_sim_to_other_word = max(max_sim_to_other_word, similarity)

                current_syn_score += max_sim_to_other_word

            synset_scores[syn] = current_syn_score


    # Select the synset with the highest score for each word
    # This is the sense that is, on average, most similar to the most related sense of other words.
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
    # Now, we only consider hypernyms that are ancestors of the *selected* synsets.
    common_ancestors_info = defaultdict(lambda: {'synsets': set(), 'min_path_length': float('inf')})

    for selected_syn in selected_synsets_list:
        # Get all hypernyms (ancestors) for this selected synset up the tree
        hypernym_paths = selected_syn.hypernym_paths()

        for path in hypernym_paths:
            # Iterate through the path to find hypernyms and their distance from the selected synset
            for distance, hypernym_syn in enumerate(reversed(path)):
                hypernym_name = hypernym_syn.name()

                # Record the selected synset that is an ancestor of this hypernym
                common_ancestors_info[hypernym_name]['synsets'].add(selected_syn)
                # Update the minimum path length from any of the selected synsets to this hypernym
                common_ancestors_info[hypernym_name]['min_path_length'] = min(
                    common_ancestors_info[hypernym_name]['min_path_length'],
                    distance
                )
                # Store the synset object for later use
                common_ancestors_info[hypernym_name]['synset'] = hypernym_syn

    # Filter for hypernyms that are common to more than one *selected* synset
    # and exclude the 'entity' root and very shallow terms which are too general
    meaningful_candidates = {}
    # Define a minimum depth for the hypernym itself to exclude overly general terms
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
    # 2. If counts are tied, prioritize by the hypernym's depth (higher depth is more specific, better)
    # 3. If still tied, prioritize by the minimum path length (shorter path from a selected synset is better)

    sorted_candidates = sorted(
        meaningful_candidates.values(),
        key=lambda x: (-x['shared_count'], -x['depth'], x['min_path_length']) # Use negative for descending sort
    )

    # Get the top n candidates or all if less than n are available
    top_candidates = sorted_candidates[:top_n]

    # Extract the name (lemma) of each top synset, replacing underscores with spaces
    top_words = [candidate['synset'].lemmas()[0].name().replace('_', ' ') for candidate in top_candidates]

    return top_words

# --- New Function: Find Best Fitting Topic for a Single Word ---
def find_best_fitting_topic(single_word, topic_list):
    """
    Finds which word in a list of topics is most semantically similar
    to a single input word using WordNet Wu-Palmer similarity.

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
    single_word_synsets = wordnet.synsets(single_word.lower(), pos=wordnet.NOUN) # Start with Noun
    if not single_word_synsets: # If no noun synsets, try adjectives
         single_word_synsets = wordnet.synsets(single_word.lower(), pos=wordnet.ADJ)

    if not single_word_synsets:
        # print(f"Warning: '{single_word}' not found as a noun or adjective in WordNet.")
        return None # Cannot find synsets for the main word

    best_topic = None
    highest_similarity = -1.0 # Initialize with a value lower than any possible similarity

    # Use a threshold to avoid returning a topic for very low similarity scores
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
    if highest_similarity > SIGNIFICANCE_THRESHOLD:
        return best_topic
    else:
        return None # No topic found with significant similarity

# --- Flask Application ---
app = Flask(__name__)

@app.route('/')
def index():
    """
    Basic route to serve the HTML form page.
    """
    return render_template('index.html') # Render the index.html template

@app.route('/find-topic', methods=['POST'])
def find_topic():
    """
    POST route to find top 3 meaningful topic words from a list of words.
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


# --- New Route: Categorize a Single Word ---
@app.route('/categorize-word', methods=['POST'])
def categorize_word():
    """
    POST route to find the best fitting topic for a single word
    from a list of provided topics using WordNet similarity.
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