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

# --- Topic Finding Function (with WSD and revised ranking) ---
def find_meaningful_topic_word_improved(topic_words):
    """
    Finds a single meaningful descriptive word for a list of topic words
    by performing Word Sense Disambiguation (WSD) using Wu-Palmer similarity
    and then identifying common, relatively specific hypernyms using WordNet.

    Args:
        topic_words (list): A list of strings, where each string is a word
                            from the topic.

    Returns:
        str or None: The most representative common hypernym found based on WSD
                     and ranking, or the original word if only one is provided,
                     or None if no meaningful common hypernyms are identified.
    """
    if not topic_words:
        return None
    if len(topic_words) == 1:
        # If only one word, return the word itself.
        return topic_words[0]

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
        return None

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
        return None

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
        # Fallback: if no meaningful common ancestors found for selected synsets, return None
        return None

    # --- Ranking ---
    # Rank candidates:
    # 1. Prioritize by the number of shared *selected synsets* (higher is better)
    # 2. If counts are tied, prioritize by the hypernym's depth (higher depth is more specific, better)
    # 3. If still tied, prioritize by the minimum path length (shorter path from a selected synset is better)

    sorted_candidates = sorted(
        meaningful_candidates.values(),
        key=lambda x: (-x['shared_count'], -x['depth'], x['min_path_length']) # Use negative for descending sort
    )

    # The best candidate is the first one after sorting
    best_synset_data = sorted_candidates[0]
    best_synset = best_synset_data['synset']

    # Return the name (lemma) of the best synset, replacing underscores with spaces
    return best_synset.lemmas()[0].name().replace('_', ' ')

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
    POST route to find a meaningful topic word from a list of words.
    Expects a JSON payload like: {"words": ["apple", "banana", "orange"]}
    Returns a JSON response like: {"topic_word": "fruit"} or {"topic_word": null}
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

    # Use the improved function
    suggested_word = find_meaningful_topic_word_improved(topic_words)

    return jsonify({"topic_word": suggested_word})

if __name__ == '__main__':
    # Run the Flask app
    # debug=True allows for automatic reloading on code changes and shows detailed errors
    # In a production environment, set debug=False and use a production server like Gunicorn
    # This block is primarily for running the script directly for testing/debugging outside Docker.
    # The host='0.0.0.0' makes the server accessible externally, useful for Docker or other environments.
    app.run(debug=True, host='0.0.0.0')