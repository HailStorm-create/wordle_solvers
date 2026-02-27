def load_words(path):
    with open(path, "r") as f:
        return [w.strip().upper() for w in f if len(w.strip()) == 5]


def get_feedback(guess, answer):
    result = ["B"] * 5
    answer_chars = list(answer)

    for i in range(5):
        if guess[i] == answer[i]:
            result[i] = "G"
            answer_chars[i] = None

    for i in range(5):
        if result[i] == "B" and guess[i] in answer_chars:
            result[i] = "Y"
            answer_chars[answer_chars.index(guess[i])] = None

    return "".join(result)


def consistent(word, guess, feedback):
    return get_feedback(guess, word) == feedback


def filter_words(candidates, guess, feedback):
    return [w for w in candidates if consistent(w, guess, feedback)]


def score_words(words):
    freq = {}

    for w in words:
        for c in set(w):
            freq[c] = freq.get(c, 0) + 1

    scores = {}
    for w in words:
        scores[w] = sum(freq[c] for c in set(w))

    return scores


def best_guess(words):
    scores = score_words(words)
    return max(words, key=lambda w: scores[w])
