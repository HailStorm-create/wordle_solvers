import random
import math
from collections import defaultdict

from wordleformatter import PygameWordleRenderer
from wordlestats import LiveStats
import pygame
from wordleformatter import MAP, COL_B, COL_Y, COL_G, COL_EMPTY


def load_words(path):
    with open(path, "r") as f:
        return [w.strip().upper() for w in f if len(w.strip()) == 5]


def load_words_flat(path):
    words = []
    with open(path, "r") as f:
        for line in f:
            for w in line.strip().split():
                if len(w) == 5 and w.isalpha():
                    words.append(w.upper())
    return words


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


def filter_words(candidates, guess, feedback):
    return [w for w in candidates if get_feedback(guess, w) == feedback]


def entropy_score(guess, candidates):
    buckets = defaultdict(int)

    for answer in candidates:
        fb = get_feedback(guess, answer)
        buckets[fb] += 1

    total = len(candidates)
    entropy = 0.0

    for count in buckets.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


def compute_best_opener(all_guesses, answers):
    print(f"Computing best opening word ({len(all_guesses)} guesses vs {len(answers)} answers)...")
    best_word = None
    best_score = -1

    for i, word in enumerate(all_guesses):
        if (i + 1) % 1000 == 0:
            print(f"  Checked {i + 1}/{len(all_guesses)} openers...")
        score = entropy_score(word, answers)
        if score > best_score:
            best_score = score
            best_word = word

    print(f"Best opener: {best_word} (entropy: {best_score:.3f})")
    return best_word


def best_guess(candidates):
    best_word = None
    best_score = -1
    
    for word in candidates:
        score = entropy_score(word, candidates)
        if score > best_score:
            best_score = score
            best_word = word

    return best_word


def solve_game(answer, all_guesses, answers, opener):
    candidates = answers[:]
    history = []

    for turn in range(1, 7):
        guess = opener if turn == 1 else best_guess(candidates)
        feedback = get_feedback(guess, answer)
        history.append((guess, feedback))

        if guess == answer:
            return history

        candidates = filter_words(candidates, guess, feedback)

        if not candidates:
            return None

    return None


def replay_game(history, renderer):
    for turn, (guess, feedback) in enumerate(history, 1):
        renderer.render_turn(guess, feedback, turn)
    
    won = history[-1][1] == "GGGGG"
    renderer.end_game(won, len(history))


def run_visual_batch(all_guesses, answers):

    shuffled_words = answers[:]
    random.shuffle(shuffled_words)

    opener = compute_best_opener(all_guesses, answers)

    print("Solving all games...")
    solutions = {}
    
    for i, answer in enumerate(shuffled_words):
        if (i + 1) % 100 == 0:
            print(f"  Solved {i + 1}/{len(shuffled_words)}")
        solutions[answer] = solve_game(answer, all_guesses, answers, opener)
    
    print("Done solving! Starting playback...", flush=True)

    renderer = PygameWordleRenderer(
        games_total=len(shuffled_words),
        fps=1000,
        ms_per_reveal=2,
        ms_between_turns=2,
        ms_between_games=5,
        show_answer=False
    )

    stats = LiveStats()

    wins = 0
    total_turns = 0
    failed_words = []

    for answer in shuffled_words:
        renderer.start_game(answer)
        history = solutions[answer]

        if history is not None:
            replay_game(history, renderer)
            wins += 1
            total_turns += len(history)
            stats.record(len(history), live_update=not renderer.turbo)
        else:
            renderer.end_game(False, None)
            stats.record(None, live_update=not renderer.turbo)
            failed_words.append(answer)

    renderer.close()
    stats.show_final()

    print("Win rate:", wins / len(shuffled_words))
    if wins:
        print("Average guesses:", total_turns / wins)
    
    if failed_words:
        print(f"\nFailed words ({len(failed_words)}):", flush=True)
        for word in failed_words:
            print(f"  {word}", flush=True)
        
        print(f"\nReplaying {len(failed_words)} failed words...\n", flush=True)
        playback_renderer = PygameWordleRenderer(
            games_total=len(failed_words),
            fps=1000,
            ms_per_reveal=100,
            ms_between_turns=500,
            ms_between_games=1000,
            show_answer=True
        )
        
        for answer in failed_words:
            playback_renderer.start_game(answer)
            history = solutions[answer]
            if history is not None:
                replay_game(history, playback_renderer)
            else:
                playback_renderer.end_game(False, None)
        
        playback_renderer.close()
    else:
        print("No failed words!", flush=True)


def play_interactive_entropy(all_guesses, answers):
    
    opener = "TARSE"
    
    pygame.init()
    
    renderer = PygameWordleRenderer(
        games_total=1,
        fps=60,
        ms_per_reveal=100,
        ms_between_turns=200,
        ms_between_games=500,
        show_answer=False,
        title="Wordle Interactive (Entropy) - You provide feedback"
    )
    
    candidates = answers[:]
    turn = 1
    clock = pygame.time.Clock()
    
    # Initialize board
    renderer.reset_board()
    renderer.game_idx = 1
    
    while turn <= 6:
        if turn == 1:
            guess = opener
        else:
            guess = best_guess(candidates)
        
        print(f"\nTurn {turn}: AI guesses '{guess}' (entropy-based)")
        print(f"  Remaining candidates: {len(candidates)}")
        
        # Display the guess and wait for feedback
        r = turn - 1
        for c in range(5):
            renderer.grid_letters[r][c] = guess[c]
            renderer.grid_colors[r][c] = COL_EMPTY
        
        feedback = wait_for_feedback_interactive(renderer, turn)
        
        if feedback is None:  # User quit
            break
        
        print(f"  Feedback: {feedback}")
        
        # Check if won
        if feedback == "GGGGG":
            for c in range(5):
                renderer.grid_colors[r][c] = COL_G
            renderer.draw()
            pygame.display.flip()
            clock.tick(2)
            print(f"Won in {turn} guesses!")
            break
        
        # Filter candidates
        new_candidates = filter_words(candidates, guess, feedback)
        
        if not new_candidates:
            print(f"  ERROR: No valid candidates remain!")
            break
        
        candidates = new_candidates
        
        # Show the feedback colors
        for c in range(5):
            renderer.grid_colors[r][c] = MAP[feedback[c]]
        renderer.draw()
        pygame.display.flip()
        clock.tick(2)
        
        turn += 1
    else:
        print("Out of guesses!")
        clock.tick(1)
    
    pygame.quit()


def wait_for_feedback_interactive(renderer, turn):
    r = turn - 1
    feedback_colors = [COL_B for _ in range(5)]
    
    # Copy to display
    for c in range(5):
        renderer.grid_colors[r][c] = feedback_colors[c]
    
    clock = pygame.time.Clock()
    confirmed = False
    
    print(f"  Click tiles to set: Grey/Yellow/Green, then press ENTER")
    
    while not confirmed:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    confirmed = True
                    break
                
                if event.key == pygame.K_ESCAPE:
                    return None
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                col = get_tile_col_at_interactive(renderer, event.pos, turn)
                if col is not None:
                    if feedback_colors[col] == COL_B or feedback_colors[col] == COL_EMPTY:
                        feedback_colors[col] = COL_Y
                    elif feedback_colors[col] == COL_Y:
                        feedback_colors[col] = COL_G
                    else:
                        feedback_colors[col] = COL_B
                    
                    renderer.grid_colors[r][col] = feedback_colors[col]
        
        renderer.draw()
        pygame.display.flip()
        clock.tick(60)
    
    # Convert colors to feedback string
    feedback = ""
    for color in feedback_colors:
        if color == COL_G:
            feedback += "G"
        elif color == COL_Y:
            feedback += "Y"
        else:
            feedback += "B"
    
    return feedback


def get_tile_col_at_interactive(renderer, pos, turn):
    r = turn - 1
    top = renderer.header_h + renderer.pad
    left = renderer.pad
    mx, my = pos
    
    y_start = top + r * (renderer.tile + renderer.gap)
    y_end = y_start + renderer.tile
    
    if not (y_start <= my <= y_end):
        return None
    
    for c in range(5):
        x_start = left + c * (renderer.tile + renderer.gap)
        x_end = x_start + renderer.tile
        if x_start <= mx <= x_end:
            return c
    
    return None


# -----------------------------
# Main
# -----------------------------
vers = 0
if __name__ == "__main__":
    # All valid guesses (~15k words) - one per line
    all_guesses = load_words("allwordsguess.txt")
    # Possible answers (~2.3k words)
    answers = load_words("wordlesol.txt")
    
    print(f"Loaded {len(all_guesses)} valid guesses, {len(answers)} possible answers")
    
    if (vers == 0):
        run_visual_batch(all_guesses, answers)
    else:
        play_interactive_entropy(all_guesses, answers)
