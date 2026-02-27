import random
import pygame
from wordleformatter import PygameWordleRenderer
from wordlestats import LiveStats
from wordle_imit import play_interactive_game
from wordle_utils import get_feedback, best_guess, filter_words, load_words


def play_game(answer, word_list, renderer=None):
    candidates = word_list[:]

    for turn in range(1, 7):
        guess = best_guess(candidates)
        feedback = get_feedback(guess, answer)

        if renderer:
            renderer.render_turn(guess, feedback, turn)

        if guess == answer:
            if renderer:
                renderer.end_game(True, turn)
            return turn

        candidates = filter_words(candidates, guess, feedback)

        if not candidates:
            if renderer:
                renderer.end_game(False, None)
            return None

    if renderer:
        renderer.end_game(False, None)
    return None


def run_visual_batch(word_list):

    shuffled_words = word_list[:]
    random.shuffle(shuffled_words)

    renderer = PygameWordleRenderer(
        games_total=len(shuffled_words),
        fps=1000,
        ms_per_reveal=12,
        ms_between_turns=12,
        ms_between_games=30,
        show_answer=False
    )

    stats = LiveStats()

    wins = 0
    total_turns = 0
    skip_remaining = 0
    failed_words = []

    for answer in shuffled_words:

        if renderer.skip_request > 0:
            skip_remaining += renderer.skip_request
            renderer.skip_request = 0

        active_renderer = None if skip_remaining > 0 else renderer

        if active_renderer:
            active_renderer.start_game(answer)
        else:
            renderer.game_idx += 1

        turns = play_game(answer, word_list, renderer=active_renderer)

        if turns is not None:
            wins += 1
            total_turns += turns
        else:
            failed_words.append(answer)
        stats.record(turns, live_update=not renderer.turbo)

    renderer.close()
    stats.show_final()

    print("Win rate:", wins / len(shuffled_words))
    print("Average guesses:", total_turns / wins)
    print(f"DEBUG: failed_words = {failed_words}", flush=True)
    
    if failed_words:
        print(f"\nFailed words ({len(failed_words)}):", flush=True)
        for word in failed_words:
            print(f"  {word}", flush=True)
        
        # Playback failed words
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
            play_game(answer, word_list, renderer=playback_renderer)
        
        playback_renderer.close()
    else:
        print("No failed words!", flush=True)


vers = 0
if __name__ == "__main__":

    if (vers == 0): 
        words = load_words("wordlesol.txt")
        run_visual_batch(words)
    else:
        words = load_words("wordlesol.txt")
        play_interactive_game(words)
