import pygame
from wordleformatter import PygameWordleRenderer, MAP, COL_B, COL_Y, COL_G, COL_EMPTY
from wordle_utils import get_feedback, best_guess, filter_words, load_words


def play_interactive_game(word_list):
    
    pygame.init()
    
    # Create renderer just for display (don't use its game loop)
    renderer = PygameWordleRenderer(
        games_total=1,
        fps=60,
        ms_per_reveal=100,
        ms_between_turns=200,
        ms_between_games=500,
        show_answer=False,
        title="Wordle Interactive - You provide feedback"
    )
    
    candidates = word_list[:]
    turn = 1
    clock = pygame.time.Clock()
    
    renderer.reset_board()
    renderer.game_idx = 1
    
    while turn <= 6:
        guess = best_guess(candidates)
        print(f"\nTurn {turn}: AI guesses '{guess}'")
        print(f"  Remaining candidates: {len(candidates)}")
        
        r = turn - 1
        for c in range(5):
            renderer.grid_letters[r][c] = guess[c]
            renderer.grid_colors[r][c] = COL_EMPTY
        
        feedback = wait_for_feedback(renderer, turn)
        
        if feedback is None:
            break
        
        print(f"  Feedback: {feedback}")
        
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


def wait_for_feedback(renderer, turn):
    r = turn - 1
    feedback_colors = [COL_B for _ in range(5)]
    
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
                col = get_tile_col_at(renderer, event.pos, turn)
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


def get_tile_col_at(renderer, pos, turn):
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


if __name__ == "__main__":
    words = load_words("wordlesol.txt")
    print(f"Loaded {len(words)} words")
    play_interactive_game(words)
