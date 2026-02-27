import pygame
import sys
from wordleformatter import PygameWordleRenderer, COL_TEXT, COL_BG

def get_feedback(guess, answer):
    feedback = ['B'] * 5  
    answer_chars = list(answer)
    
    for i in range(5):
        if guess[i] == answer[i]:
            feedback[i] = 'G'
            answer_chars[i] = None
    
    for i in range(5):
        if feedback[i] == 'B' and guess[i] in answer_chars:
            feedback[i] = 'Y'
            answer_chars[answer_chars.index(guess[i])] = None
    
    return ''.join(feedback)

def main():
    print("=" * 50)
    print("WORDLE GAME")
    print("=" * 50)
    answer = input("Enter the answer word (5 letters): ").strip().upper()
    
    if len(answer) != 5:
        print("Answer must be 5 letters!")
        sys.exit(1)
    
    renderer = PygameWordleRenderer(
        games_total=1,
        show_answer=False,
        title="Wordle - Type to Guess"
    )
    renderer.start_game(answer)
    
    turn = 1
    game_won = False
    
    input_text = ""
    font_small = pygame.font.SysFont("arial", 20)
    
    while turn <= 6:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    renderer.turbo = not renderer.turbo
                elif event.key == pygame.K_RETURN:
                    guess = input_text.strip().upper()
                    if len(guess) == 5:
                        feedback = get_feedback(guess, answer)
                        renderer.render_turn(guess, feedback, turn)
                        
                        if guess == answer:
                            game_won = True
                            break
                        
                        turn += 1
                        input_text = ""
                    else:
                        print("Guess must be 5 letters!")
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                elif event.unicode.isalpha() and len(input_text) < 5:
                    input_text += event.unicode.upper()
        
        if game_won:
            break
        
        renderer.draw()
        prompt = f"Guess {turn}/6: {input_text}{'_' * (5 - len(input_text))}"
        text = font_small.render(prompt, True, COL_TEXT)
        renderer.screen.blit(text, (renderer.pad, renderer.screen.get_height() - 30))
        pygame.display.flip()
        renderer.clock.tick(60)
    
    renderer.draw()
    if game_won:
        result = f" WON in {turn} turns!"
    else:
        result = f" Lost! Answer was: {answer}"
    
    text = font_small.render(result, True, COL_TEXT)
    renderer.screen.blit(text, (renderer.pad, renderer.screen.get_height() - 30))
    pygame.display.flip()
    
    renderer.close()

if __name__ == "__main__":
    main()
