import random
import math
import pickle
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from wordleformatter import PygameWordleRenderer
from wordlestats import LiveStats


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


def filter_words(candidates, guess, feedback):
    return [w for w in candidates if get_feedback(guess, w) == feedback]


class WordleStateEncoder:
    def __init__(self):
        self.feature_size = 314
    
    def encode(self, history, candidates, word_list, turn):
        green_at = [set() for _ in range(5)]
        not_at = [set() for _ in range(5)]
        
        in_word = set()
        not_in_word = set()
        
        for guess, feedback in history:
            for i, (g, f) in enumerate(zip(guess, feedback)):
                if f == "G":
                    green_at[i].add(g)
                    in_word.add(g)
                elif f == "Y":
                    not_at[i].add(g)
                    in_word.add(g)
                else:
                    if g not in in_word:
                        not_in_word.add(g)
                    not_at[i].add(g)
        
        features = []
        
        for pos in range(5):
            for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                features.append(1.0 if c in green_at[pos] else 0.0)
            for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                features.append(1.0 if c in not_at[pos] else 0.0)
        
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            features.append(1.0 if c in in_word else 0.0)
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            features.append(1.0 if c in not_in_word else 0.0)
        
        features.append(turn / 6.0)
        features.append(len(candidates) / len(word_list))
        
        return torch.tensor(features, dtype=torch.float32)


class WordlePolicy(nn.Module):
    def __init__(self, state_size, vocab_size, hidden_size=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size)
        )
    
    def forward(self, state):
        return self.net(state)


class WordleRLTrainer:
    def __init__(self, word_list, lr=0.001, gamma=0.99):
        self.word_list = word_list
        self.word_to_idx = {w: i for i, w in enumerate(word_list)}
        
        self.encoder = WordleStateEncoder()
        self.model = WordlePolicy(
            state_size=self.encoder.feature_size,
            vocab_size=len(word_list)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        
        self.all_indices = set(range(len(word_list)))
    
    def get_action(self, state, candidates, epsilon=0.1):
        valid_indices = [self.word_to_idx[w] for w in candidates]
        
        if random.random() < epsilon:
            idx = random.choice(valid_indices)
        else:
            with torch.no_grad():
                scores = self.model(state)
                mask = torch.full((len(self.word_list),), float('-inf'))
                for i in valid_indices:
                    mask[i] = 0
                masked_scores = scores + mask
                idx = torch.argmax(masked_scores).item()
        
        return self.word_list[idx], idx
    
    def play_episode(self, answer, epsilon=0.1):
        candidates = self.word_list[:]
        history = []
        
        states = []
        actions = []
        rewards = []
        
        for turn in range(1, 7):
            state = self.encoder.encode(history, candidates, self.word_list, turn)
            guess, action_idx = self.get_action(state, candidates, epsilon)
            feedback = get_feedback(guess, answer)
            
            states.append(state)
            actions.append(action_idx)
            
            history.append((guess, feedback))
            
            if guess == answer:
                reward = (7 - turn) ** 2 / 36.0
                rewards.append(reward)
                return states, actions, rewards, True, turn, history
            
            candidates = filter_words(candidates, guess, feedback)
            
            if not candidates:
                rewards.append(-1.0)
                return states, actions, rewards, False, None, history
            
            rewards.append(-0.05)
        
        # Ran out of turns
        rewards[-1] = -1.0
        return states, actions, rewards, False, None, history
    
    def compute_returns(self, rewards):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns
    
    def train_episode(self, answer, epsilon=0.1):
        states, actions, rewards, won, turns, history = self.play_episode(answer, epsilon)
        returns = self.compute_returns(rewards)
        
        self.optimizer.zero_grad()
        
        total_loss = 0
        for state, action, G in zip(states, actions, returns):
            logits = self.model(state)
            log_probs = torch.log_softmax(logits, dim=0)
            loss = -log_probs[action] * G
            total_loss += loss
        
        total_loss.backward()
        self.optimizer.step()
        
        return won, turns, history
    
    def train(self, epochs=30, games_per_epoch=500):
        print("=" * 50)
        print("TRAINING WORDLE RL AGENT")
        print("=" * 50)
        
        all_training_games = []
        
        for epoch in range(epochs):
            wins = 0
            total_turns = 0
            
            epsilon = max(0.05, 0.5 * (0.95 ** epoch))
            
            for _ in range(games_per_epoch):
                answer = random.choice(self.word_list)
                won, turns, history = self.train_episode(answer, epsilon)
                all_training_games.append(history)
                
                if won:
                    wins += 1
                    total_turns += turns
            
            win_rate = wins / games_per_epoch
            avg_turns = total_turns / wins if wins > 0 else 0
            
            print(f"Epoch {epoch + 1:3d} | Win: {win_rate:.1%} | Avg: {avg_turns:.2f} | ε: {epsilon:.3f}")
        
        print("Training complete!")
        return all_training_games
    
    def solve_game(self, answer):
        candidates = self.word_list[:]
        history = []
        
        for turn in range(1, 7):
            state = self.encoder.encode(history, candidates, self.word_list, turn)
            guess, _ = self.get_action(state, candidates, epsilon=0)
            feedback = get_feedback(guess, answer)
            history.append((guess, feedback))
            
            if guess == answer:
                return history
            
            candidates = filter_words(candidates, guess, feedback)
            if not candidates:
                return None
        
        return None
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")


def replay_game(history, renderer):
    for turn, (guess, feedback) in enumerate(history, 1):
        renderer.render_turn(guess, feedback, turn)
    
    won = history[-1][1] == "GGGGG"
    renderer.end_game(won, len(history))


def run_visual_batch(word_list, force_retrain=False):
    model_path = "wordle_rl_model.pth"
    games_path = "wordle_training_games.pkl"
    
    trainer = WordleRLTrainer(word_list)
    
    if os.path.exists(model_path) and not force_retrain:
        print(f"Found existing model at {model_path}")
        trainer.load(model_path)
        
        if os.path.exists(games_path):
            print(f"Loading cached training games from {games_path}")
            with open(games_path, "rb") as f:
                training_games = pickle.load(f)
        else:
            training_games = []
    else:
        training_games = trainer.train(epochs=30, games_per_epoch=500)
        trainer.save(model_path)
        
        with open(games_path, "wb") as f:
            pickle.dump(training_games, f)
        print(f"Training games cached to {games_path}")
    
    if training_games:
        print(f"\nPlayback training games ({len(training_games)} games)...")
        train_renderer = PygameWordleRenderer(
            games_total=len(training_games),
            fps=1000,
            ms_per_reveal=2,
            ms_between_turns=2,
            ms_between_games=5,
            show_answer=False,
            title="Training Playback"
        )
        
        for i, history in enumerate(training_games):
            if history is None:
                train_renderer.start_game("XXXXX")
                train_renderer.end_game(False, None)
            else:
                train_renderer.start_game(history[0][0])
                replay_game(history, train_renderer)
        
        train_renderer.close()
        print("Training playback complete!")
    
    # ========== PHASE 2: Solve all test games ==========
    shuffled_words = word_list[:]
    random.shuffle(shuffled_words)
    
    print("\nSolving all test games with trained model...")
    solutions = {}
    
    for i, answer in enumerate(shuffled_words):
        if (i + 1) % 500 == 0:
            print(f"  Solved {i + 1}/{len(shuffled_words)}")
        solutions[answer] = trainer.solve_game(answer)
    
    print("Done solving! Starting test playback...")
    
    renderer = PygameWordleRenderer(
        games_total=len(shuffled_words),
        fps=1000,
        ms_per_reveal=2,
        ms_between_turns=2,
        ms_between_games=5,
        show_answer=False,
        title="Test Set Results"
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
    
    print(f"\nWin rate: {wins / len(shuffled_words):.1%}")
    if wins:
        print(f"Average guesses: {total_turns / wins:.2f}")
    
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


vers = 0
if __name__ == "__main__":
    words = load_words("wordlesol.txt")
    
    if (vers == 0):
        run_visual_batch(words, force_retrain=True)
    # runs interactive mode with trained AI - you provide feedback
    else:
        import pygame
        from wordleformatter import MAP, COL_B, COL_Y, COL_G, COL_EMPTY
        
        model_path = "wordle_rl_model.pth"
        
        trainer = WordleRLTrainer(words)
        
        if os.path.exists(model_path) and True:
            print(f"Found existing model at {model_path}")
            trainer.load(model_path)
        else:
            print("Training AI with optimized settings...")
            trainer.train(epochs=30, games_per_epoch=500)
            trainer.save(model_path)
        
        pygame.init()
        
        renderer = PygameWordleRenderer(
            games_total=1,
            fps=60,
            ms_per_reveal=100,
            ms_between_turns=200,
            ms_between_games=500,
            show_answer=False,
            title="Wordle Interactive AI - You provide feedback"
        )
        
        candidates = words[:]
        turn = 1
        clock = pygame.time.Clock()
        history = []
        
        renderer.reset_board()
        renderer.game_idx = 1
        
        while turn <= 6:
            state = trainer.encoder.encode(history, candidates, words, turn)
            guess, _ = trainer.get_action(state, candidates, epsilon=0)
            
            print(f"\nTurn {turn}: AI guesses '{guess}' (RL model)")
            print(f"  Remaining candidates: {len(candidates)}")
            
            r = turn - 1
            for c in range(5):
                renderer.grid_letters[r][c] = guess[c]
                renderer.grid_colors[r][c] = COL_EMPTY
            
            feedback_colors = [COL_B for _ in range(5)]
            for c in range(5):
                renderer.grid_colors[r][c] = feedback_colors[c]
            
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
                            pygame.quit()
                            raise SystemExit
                    
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        tile_r = -1
                        for check_r in range(6):
                            top = renderer.header_h + renderer.pad
                            left = renderer.pad
                            y_start = top + check_r * (renderer.tile + renderer.gap)
                            y_end = y_start + renderer.tile
                            
                            if y_start <= event.pos[1] <= y_end:
                                for check_c in range(5):
                                    x_start = left + check_c * (renderer.tile + renderer.gap)
                                    x_end = x_start + renderer.tile
                                    if x_start <= event.pos[0] <= x_end and check_r == r:
                                        if feedback_colors[check_c] == COL_B or feedback_colors[check_c] == COL_EMPTY:
                                            feedback_colors[check_c] = COL_Y
                                        elif feedback_colors[check_c] == COL_Y:
                                            feedback_colors[check_c] = COL_G
                                        else:
                                            feedback_colors[check_c] = COL_B
                                        renderer.grid_colors[r][check_c] = feedback_colors[check_c]
                
                renderer.draw()
                pygame.display.flip()
                clock.tick(60)
            
            feedback = ""
            for color in feedback_colors:
                if color == COL_G:
                    feedback += "G"
                elif color == COL_Y:
                    feedback += "Y"
                else:
                    feedback += "B"
            
            print(f"  Feedback: {feedback}")
            history.append((guess, feedback))
            
            if feedback == "GGGGG":
                for c in range(5):
                    renderer.grid_colors[r][c] = COL_G
                renderer.draw()
                pygame.display.flip()
                clock.tick(2)
                print(f"Won in {turn} guesses!")
                break
            
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
