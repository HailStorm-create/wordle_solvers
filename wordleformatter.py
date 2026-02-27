import pygame

COL_BG = (18, 18, 19)
COL_EMPTY = (58, 58, 60)
COL_TEXT = (255, 255, 255)
COL_G = (106, 170, 100)
COL_Y = (201, 180, 88)
COL_B = (58, 58, 60)

MAP = {"G": COL_G, "Y": COL_Y, "B": COL_B}


class PygameWordleRenderer:
    def __init__(
        self,
        games_total,
        fps=1000,                # high cap so turbo isn't limited
        ms_per_reveal=18,
        ms_between_turns=40,
        ms_between_games=140,
        show_answer=False,
        title="Wordle AI Speedrun"
    ):
        pygame.init()

        self.games_total = games_total
        self.game_idx = 0
        self.fps = fps
        self.ms_per_reveal = ms_per_reveal
        self.ms_between_turns = ms_between_turns
        self.ms_between_games = ms_between_games
        self.show_answer = show_answer

        self.turbo = False

        self.rows = 6
        self.cols = 5
        self.tile = 72
        self.gap = 10
        self.pad = 30
        self.header_h = 70

        w = self.pad * 2 + self.cols * self.tile + (self.cols - 1) * self.gap
        h = self.header_h + self.pad * 2 + self.rows * self.tile + (self.rows - 1) * self.gap

        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

        self.font_big = pygame.font.SysFont("arial", 28, bold=True)
        self.font_tile = pygame.font.SysFont("arial", 36, bold=True)

        self.skip_request = 0

        self.reset_board()

    def reset_board(self):
        self.grid_letters = [["" for _ in range(self.cols)] for _ in range(self.rows)]
        self.grid_colors = [[COL_EMPTY for _ in range(self.cols)] for _ in range(self.rows)]
        self.answer = None

    def pump(self, ms=0):
        start = pygame.time.get_ticks()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise SystemExit

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_t:
                        self.turbo = not self.turbo

            self.draw()

            if not self.turbo:
                self.clock.tick(self.fps)

            if ms <= 0:
                return

            if pygame.time.get_ticks() - start >= ms:
                return

    def start_game(self, answer):
        self.game_idx += 1
        self.reset_board()
        self.answer = answer
        self.pump(1)

    def render_turn(self, guess, feedback, turn):
        r = turn - 1

        for c in range(self.cols):
            self.grid_letters[r][c] = guess[c]

        self.pump(1)

        if self.turbo:
            for c in range(self.cols):
                self.grid_colors[r][c] = MAP[feedback[c]]
            self.pump(1)
        else:
            for c in range(self.cols):
                self.grid_colors[r][c] = MAP[feedback[c]]
                self.pump(self.ms_per_reveal)

        self.pump(self.ms_between_turns)

    def end_game(self, win, turns):
        if not self.turbo:
            self.pump(self.ms_between_games)
        else:
            self.pump(1)

    def draw(self):
        self.screen.fill(COL_BG)

        title = f"Game {self.game_idx}/{self.games_total}"
        if self.turbo:
            title += "   TURBO"

        if self.show_answer and self.answer:
            title += f"   {self.answer}"

        text = self.font_big.render(title, True, COL_TEXT)
        self.screen.blit(text, (20, 20))

        top = self.header_h + self.pad
        left = self.pad

        for r in range(self.rows):
            for c in range(self.cols):
                x = left + c * (self.tile + self.gap)
                y = top + r * (self.tile + self.gap)

                pygame.draw.rect(
                    self.screen,
                    self.grid_colors[r][c],
                    (x, y, self.tile, self.tile),
                    border_radius=8,
                )

                letter = self.grid_letters[r][c]
                if letter:
                    t = self.font_tile.render(letter, True, COL_TEXT)
                    rect = t.get_rect(center=(x + self.tile/2, y + self.tile/2))
                    self.screen.blit(t, rect)

        pygame.display.flip()

    def close(self):
        self.pump(200)
        pygame.quit()
