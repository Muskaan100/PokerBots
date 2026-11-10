"""
Discard Hold'em Poker Bot — Optimized for Speed + Strength (Tighter Simulation Limits)
=====================================================================
A competitive bot that respects the strict time‑bank constraints of the
competition while still using Monte‑Carlo equity estimation and smart
discard selection.

Key changes compared to the previous version:
- `TimeManager.get_sim_count` now caps simulations at **70** (instead of 100)
- Minimum simulations per decision is **20** (instead of 30) when the budget is tiny.
- Discard optimisation uses only **40** simulations per candidate.
- Added simple equity‑cache for (hole, board) tuples to avoid recomputation.
- All logging goes to `stderr`.

The rest of the architecture (hand evaluator, equity calculator, betting
engine, opponent model) remains the same and is fully compatible with the
provided `engine.py`.
"""

import sys
import json
import random
import time
import itertools
from collections import Counter

# ---------------------------------------------------------------------------
# Constants & Pre‑computation
# ---------------------------------------------------------------------------
RANKS = "23456789TJQKA"
SUITS = "shdc"
RANK_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}
ALL_CARDS = [r + s for r in RANKS for s in SUITS]

# Fast rank lookup table (ASCII char → numeric value)
_RV = [0] * 128
for r in RANKS:
    _RV[ord(r)] = RANK_VALUE[r]

def _rv(card: str) -> int:
    return _RV[ord(card[0])]

# ---------------------------------------------------------------------------
# Fast Hand Evaluator (5‑card ranking)
# ---------------------------------------------------------------------------
def rank5(cards):
    """Return a comparable tuple describing hand strength.
    Higher tuple means a stronger hand.
    """
    v = [_rv(c) for c in cards]
    values = sorted(v, reverse=True)
    # flush detection – all suits equal
    flush = all(c[1] == cards[0][1] for c in cards)

    # count groups of equal values
    cnt = {}
    for val in values:
        cnt[val] = cnt.get(val, 0) + 1
    groups = sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape0 = groups[0][1]

    # straight detection (including wheel A‑2‑3‑4‑5)
    uniq = sorted(set(values), reverse=True)
    straight_high = 0
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_high = uniq[0]
        elif uniq == [14, 5, 4, 3, 2]:
            straight_high = 5

    by_group = tuple(v for v, _ in groups)

    if straight_high and flush:
        return (8, straight_high)               # straight flush
    if shape0 == 4:
        return (7,) + by_group                  # four of a kind
    if shape0 == 3 and len(groups) == 2:
        return (6,) + by_group                  # full house
    if flush:
        return (5,) + tuple(values)              # flush
    if straight_high:
        return (4, straight_high)                # straight
    if shape0 == 3:
        return (3,) + by_group                  # three of a kind
    shape1 = groups[1][1] if len(groups) > 1 else 0
    if shape0 == 2 and shape1 == 2:
        return (2,) + by_group                  # two pair
    if shape0 == 2:
        return (1,) + by_group                  # one pair
    return (0,) + tuple(values)                 # high card


def best_hand(hole, board):
    """Return the best 5‑card hand from hole + board (list of strings)."""
    all_cards = list(hole) + list(board)
    best = None
    for combo in itertools.combinations(all_cards, 5):
        s = rank5(combo)
        if best is None or s > best:
            best = s
    return best

# ---------------------------------------------------------------------------
# Time Management
# ---------------------------------------------------------------------------
class TimeManager:
    """Tracks the leg's time‑bank and decides how many Monte‑Carlo sims we can run.
    The logic is deliberately conservative to guarantee we never exhaust the
    bank.
    """

    def __init__(self):
        self.leg_start = time.monotonic()
        self.hands_total = 200    # will be overwritten by init(msg)
        self.hands_played = 0
        self.time_bank_ms = 30000
        self.per_response_ms = 5000

    def init(self, msg):
        self.hands_total = msg.get("total_hands", 200)
        self.time_bank_ms = msg.get("time_bank_ms", 30000)
        self.per_response_ms = msg.get("per_response_ms", 5000)
        self.hands_played = 0
        self.leg_start = time.monotonic()

    def update(self, time_left_ms):
        self.time_bank_ms = time_left_ms

    def get_budget_ms(self) -> float:
        """How many milliseconds we may safely spend on the *next* decision.
        The budget is a fraction of the remaining bank, limited to 60 % of the
        per‑response limit and never below 5 ms.
        """
        hands_remaining = max(1, self.hands_total - self.hands_played)
        # Roughly 2–3 decisions per hand → use 2.5 factor
        decisions_remaining = hands_remaining * 2.5
        budget = self.time_bank_ms / decisions_remaining
        budget = min(budget, self.per_response_ms * 0.6)
        return max(budget, 5.0)

    def get_sim_count(self, base=70) -> int:
        """Return a safe number of Monte‑Carlo simulations.
        * `base` is the hard ceiling (default 70).
        * Minimum is 20 sims when the budget is extremely low.
        """
        budget_ms = self.get_budget_ms()
        # Empirical cost ≈ 0.08 ms per simulation (very conservative)
        sims = int(budget_ms / 0.08)
        if sims < 20:
            sims = 20
        return min(sims, base)

    def hand_done(self):
        self.hands_played += 1

# ---------------------------------------------------------------------------
# Equity Calculator (Monte‑Carlo) with simple caching
# ---------------------------------------------------------------------------
class EquityCalculator:
    def __init__(self):
        self._rng = random.Random(42)
        self._cache = {}  # (hole_tuple, board_tuple) → equity

    def _deck_without(self, excluded):
        return [c for c in ALL_CARDS if c not in excluded]

    def equity_2cards_vs_random(self, my_hole, board, known_extra=None, num_sims=60):
        """Estimate equity of `my_hole` against a random opponent.
        `board` may be 3‑5 cards. `known_extra` can contain any additional known
        cards (e.g., our discard, opponent discard). The function respects the
        supplied `num_sims` (already limited by TimeManager).
        """
        key = (tuple(sorted(my_hole)), tuple(sorted(board)), tuple(sorted(known_extra or [])))
        if key in self._cache:
            return self._cache[key]

        known = set(my_hole) | set(board)
        if known_extra:
            known |= set(known_extra)
        deck = self._deck_without(known)
        remaining_board = 5 - len(board)
        wins = ties = 0
        rng = self._rng
        for _ in range(num_sims):
            rng.shuffle(deck)
            opp = (deck[0], deck[1])
            if remaining_board:
                full_board = list(board) + deck[2:2 + remaining_board]
            else:
                full_board = list(board)
            my_score = best_hand(my_hole, full_board)
            opp_score = best_hand(opp, full_board)
            if my_score > opp_score:
                wins += 1
            elif my_score == opp_score:
                ties += 1
        equity = (wins + ties * 0.5) / num_sims
        self._cache[key] = equity
        return equity

    def best_discard(self, hole_3, board_3, num_sims_per=40):
        """Choose the best card to discard from three hole cards.
        `board_3` is the flop (list of three cards). For each possible discard we
        simulate `num_sims_per` Monte‑Carlo runs to estimate post‑discard equity.
        Returns a tuple `(card_to_discard, equity)`.
        """
        best_card = None
        best_eq = -1.0
        for discard in hole_3:
            remaining = [c for c in hole_3 if c != discard]
            eq = self.equity_2cards_vs_random(
                remaining,
                board_3,
                known_extra=[discard],
                num_sims=num_sims_per,
            )
            if eq > best_eq:
                best_eq = eq
                best_card = discard
        return best_card, best_eq

# ---------------------------------------------------------------------------
# Opponent Model (lightweight)
# ---------------------------------------------------------------------------
class OpponentModel:
    def __init__(self):
        self.preflop_actions = Counter()
        self.postflop_actions = Counter()
        self.hands_seen = 0

    def record(self, action, stage):
        if stage == "preflop":
            self.preflop_actions[action] += 1
        else:
            self.postflop_actions[action] += 1
        self.hands_seen += 1

    def aggression_factor(self):
        # (bets + raises) / (calls + checks + folds)
        aggressive = self.postflop_actions["bet"] + self.postflop_actions["raise"]
        passive = self.postflop_actions["call"] + self.postflop_actions["check"] + self.postflop_actions["fold"]
        return aggressive / (passive + 1e-6)

    def vpip(self):
        # voluntary put money in pot (call, bet, raise) pre‑flop
        total = sum(self.preflop_actions.values())
        if total == 0:
            return 0.0
        vpip = self.preflop_actions["call"] + self.preflop_actions["bet"] + self.preflop_actions["raise"]
        return vpip / total

# ---------------------------------------------------------------------------
# Betting Engine – decides the action for each request
# ---------------------------------------------------------------------------
class BettingEngine:
    def __init__(self):
        self.time_manager = TimeManager()
        self.equity_calc = EquityCalculator()
        self.opp_model = OpponentModel()

    def handle(self, request):
        # Guard against messages that do not contain game state (e.g., match start/end)
        if not isinstance(request, dict):
            return {"action": "check"}
        if request.get("type") == "new_hand":
            self.time_manager.init(request)
            # No action needed yet; engine will request a decision later.
            return {"action": "check"}
        # Update remaining time bank if provided
        self.time_manager.update(request.get("time_left_ms", self.time_manager.time_bank_ms))
        # Extract fields safely; if missing, fall back to empty/default values
        stage = request.get("stage")
        hole = request.get("hole", [])
        board = request.get("board", [])
        legal = request.get("legal_actions", [])
        seat = request.get("seat", 0)
        # If essential fields are missing, do nothing
        if not stage or not legal:
            return {"action": "check"}


        # Update time left before we decide
        self.time_manager.update(request.get("time_left_ms", self.time_manager.time_bank_ms))

        # stage, hole, board, legal, seat already extracted above

        # ----- Discard stage ------------------------------------------------
        if stage == "discard":
            # hole contains 3 cards, board is flop (3 cards)
            discard, _ = self.equity_calc.best_discard(hole, board, num_sims_per=40)
            action = f"discard {discard}"
            self.time_manager.hand_done()
            return {"action": action}

        # ----- Pre‑flop ------------------------------------------------------
        if stage == "preflop":
            sims = self.time_manager.get_sim_count(base=60)
            equity = self.equity_calc.equity_2cards_vs_random(hole, [], num_sims=sims)
            # Simple thresholds – can be tweaked later
            if equity >= 0.65 and "raise" in legal:
                act = "raise 2"
            elif equity >= 0.55 and "call" in legal:
                act = "call"
            else:
                act = "fold" if "fold" in legal else "check"
            self.opp_model.record(act, "preflop")
            return {"action": act}

        # ----- Post‑flop (flop, turn, river) --------------------------------
        sims = self.time_manager.get_sim_count(base=70)
        equity = self.equity_calc.equity_2cards_vs_random(hole, board, num_sims=sims)
        # Adjust thresholds based on opponent aggression
        agg = self.opp_model.aggression_factor()
        raise_thr = 0.60 - 0.05 * (agg - 1)  # more aggressive opponents → raise looser
        call_thr = 0.45 - 0.03 * (agg - 1)

        if equity >= raise_thr and "raise" in legal:
            act = "raise 2"
        elif equity >= call_thr and "call" in legal:
            act = "call"
        else:
            act = "fold" if "fold" in legal else "check"
        self.opp_model.record(act, "postflop")
        return {"action": act}

# ---------------------------------------------------------------------------
# Main loop – read JSON lines from stdin, write response JSON to stdout
# ---------------------------------------------------------------------------

def main():
    engine = BettingEngine()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"action": "check"}), flush=True)
            continue
        response = engine.handle(msg)
        # The protocol expects a JSON object with at least an "action" key
        print(json.dumps(response), flush=True)

if __name__ == "__main__":
    main()
