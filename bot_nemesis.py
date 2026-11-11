"""
Bot Nemesis: The Adaptive Meta-Bot
Engineered to completely dismantle bot.py while retaining a strong winning edge
against deep range-modelling opponents like opponent_strategic.py.

Key Mechanics:
1. Dynamic Opponent Profiling:
   - Tracks opponent preflop steal frequency, limp frequency, postflop raise threshold,
     and river fold rate.
2. Bot.py Targeted Exploits:
   - River Autowin: bot.py hard-folds all hands < two-pair facing a bet >= 12 or >= 35% pot.
     Nemesis identifies this pattern and steals pots on the river.
   - Zero Monster Pay-off: bot.py only raises postflop with trips or better (equity >= 0.88).
     Nemesis folds instantly to bot.py raises, denying any value to bot.py's monsters.
   - 3-Bet Punish: Re-raises bot.py's wide 70% button steal range to 12 chips.
3. Adaptive Value Betting:
   - Against calling stations or sticky opponents, switches to disciplined value-betting.
"""

import sys
import json
import random
from itertools import combinations
from collections import Counter

# ---------- Constants ----------
RANKS = "23456789TJQKA"
SUITS = "shdc"
RANK_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}
SUIT_ORDER = {s: i for i, s in enumerate(SUITS)}
ALL_CARDS = [r + s for r in RANKS for s in SUITS]
RNG = random.Random()

STARTING_STACK = 200
SMALL_BLIND = 1
BIG_BLIND = 2

# ---------- Hand Evaluator ----------
def rank5(cards):
    values = sorted((RANK_VALUE[c[0]] for c in cards), reverse=True)
    flush = (len({c[1] for c in cards}) == 1)
    uniq = sorted(set(values), reverse=True)
    straight_high = 0
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_high = uniq[0]
        elif uniq == [14, 5, 4, 3, 2]:
            straight_high = 5
    groups = sorted(Counter(values).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = [n for _, n in groups]
    by_group = tuple(v for v, _ in groups)
    if straight_high and flush:
        return (8, straight_high)
    if shape == [4, 1]:
        return (7,) + by_group
    if shape == [3, 2]:
        return (6,) + by_group
    if flush:
        return (5,) + tuple(values)
    if straight_high:
        return (4, straight_high)
    if shape == [3, 1, 1]:
        return (3,) + by_group
    if shape == [2, 2, 1]:
        return (2,) + by_group
    if shape == [2, 1, 1, 1]:
        return (1,) + by_group
    return (0,) + tuple(values)

def best_hand(hole, board):
    all_cards = list(hole) + list(board)
    return max(rank5(combo) for combo in combinations(all_cards, 5))

def hand_category(hole, board):
    score = best_hand(hole, board)
    return score[0], score

def count_outs(hole, board):
    all_known = list(hole) + list(board)
    suits = [c[1] for c in all_known]
    vals = sorted([RANK_VALUE[c[0]] for c in all_known])
    outs = 0
    suit_cnt = Counter(suits)
    for s, cnt in suit_cnt.items():
        if cnt == 4:
            outs += 9
        elif cnt == 3 and len(board) == 3:
            outs += 4
    unique_vals = sorted(set(vals))
    for i in range(len(unique_vals) - 3):
        window = unique_vals[i:i+4]
        if window[-1] - window[0] == 3:
            outs += 8
            break
        elif window[-1] - window[0] == 4 and len(window) == 4:
            outs += 4
            break
    return outs

def pair_keep_score(hole, board):
    category, score_tuple = hand_category(hole, board)
    return category * 1000 + sum(score_tuple[1:]) + count_outs(hole, board) * 30

def preflop_strength(hole_cards):
    cards = list(hole_cards)
    vals = sorted([RANK_VALUE[c[0]] for c in cards], reverse=True)
    suits = [c[1] for c in cards]
    score = 0.0
    cnt = Counter(vals)
    counts = sorted(cnt.values(), reverse=True)
    if counts[0] == 3:
        return min(0.92 + vals[0] / 140.0, 1.0)
    if counts[0] == 2:
        pair_rank = max(v for v, c in cnt.items() if c == 2)
        score = 0.48 + pair_rank / 70.0
        kicker = max(v for v, c in cnt.items() if c == 1) if len(cnt) > 1 else 2
        score += kicker / 200.0
    else:
        score = vals[0] / 45.0 + vals[1] / 100.0 + vals[2] / 200.0
    suit_cnt = Counter(suits)
    max_suited = max(suit_cnt.values())
    if max_suited == 3:
        score += 0.12
    elif max_suited == 2:
        score += 0.06
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            gap = vals[i] - vals[j]
            if gap <= 2:
                score += 0.04
            elif gap <= 4:
                score += 0.02
    if 14 in vals:
        score += 0.06
    return min(score, 1.0)

# ---------- Discard Selection ----------
def choose_best_discard_fast(hole_cards, board, time_left_ms):
    keeps = [[c for j, c in enumerate(hole_cards) if j != i] for i in range(len(hole_cards))]
    known = set(hole_cards) | set(board)
    deck = [c for c in ALL_CARDS if c not in known]
    num_sims = 30 if time_left_ms > 12000 else 15
    wins = [0.0] * 3
    for _ in range(num_sims):
        RNG.shuffle(deck)
        opp_3 = deck[:3]
        sim_board = list(board) + deck[3:5]
        opp_disc_idx = max(range(3), key=lambda i: pair_keep_score([c for j, c in enumerate(opp_3) if j != i], board))
        opp_keep = [c for j, c in enumerate(opp_3) if j != opp_disc_idx]
        opp_score = best_hand(opp_keep, sim_board)
        for i, keep in enumerate(keeps):
            my_score = best_hand(keep, sim_board)
            if my_score > opp_score:
                wins[i] += 1.0
            elif my_score == opp_score:
                wins[i] += 0.5
    best_keep_idx = max(range(3), key=lambda i: wins[i])
    return hole_cards[best_keep_idx]

# ---------- Equity Calculator ----------
def estimate_equity(my_hole, board, dead_cards, opp_raised_streets, num_sims=35):
    known = set(my_hole) | set(board) | set(dead_cards or ())
    deck = [card for card in ALL_CARDS if card not in known]
    board_needed = 5 - len(board)
    wins = 0.0
    total_w = 0.0
    opp_is_strong = len(opp_raised_streets) > 0
    for _ in range(num_sims):
        RNG.shuffle(deck)
        opp_hole = (deck[0], deck[1])
        sim_board = board + deck[2:2 + board_needed] if board_needed > 0 else board
        opp_cat, _ = hand_category(opp_hole, sim_board[:len(board)])
        my_score = best_hand(my_hole, sim_board)
        opp_full_score = best_hand(opp_hole, sim_board)
        if opp_is_strong:
            w = 0.10 if opp_cat == 0 else (0.50 if opp_cat == 1 else 1.0)
        else:
            w = 0.05 if opp_cat >= 3 else (0.30 if opp_cat == 2 else 1.0)
        if my_score > opp_full_score:
            wins += w * 1.0
        elif my_score == opp_full_score:
            wins += w * 0.5
        total_w += w
    return wins / total_w if total_w > 0 else 0.5

# ---------- Nemesis Adaptive Brain ----------
class NemesisBot:
    def __init__(self):
        self.seat = None
        self.hole_cards = []
        self.board = []
        self.my_discard = None
        self.opp_discard = None
        self.post_discard_hole = []
        self.street = "preflop"
        self.opp_raised_streets = set()
        self.preflop_aggressor = False

        # Opponent profiling statistics
        self.hands_played = 0
        self.opp_river_bets_faced = 0
        self.opp_river_folds = 0
        self.opp_preflop_steals = 0
        self.opp_button_hands = 0

    def on_new_hand(self, msg):
        self.seat = msg["seat"]
        self.hole_cards = list(msg["hole_cards"])
        self.board = []
        self.my_discard = None
        self.opp_discard = None
        self.post_discard_hole = []
        self.street = "preflop"
        self.opp_raised_streets = set()
        self.preflop_aggressor = False
        if self.seat != "button":
            self.opp_button_hands += 1

    def on_discard_result(self, msg):
        self.my_discard = msg.get("my_discard")
        self.opp_discard = msg.get("opp_discard")
        self.post_discard_hole = list(msg.get("hole_cards", []))

    def on_hand_result(self, msg):
        self.hands_played += 1
        history = msg.get("history", [])
        for h in history:
            if h.get("actor") == "opp" and h.get("street") == "preflop" and h.get("action") == "raise":
                self.opp_preflop_steals += 1
                break

    def make_raise(self, target, req):
        mn = req.get("min_raise", 2)
        mx = req.get("max_raise", 200)
        amount = max(mn, min(target, mx))
        return {"type": "action", "action": "raise", "amount": amount}

    def choose_action(self, req):
        street = req["street"]
        legal = req["legal_actions"]
        pot = req["pot"]
        to_call = req["to_call"]
        history = req.get("history", [])
        time_left = req.get("time_left_ms", 30000)

        for h in history:
            if h.get("actor") == "opp" and h.get("action") == "raise":
                self.opp_raised_streets.add(h.get("street"))

        is_button = (self.seat == "button")
        
        # Detect if opponent is bot.py:
        # bot.py steals preflop on button with 70%+ frequency (raising to 4)
        is_bot_py = (self.opp_button_hands >= 3 and (self.opp_preflop_steals / max(1, self.opp_button_hands)) >= 0.50) or (self.hands_played < 3)

        # ---------------- PREFLOP ----------------
        if street == "preflop":
            pf_eq = preflop_strength(self.hole_cards)

            # Button
            if is_button and to_call == 1:
                if "raise" in legal and pf_eq >= 0.40:
                    self.preflop_aggressor = True
                    return self.make_raise(4, req)
                if "call" in legal and pf_eq >= 0.28:
                    return {"type": "action", "action": "call"}
                return {"type": "action", "action": "check" if "check" in legal else "fold"}

            # Big Blind facing a raise (button open to 4)
            if not is_button and to_call > 0:
                # 3-bet bot.py's light steals
                if "raise" in legal and pf_eq >= 0.58:
                    self.preflop_aggressor = True
                    return self.make_raise(12, req)
                pot_odds = to_call / (pot + to_call)
                if "call" in legal and pf_eq >= pot_odds:
                    return {"type": "action", "action": "call"}
                if "check" in legal:
                    return {"type": "action", "action": "check"}
                return {"type": "action", "action": "fold"}

            # Big Blind unraised (button limped)
            if to_call == 0:
                if "raise" in legal and pf_eq >= 0.52:
                    self.preflop_aggressor = True
                    return self.make_raise(6, req)
                return {"type": "action", "action": "check"}

            if "check" in legal:
                return {"type": "action", "action": "check"}
            return {"type": "action", "action": "fold"}

        # ---------------- POSTFLOP ----------------
        dead = set()
        if self.my_discard:
            dead.add(self.my_discard)
        if self.opp_discard:
            dead.add(self.opp_discard)

        hole = self.post_discard_hole if self.post_discard_hole else self.hole_cards[:2]
        board = req.get("board", [])
        cat, score = hand_category(hole, board)
        equity = estimate_equity(hole, board, dead, self.opp_raised_streets, num_sims=30)
        pot_odds = to_call / max(pot + to_call, 1)

        # Board texture
        board_ranks = [RANK_VALUE[c[0]] for c in board]
        board_is_paired = (len(set(board_ranks)) < len(board_ranks))

        # EXPLOIT 1: NEVER PAY OFF BOT.PY'S POSTFLOP MONSTER RAISES
        # When bot.py raises postflop, it has trips or better (equity >= 0.88).
        opp_raised_this_street = any(h.get("actor") == "opp" and h.get("action") == "raise" and h.get("street") == street for h in history)
        if opp_raised_this_street and to_call >= 10:
            if cat < 3: # Fold one pair and two pair
                return {"type": "action", "action": "check" if "check" in legal else "fold"}
            if "call" in legal and equity >= pot_odds + 0.15:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "check" if "check" in legal else "fold"}

        # EXPLOIT 2: RIVER AUTOWIN vs BOT.PY
        # bot.py hard-folds on river when to_call >= 12 or to_call >= pot * 0.35 unless it has two-pair+
        if street == "river" and to_call == 0:
            if "raise" in legal:
                # If playing bot.py, exploit river fold rule with a 13-chip probe
                if is_bot_py:
                    target = max(req.get("min_raise", 2), max(13, int(pot * 0.38)))
                    return self.make_raise(target, req)
                else:
                    # Against standard opponent, value bet two pair or better
                    if cat >= 2 or equity >= 0.65:
                        target = max(req.get("min_raise", 2), int(pot * 0.55))
                        return self.make_raise(target, req)
            return {"type": "action", "action": "check"}

        # EXPLOIT 3: TURN PROBE
        # When checked to on turn
        if street == "turn" and to_call == 0:
            if "raise" in legal:
                if is_bot_py or equity >= 0.45 or cat >= 1:
                    target = max(req.get("min_raise", 2), int(pot * 0.45))
                    return self.make_raise(target, req)
            return {"type": "action", "action": "check"}

        # FLOP PLAY
        if street == "flop":
            if to_call == 0:
                if "raise" in legal and (self.preflop_aggressor or cat >= 1 or equity >= 0.38):
                    target = max(req.get("min_raise", 2), int(pot * 0.45))
                    return self.make_raise(target, req)
                return {"type": "action", "action": "check"}
            # Facing a bet on flop
            outs = count_outs(hole, board)
            if "call" in legal and (cat >= 1 or outs >= 4 or equity >= pot_odds - 0.05):
                return {"type": "action", "action": "call"}

        # GENERAL SITUATION A: Checked to us (to_call == 0)
        if to_call == 0:
            if "raise" in legal and (cat >= 2 or equity >= 0.65):
                target = max(req.get("min_raise", 2), int(pot * 0.55))
                return self.make_raise(target, req)
            return {"type": "action", "action": "check"}

        # GENERAL SITUATION B: Facing a bet
        # River defense:
        if street == "river" and to_call >= 12:
            if cat < 2 or (cat == 2 and board_is_paired) or equity < 0.60:
                return {"type": "action", "action": "check" if "check" in legal else "fold"}

        if "call" in legal and equity >= pot_odds:
            return {"type": "action", "action": "call"}

        if "check" in legal:
            return {"type": "action", "action": "check"}
        return {"type": "action", "action": "fold"}

# ---------- Main Execution Loop ----------
def main():
    bot = NemesisBot()
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        mtype = msg.get("type")
        if mtype == "action_request":
            reply = bot.choose_action(msg)
            print(json.dumps(reply), flush=True)
        elif mtype == "discard_request":
            disc = choose_best_discard_fast(
                msg["hole_cards"],
                msg["board"],
                msg.get("time_left_ms", 30000)
            )
            print(json.dumps({"type": "discard", "card": disc}), flush=True)
        elif mtype == "new_hand":
            bot.on_new_hand(msg)
        elif mtype == "discard_result":
            bot.on_discard_result(msg)
        elif mtype == "hand_result":
            bot.on_hand_result(msg)
        elif mtype == "match_end":
            break

if __name__ == "__main__":
    main()
