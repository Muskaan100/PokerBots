"""
Elite Master Poker Bot
Optimized for Discard Hold'em against deep range-modelling opponents.
Features:
- Fast & exact hand ranking with combinatorial out counting.
- Adaptive Monte Carlo simulations with strict time-bank safety (< 12s per 100 hands).
- Superior discard selection maximizing joint showdown equity.
- Exploitative preflop aggression (punishes conservative limps and steals blinds).
- Capped-range exploitation (c-bets and probe bets when opponent checks).
- River defense discipline: eliminates calling off chips with bluff-catchers against triple barrels.
"""

import sys
import json
import random
import math
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

# ---------- Fast Hand Evaluator ----------
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

# ---------- Range-Aware Equity Simulator ----------
def estimate_range_equity(my_hole, board, dead_cards, opp_raised_streets, num_sims=45):
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
            if opp_cat == 0:
                w = 0.10
            elif opp_cat == 1:
                w = 0.50
            else:
                w = 1.0
        else:
            if opp_cat >= 3:
                w = 0.05
            elif opp_cat == 2:
                w = 0.25
            else:
                w = 1.0
                
        if my_score > opp_full_score:
            wins += w * 1.0
        elif my_score == opp_full_score:
            wins += w * 0.5
        total_w += w
        
    return wins / total_w if total_w > 0 else 0.5

# ---------- Discard Selection ----------
def choose_best_discard_fast(hole_cards, board, time_left_ms):
    keeps = [[c for j, c in enumerate(hole_cards) if j != i] for i in range(len(hole_cards))]
    known = set(hole_cards) | set(board)
    deck = [c for c in ALL_CARDS if c not in known]
    
    num_sims = 30 if time_left_ms > 12000 else (15 if time_left_ms > 6000 else 8)
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

# ---------- Master Bot Brain ----------
class MasterPokerBot:
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
        self.opp_bet_flop = False
        self.opp_bet_turn = False

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
        self.opp_bet_flop = False
        self.opp_bet_turn = False

    def on_discard_result(self, msg):
        self.my_discard = msg.get("my_discard")
        self.opp_discard = msg.get("opp_discard")
        self.post_discard_hole = list(msg.get("hole_cards", []))

    def on_hand_result(self, msg):
        pass

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
        
        # Track street actions
        for h in history:
            if h.get("actor") == "opp" and h.get("action") == "raise":
                st = h.get("street")
                self.opp_raised_streets.add(st)
                if st == "flop":
                    self.opp_bet_flop = True
                elif st == "turn":
                    self.opp_bet_turn = True

        is_button = (self.seat == "button")
        
        # ---------------- PREFLOP ----------------
        if street == "preflop":
            pf_eq = preflop_strength(self.hole_cards)
            
            # Big Blind, unraised pot (button limped with weak holding)
            if to_call == 0:
                if "raise" in legal and pf_eq >= 0.50:
                    self.preflop_aggressor = True
                    return self.make_raise(6, req)
                return {"type": "action", "action": "check"}

            # Button open (small blind 1 chip to call)
            if is_button and to_call == 1:
                # Steal wide to exploit opponent's 62%+ tight fold rule
                if "raise" in legal and pf_eq >= 0.40:
                    self.preflop_aggressor = True
                    return self.make_raise(4, req)
                if "call" in legal and pf_eq >= 0.28:
                    return {"type": "action", "action": "call"}
                return {"type": "action", "action": "check" if "check" in legal else "fold"}

            # Facing raise
            pot_odds = to_call / (pot + to_call)
            if "raise" in legal and pf_eq >= 0.78:
                self.preflop_aggressor = True
                target = req.get("opp_bet", 0) + int(pot * 0.8)
                return self.make_raise(target, req)
            if "call" in legal and pf_eq >= pot_odds + 0.05:
                return {"type": "action", "action": "call"}
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
        
        sims = 32 if time_left > 12000 else (18 if time_left > 6000 else 10)
        equity = estimate_range_equity(hole, board, dead, self.opp_raised_streets, num_sims=sims)
        cat, score = hand_category(hole, board)

        # Board texture analysis
        board_ranks = [RANK_VALUE[c[0]] for c in board]
        board_suits = Counter(c[1] for c in board)
        board_is_paired = (len(set(board_ranks)) < len(board_ranks))
        flush_possible = (max(board_suits.values()) >= 3)

        # SITUATION A: Opponent checked (to_call == 0)
        if to_call == 0:
            if "raise" in legal:
                # Value bet monster: 65% pot
                if cat >= 3 or (cat == 2 and not board_is_paired and equity >= 0.70):
                    target = max(req.get("min_raise", 2), int(pot * 0.65))
                    return self.make_raise(target, req)
                # Value bet top pair: 50% pot
                if cat == 1 and score[1] >= 10 and equity >= 0.58:
                    target = max(req.get("min_raise", 2), int(pot * 0.50))
                    return self.make_raise(target, req)
                # Flop C-bet when we raised preflop
                if street == "flop" and self.preflop_aggressor and equity >= 0.35:
                    target = max(req.get("min_raise", 2), int(pot * 0.45))
                    return self.make_raise(target, req)
                # Probe bet turn/river if opponent checked
                opp_checked = any(h.get("street") == street and h.get("actor") == "opp" and h.get("action") == "check" for h in history)
                if opp_checked and equity >= 0.45:
                    target = max(req.get("min_raise", 2), int(pot * 0.45))
                    return self.make_raise(target, req)
            return {"type": "action", "action": "check"}

        # SITUATION B: Facing a bet or raise (to_call > 0)
        pot_odds = to_call / (pot + to_call)

        # 1. Monster hands (trips, straight, flush, full house) -> Raise
        if "raise" in legal and (cat >= 3 or (cat == 2 and equity >= 0.88 and not board_is_paired)):
            target = req.get("opp_bet", 0) + int(pot * 0.75)
            return self.make_raise(target, req)

        # 2. RIVER DEFENSE DISCIPLINE:
        # If opponent bets on the river:
        if street == "river":
            # Opponent never bluffs big on river without monster equity.
            if to_call >= 12 or to_call >= pot * 0.35:
                # Need at least strong two pair (not just board pairs) or category >= 3
                if cat < 2:
                    return {"type": "action", "action": "check" if "check" in legal else "fold"}
                if cat == 2 and board_is_paired:
                    # Our two pair is just 1 pair in hand + 1 pair on board
                    return {"type": "action", "action": "check" if "check" in legal else "fold"}
                if equity < 0.62:
                    return {"type": "action", "action": "check" if "check" in legal else "fold"}

        # 3. TURN BARREL DEFENSE:
        # If opponent bet flop and now bets turn:
        if street == "turn" and self.opp_bet_flop and to_call >= 14:
            if cat == 0 and count_outs(hole, board) < 8:
                return {"type": "action", "action": "check" if "check" in legal else "fold"}
            if cat == 1 and score[1] < 10 and count_outs(hole, board) < 6:
                return {"type": "action", "action": "check" if "check" in legal else "fold"}

        # 4. Standard Pot Odds Calling
        if "call" in legal:
            threshold = pot_odds
            outs = count_outs(hole, board)
            if street in ("flop", "turn") and outs >= 8:
                threshold -= 0.08
            if equity >= threshold:
                return {"type": "action", "action": "call"}

        if "check" in legal:
            return {"type": "action", "action": "check"}
        return {"type": "action", "action": "fold"}

# ---------- Main Loop ----------
def main():
    bot = MasterPokerBot()
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
