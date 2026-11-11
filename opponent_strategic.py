"""
Discard Hold'em — Elite Bot v2
Optimized for speed: fast heuristic evaluation + targeted Monte Carlo.
Uses only standard library (no numpy/scipy needed).
"""

import sys, json, random, math
from itertools import combinations
from collections import Counter

# ---------- constants ----------
RANKS = "23456789TJQKA"
SUITS = "shdc"
RANK_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}
SUIT_ORDER = {s: i for i, s in enumerate(SUITS)}
ALL_CARDS = [r + s for r in RANKS for s in SUITS]
FULL_DECK = set(ALL_CARDS)
RNG = random.Random()

STARTING_STACK = 200
SMALL_BLIND = 1
BIG_BLIND = 2

# ---------- hand evaluator ----------
def rank5(cards):
    """Strength of exactly five cards as a comparable tuple (higher is better)."""
    values = sorted((RANK_VALUE[c[0]] for c in cards), reverse=True)
    flush = len({c[1] for c in cards}) == 1
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
    """Best five-card hand from hole + board."""
    all_cards = list(hole) + list(board)
    return max(rank5(combo) for combo in combinations(all_cards, 5))

# ---------- fast hand category ----------
def hand_category(hole, board):
    """Return (category_number, score_tuple) for the best hand."""
    score = best_hand(hole, board)
    return score[0], score

# ---------- outs counting ----------
def count_outs(hole, board):
    """Rough count of outs for flush/straight draws after the flop."""
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

# ---------- Monte Carlo equity ----------
def estimate_equity(my_hole, board, dead_cards=None, num_sims=150):
    """Monte Carlo equity estimation."""
    if dead_cards is None:
        dead_cards = set()
    known = set(my_hole) | set(board) | dead_cards
    deck = [c for c in ALL_CARDS if c not in known]
    board_needed = 5 - len(board)
    wins = 0.0
    for _ in range(num_sims):
        RNG.shuffle(deck)
        opp = (deck[0], deck[1])
        sim_board = board + deck[2:2+board_needed] if board_needed > 0 else board
        my_score = best_hand(my_hole, sim_board)
        opp_score = best_hand(opp, sim_board)
        if my_score > opp_score:
            wins += 1.0
        elif my_score == opp_score:
            wins += 0.5
    return wins / num_sims

# ---------- opponent range model ----------
def pair_keep_score(hole, board):
    """Public-board hand and draw score used to model discard choices."""
    category, score_tuple = hand_category(hole, board)
    return category * 1000 + sum(score_tuple[1:]) + count_outs(hole, board) * 30

def discard_likelihood(kept, discarded, flop):
    """Soft likelihood that a player kept this pair and exposed this discard."""
    actual = pair_keep_score(kept, flop)
    alternatives = (
        pair_keep_score([kept[0], discarded], flop),
        pair_keep_score([kept[1], discarded], flop),
    )
    scores = (actual,) + alternatives
    peak = max(scores)
    weights = [math.exp(max(-20.0, (score - peak) / 350.0)) for score in scores]
    model_probability = weights[0] / sum(weights)
    # Keep uncertainty because opponents may use a different discard strategy.
    return 0.25 + 0.75 * model_probability

def strength_for_range(hole, street, board):
    if street == "preflop" and len(hole) == 3:
        return preflop_strength(hole)
    if not board:
        return preflop_strength(hole) if len(hole) == 3 else 0.5
    category, score = hand_category(hole, board)
    base = (0.08, 0.25, 0.45, 0.58, 0.69, 0.78, 0.87, 0.94, 0.99)[category]
    kicker = sum(score[1:]) / max(1, len(score) - 1)
    return min(1.0, base + kicker / 500.0)

def history_action_context(history, hero_seat):
    """Reconstruct rough pot and bet-size context for each public action."""
    hero_blind = 1 if hero_seat == "button" else 2
    opp_blind = 3 - hero_blind
    street_bets = {"preflop": {"me": hero_blind, "opp": opp_blind}}
    pot = 3
    contexts = []
    for entry in history:
        street = entry.get("street", "preflop")
        if street not in street_bets:
            street_bets[street] = {"me": 0, "opp": 0}
        bets = street_bets[street]
        actor = entry.get("actor", "opp")
        action = entry.get("action")
        amount = entry.get("amount")
        added = 0
        if action in ("call", "raise") and amount is not None:
            added = max(0, int(amount) - bets[actor])
        contexts.append({
            "pot": pot,
            "size_fraction": min(1.0, added / max(pot + added, 1)),
        })
        if added:
            pot += added
            bets[actor] = int(amount)
    return contexts


def action_likelihood(action, strength, size_fraction=0.0, bluff_rate=0.20):
    if action == "raise":
        size_fraction = min(1.0, max(0.0, size_fraction))
        threshold = 0.48 + 0.10 * size_fraction
        z = max(-12.0, min(12.0, (strength - threshold) * 10.0))
        value_likelihood = 0.04 + 0.94 / (1.0 + math.exp(-z))
        # Keep some probability mass on weak hands when this opponent has shown bluffs.
        bluff_rate = min(0.55, max(0.05, bluff_rate))
        return (1.0 - bluff_rate) * value_likelihood + bluff_rate * 0.35
    if action == "call":
        size_fraction = min(1.0, max(0.0, size_fraction))
        threshold = 0.30 + 0.08 * size_fraction
        z = max(-12.0, min(12.0, (strength - threshold) * 8.0))
        return 0.20 + 0.72 / (1.0 + math.exp(-z))
    if action == "check":
        return 0.45 + 0.35 * (1.0 - strength)
    if action == "fold":
        return max(0.08, 0.60 - strength)
    return 1.0

def opponent_range_weight(opp_hole, history, opp_discard=None, board=None,
                          hero_seat="button", bluff_rate=0.20, contexts=None):
    """Weight a candidate opponent holding using only visible actions/cards."""
    weight = 1.0
    if contexts is None:
        contexts = history_action_context(history, hero_seat)
    current_board = board or []
    for index, entry in enumerate(history):
        if entry.get("actor") != "opp":
            continue
        street = entry.get("street", "preflop")
        if street == "preflop":
            if opp_discard is not None:
                candidate = list(opp_hole) + [opp_discard]
            else:
                candidate = list(opp_hole)
        elif street == "flop":
            candidate = list(opp_hole)
            if opp_discard is not None:
                candidate.append(opp_discard)
        else:
            candidate = list(opp_hole)
        seen = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}.get(street, 0)
        strength = strength_for_range(candidate, street, current_board[:seen])
        context = contexts[index] if index < len(contexts) else {}
        observed_bluff_rate = bluff_rate if street != "preflop" else 0.10
        weight *= action_likelihood(
            entry.get("action"), strength, context.get("size_fraction", 0.0),
            observed_bluff_rate,
        )
    if history:
        # Temper accumulated evidence to avoid overfitting a short action sequence.
        weight = max(0.015, weight ** 0.65)
    if opp_discard is not None and current_board:
        weight *= discard_likelihood(opp_hole, opp_discard, current_board[:3])
    return weight

def estimate_range_equity(my_hole, board, dead_cards, history, opp_discard,
                          hero_seat="button", bluff_rate=0.20, num_sims=96):
    """Equity against an action- and discard-conditioned opponent range."""
    known = set(my_hole) | set(board) | set(dead_cards or ())
    deck = [card for card in ALL_CARDS if card not in known]
    board_needed = 5 - len(board)
    wins = 0.0
    total_weight = 0.0
    contexts = history_action_context(history, hero_seat)

    for _ in range(max(1, num_sims)):
        RNG.shuffle(deck)
        opp_hole = (deck[0], deck[1])
        sim_board = board + deck[2:2 + board_needed]
        weight = opponent_range_weight(
            opp_hole, history, opp_discard, board, hero_seat, bluff_rate, contexts,
        )
        my_score = best_hand(my_hole, sim_board)
        opp_score = best_hand(opp_hole, sim_board)
        wins += weight * (1.0 if my_score > opp_score else 0.5 if my_score == opp_score else 0.0)
        total_weight += weight
    return wins / total_weight if total_weight else 0.5

def estimate_flop_equities(hole_cards, board, history, hero_seat="button",
                           bluff_rate=0.20, num_sims=40):
    """Estimate showdown equity for each discard using the visible flop and action history."""
    keeps = [[card for j, card in enumerate(hole_cards) if j != i]
             for i in range(len(hole_cards))]
    known = set(hole_cards) | set(board)
    deck = [card for card in ALL_CARDS if card not in known]
    wins = [0.0] * len(keeps)
    total_weight = 0.0
    contexts = history_action_context(history, hero_seat)
    for _ in range(max(1, num_sims)):
        RNG.shuffle(deck)
        opp_three = tuple(deck[:3])
        full_board = board + deck[3:5]
        weight = opponent_range_weight(
            opp_three, history, board=board, hero_seat=hero_seat,
            bluff_rate=bluff_rate, contexts=contexts,
        )
        opp_discard_idx = max(
            range(3),
            key=lambda i: pair_keep_score([c for j, c in enumerate(opp_three) if j != i], board),
        )
        opp_keep = [card for j, card in enumerate(opp_three) if j != opp_discard_idx]
        opp_score = best_hand(opp_keep, full_board)
        for i, keep in enumerate(keeps):
            my_score = best_hand(keep, full_board)
            result = 1.0 if my_score > opp_score else 0.5 if my_score == opp_score else 0.0
            wins[i] += weight * result
        total_weight += weight
    return [value / total_weight for value in wins] if total_weight else [0.5] * len(keeps)

# ---------- preflop strength ----------
def preflop_strength(hole_cards):
    """Fast 3‑card hand strength score 0.0‑1.0."""
    cards = list(hole_cards)
    vals = sorted([RANK_VALUE[c[0]] for c in cards], reverse=True)
    suits = [c[1] for c in cards]
    score = 0.0
    cnt = Counter(vals)
    counts = sorted(cnt.values(), reverse=True)
    if counts[0] == 3:
        return min(0.90 + vals[0] / 140.0, 1.0)
    if counts[0] == 2:
        pair_rank = max(v for v, c in cnt.items() if c == 2)
        score = 0.45 + pair_rank / 70.0
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
                score += 0.03
            elif gap <= 4:
                score += 0.01
    if 14 in vals:
        score += 0.05
    return min(score, 1.0)


def estimate_preflop_equity(hole_cards, history, hero_seat, bluff_rate,
                            num_sims=24):
    """Estimate preflop showdown equity after both players choose a flop discard."""
    known = set(hole_cards)
    deck = [card for card in ALL_CARDS if card not in known]
    wins = total_weight = 0.0
    contexts = history_action_context(history, hero_seat)
    for _ in range(max(1, num_sims)):
        RNG.shuffle(deck)
        opp_three = deck[:3]
        flop = deck[3:6]
        full_board = deck[3:8]
        hero_keep = max(
            (card for card in hole_cards if card),
            key=lambda discarded: pair_keep_score(
                [card for card in hole_cards if card != discarded], flop,
            ),
        )
        hero_pair = [card for card in hole_cards if card != hero_keep]
        opp_discard = max(
            opp_three,
            key=lambda discarded: pair_keep_score(
                [card for card in opp_three if card != discarded], flop,
            ),
        )
        opp_pair = [card for card in opp_three if card != opp_discard]
        weight = opponent_range_weight(
            opp_three, history, board=[], hero_seat=hero_seat,
            bluff_rate=bluff_rate, contexts=contexts,
        )
        hero_score = best_hand(hero_pair, full_board)
        opp_score = best_hand(opp_pair, full_board)
        result = 1.0 if hero_score > opp_score else 0.5 if hero_score == opp_score else 0.0
        wins += weight * result
        total_weight += weight
    return wins / total_weight if total_weight else 0.5

# ---------- discard optimization ----------
def choose_best_discard(hole_cards, board, time_budget):
    candidates = []
    for i, discard_card in enumerate(hole_cards):
        remaining = [c for j, c in enumerate(hole_cards) if j != i]
        h_score = pair_keep_score(remaining, board)
        candidates.append((i, discard_card, remaining, h_score))
    candidates.sort(key=lambda x: x[3], reverse=True)
    # The caller normally supplies a cached joint estimate from the flop action.
    # Retain a cheap fallback for all-in hands that skipped flop betting.
    sims_per = 32 if time_budget < 10000 else 40
    equities = estimate_flop_equities(hole_cards, board, [], num_sims=min(sims_per, 24))
    best_index = max(range(len(equities)), key=lambda i: equities[i])
    return hole_cards[best_index]

# ---------- bot state ----------
class BotState:
    def __init__(self):
        self.seat = None
        self.hole_cards = []
        self.board = []
        self.hand_id = 0
        self.my_discard = None
        self.opp_discard = None
        self.post_discard_hole = []
        self.street = "preflop"
        self.opp_fold_count = 0
        self.opp_call_count = 0
        self.opp_total_hands = 0
        self.opp_raises = 0
        self.opp_actions = 0
        self.opp_postflop_raise_showdowns = 0
        self.opp_weak_raise_showdowns = 0
        self.hands_played = 0
        self.latest_history = []
        self.flop_cache_signature = None
        self.flop_cached_discard = None
    def on_new_hand(self, msg):
        self.seat = msg["seat"]
        self.hole_cards = list(msg["hole_cards"])
        self.board = []
        self.hand_id = msg["hand_id"]
        self.my_discard = None
        self.opp_discard = None
        self.post_discard_hole = []
        self.street = "preflop"
        self.latest_history = []
        self.flop_cache_signature = None
        self.flop_cached_discard = None
        self.opp_total_hands += 1
    def on_discard_result(self, msg):
        self.my_discard = msg["my_discard"]
        self.opp_discard = msg["opp_discard"]
        self.post_discard_hole = list(msg["hole_cards"])
    def on_hand_result(self, msg):
        self.hands_played += 1
        history = msg.get("history", [])
        for entry in history:
            if entry["actor"] == "opp":
                self.opp_actions += 1
                if entry["action"] == "raise":
                    self.opp_raises += 1
                if entry["action"] == "fold":
                    self.opp_fold_count += 1
                elif entry["action"] == "call":
                    self.opp_call_count += 1

        # Showdown cards give a noisy but useful sample of whether this opponent
        # raises postflop with air. Count at most one sample per hand.
        opp_hole = msg.get("opp_hole_cards")
        board = msg.get("board", [])
        shown_raises = [entry for entry in history
                        if entry.get("actor") == "opp"
                        and entry.get("action") == "raise"
                        and entry.get("street") in ("flop", "turn", "river")]
        if opp_hole and len(opp_hole) == 2 and shown_raises:
            latest = shown_raises[-1]
            visible = board[:{"flop": 3, "turn": 4, "river": 5}[latest["street"]]]
            category, _ = hand_category(opp_hole, visible)
            if category == 0 and count_outs(opp_hole, visible) < 4:
                self.opp_weak_raise_showdowns += 1
            self.opp_postflop_raise_showdowns += 1
    @property
    def opp_fold_rate(self):
        # Smoothed rate among observed fold-or-call responses to a bet.
        return (2.0 + self.opp_fold_count) / (
            10.0 + self.opp_fold_count + self.opp_call_count
        )
    @property
    def opp_agg_freq(self):
        if self.opp_actions < 10:
            return 0.3
        return self.opp_raises / max(self.opp_actions, 1)
    @property
    def opp_bluff_rate(self):
        # Beta prior centered at 20%; showdown samples are sparse and selective.
        return min(0.55, (2.0 + self.opp_weak_raise_showdowns) /
                   (10.0 + self.opp_postflop_raise_showdowns))
    @property
    def is_button(self):
        return self.seat == "button"

# ---------- action decision engine ----------
def choose_action(req, state):
    street = req["street"]
    board = req.get("board", [])
    history = req.get("history", [])
    state.street = street
    state.board = board
    state.latest_history = history
    if street == "preflop":
        equity = estimate_preflop_equity(
            state.hole_cards, history, state.seat, state.opp_bluff_rate,
            num_sims=response_sim_budget(req, 12),
        )
        call_price = req["to_call"] / max(req["pot"] + req["to_call"], 1)
        if (req.get("time_left_ms", 30_000) >= 16_000
                and (abs(equity - call_price) < 0.12 or req["pot"] >= 70
                     or abs(equity - 0.68) < 0.10)):
            equity = estimate_preflop_equity(
                state.hole_cards, history, state.seat, state.opp_bluff_rate,
                num_sims=response_sim_budget(req, 32),
            )
        return preflop_action(equity, req, state)
    # post‑flop
    dead = set()
    if state.my_discard:
        dead.add(state.my_discard)
    if state.opp_discard:
        dead.add(state.opp_discard)
    if state.post_discard_hole:
        my_hole = state.post_discard_hole
    elif len(state.hole_cards) == 3 and street == "flop":
        equities = estimate_flop_equities(
            state.hole_cards, board, history, state.seat,
            state.opp_bluff_rate, num_sims=response_sim_budget(req, 16),
        )
        discard_index = max(range(3), key=lambda i: equities[i])
        state.flop_cache_signature = (tuple(sorted(state.hole_cards)), tuple(board))
        state.flop_cached_discard = state.hole_cards[discard_index]
        return postflop_action(max(equities), req, state, street)
    else:
        my_hole = state.hole_cards[:2]
    num_sims = response_sim_budget(req, 32 if street == "turn" else 64)
    equity = estimate_range_equity(
        my_hole, board, dead, history, state.opp_discard,
        hero_seat=state.seat, bluff_rate=state.opp_bluff_rate, num_sims=num_sims,
    )
    # Spend a little more compute only when the first estimate is near a
    # decision boundary or a large pot makes an error especially costly.
    if req.get("to_call", 0) > 0:
        call_margin = equity - req["to_call"] / max(req["pot"] + req["to_call"], 1)
        close_decision = abs(call_margin) < 0.10
    else:
        close_decision = abs(equity - 0.62) < 0.10
    if ((close_decision or req.get("pot", 0) >= 70)
            and req.get("time_left_ms", 30_000) >= 14_000):
        equity = estimate_range_equity(
            my_hole, board, dead, history, state.opp_discard,
            hero_seat=state.seat, bluff_rate=state.opp_bluff_rate,
            num_sims=response_sim_budget(req, 96 if street == "river" else 64),
        )
    return postflop_action(equity, req, state, street)

def preflop_action(equity, req, state):
    legal = req["legal_actions"]
    to_call = req["to_call"]
    pot = req["pot"]
    fold_adj = 0.0
    if state.opp_fold_rate > 0.4 and state.opp_total_hands > 15:
        fold_adj = -0.07
    if state.opp_agg_freq > 0.45 and state.opp_total_hands > 15:
        fold_adj += 0.04
    raise_threshold = 0.62 + fold_adj
    big_raise_threshold = 0.74 + fold_adj
    if to_call == 0:
        if "raise" in legal and equity >= big_raise_threshold:
            return make_raise(max(req.get("min_raise", 4), int(pot * 2.3)), req)
        if "raise" in legal and equity >= raise_threshold:
            return make_raise(max(req.get("min_raise", 4), int(pot * 1.8)), req)
        # A small button steal is worthwhile only against a demonstrated folder
        # and a hand with reasonable equity/blockers.
        steal_frequency = min(0.20, max(0.0, (state.opp_fold_rate - 0.30) * 0.45))
        has_blocker = any(card[0] in "AKQ" for card in state.hole_cards)
        if ("raise" in legal and state.is_button and equity >= 0.40 and has_blocker
                and random.random() < steal_frequency):
            return make_raise(max(req.get("min_raise", 4), min(6, req.get("max_raise", 200))), req)
        return {"type": "action", "action": "check"}

    pot_odds = to_call / max(pot + to_call, 1)
    call_ev = equity * (pot + to_call) - to_call
    if "raise" in legal and equity >= max(big_raise_threshold, pot_odds + 0.25):
        return make_raise(max(req.get("min_raise", 4), int(req.get("opp_bet", 0) + pot * 0.8)), req)
    if "call" in legal and call_ev > 0:
        return {"type": "action", "action": "call"}
    if "check" in legal:
        return {"type": "action", "action": "check"}
    return {"type": "action", "action": "fold"}
    return {"type": "action", "action": "fold"}

def bluff_frequency(hole, board, history, state, street):
    """Small, randomized bluff frequency for weak hands with useful blockers."""
    if len(board) < 3 or len(hole) != 2 or street not in ("flop", "turn", "river"):
        return 0.0

    preflop_aggressor = any(
        entry.get("actor") == "me" and entry.get("street") == "preflop"
        and entry.get("action") == "raise" for entry in history
    )
    street_actions = [entry for entry in history if entry.get("street") == street]
    opponent_checked = any(
        entry.get("actor") == "opp" and entry.get("action") == "check"
        for entry in street_actions
    )

    board_ranks = {RANK_VALUE[card[0]] for card in board}
    blockers = sum(1 for card in hole if card[0] in "AK") * 0.025
    if preflop_aggressor and board_ranks.intersection({14, 13}):
        blockers += 0.045
    board_suits = Counter(card[1] for card in board)
    if any(board_suits[suit] >= 3 and any(card[1] == suit for card in hole)
           for suit in board_suits):
        blockers += 0.05

    fold_edge = max(0.0, state.opp_fold_rate - 0.20) * 0.28
    initiative = 0.035 if preflop_aggressor else 0.0
    checked_bonus = 0.035 if opponent_checked and state.is_button else 0.0
    station_penalty = 0.04 if state.opp_fold_rate < 0.15 else 0.0
    return min(0.18, max(0.0,
        0.025 + fold_edge + blockers + initiative + checked_bonus - station_penalty
    ))


def estimate_fold_probability(state, bet_size, pot):
    """Coarse opponent-specific fold estimate, adjusted for pressure size."""
    size_to_pot = bet_size / max(pot, 1)
    pressure_adjustment = 0.10 * (size_to_pot - 0.50)
    return min(0.78, max(0.06, state.opp_fold_rate + pressure_adjustment))


def bet_ev(equity, cost, pot, state):
    """Expected chip gain from betting, with a conservative called-range haircut."""
    if cost <= 0:
        return float("-inf")
    fold_probability = estimate_fold_probability(state, cost, pot)
    called_equity = max(0.0, equity - min(0.16, 0.04 + 0.06 * cost / max(pot, 1)))
    called_pot = pot + 2 * cost
    return fold_probability * pot + (1.0 - fold_probability) * (
        called_equity * called_pot - cost
    )


def raise_ev(equity, target, req, state):
    """Expected chip gain when raising to target and allowing a fold or call."""
    cost = target - req.get("my_bet", 0)
    opp_extra = max(0, target - req.get("opp_bet", 0))
    if cost <= 0:
        return float("-inf")
    fold_probability = estimate_fold_probability(state, cost, req["pot"])
    called_equity = max(0.0, equity - min(0.18, 0.05 + 0.06 * cost / max(req["pot"], 1)))
    called_pot = req["pot"] + cost + opp_extra
    return fold_probability * req["pot"] + (1.0 - fold_probability) * (
        called_equity * called_pot - cost
    )


def postflop_action(equity, req, state, street):
    legal = req["legal_actions"]
    pot = req["pot"]
    to_call = req["to_call"]
    if to_call == 0:
        if "raise" not in legal:
            return {"type": "action", "action": "check"}
        min_raise = req.get("min_raise", 2)
        max_raise = req.get("max_raise", 200)
        my_bet = req.get("my_bet", 0)
        if equity < 0.62:
            # Before the simultaneous discard, use the best estimated kept pair
            # for blocker checks instead of blindly taking the first two cards.
            bluff_hole = state.post_discard_hole
            if not bluff_hole and state.flop_cached_discard in state.hole_cards:
                bluff_hole = [card for card in state.hole_cards
                              if card != state.flop_cached_discard]
            if not bluff_hole:
                bluff_hole = state.hole_cards[:2]
            bluff_rate = bluff_frequency(
                bluff_hole, req.get("board", []), req.get("history", []), state, street,
            )
            target = max(min_raise, int(pot * 0.40))
            if (equity <= 0.38 and target <= max_raise
                    and min_raise <= max(2, int(pot * 0.75))
                    and random.random() < bluff_rate
                    and bet_ev(equity, target - my_bet, pot, state) > 0):
                return make_raise(target, req)
            return {"type": "action", "action": "check"}

        # Compare a modest value bet with checking. A conservative equity
        # haircut models the stronger hands that continue against a bet.
        targets = {max(min_raise, int(pot * fraction)) for fraction in (0.40, 0.65)}
        targets = [target for target in targets if target <= max_raise]
        best = max(targets, key=lambda target: bet_ev(
            equity, target - my_bet, pot, state,
        ), default=None)
        if best is not None and bet_ev(equity, best - my_bet, pot, state) > 0:
            return make_raise(best, req)
        return {"type": "action", "action": "check"}

    pot_odds = to_call / max(pot + to_call, 1)
    call_ev = equity * (pot + to_call) - to_call
    best_raise = None
    best_raise_ev = float("-inf")
    if "raise" in legal:
        minimum = req.get("min_raise", 0)
        maximum = req.get("max_raise", minimum)
        candidates = {minimum, min(maximum, req.get("opp_bet", 0) + int(pot * 0.50))}
        for target in candidates:
            if minimum <= target <= maximum:
                value = raise_ev(equity, target, req, state)
                if value > best_raise_ev:
                    best_raise, best_raise_ev = target, value
    # The action-conditioned range includes this opponent's observed bluff
    # rate, so marginal calls can correctly catch more of their bluffs.
    if best_raise is not None and best_raise_ev > max(0.0, call_ev):
        return make_raise(best_raise, req)
    if "call" in legal and call_ev > 0:
        return {"type": "action", "action": "call"}
    if "check" in legal:
        return {"type": "action", "action": "check"}
    return {"type": "action", "action": "fold"}

def make_raise(target, req):
    mn = req.get("min_raise", 2)
    mx = req.get("max_raise", 200)
    amount = max(mn, min(target, mx))
    return {"type": "action", "action": "raise", "amount": amount}


def response_sim_budget(req, normal, floor=6):
    """Reduce optional Monte Carlo work as the leg's remaining bank shrinks."""
    left = max(0, req.get("time_left_ms", 30_000))
    if left < 8_000:
        fraction = 0.25
    elif left < 14_000:
        fraction = 0.40
    elif left < 22_000:
        fraction = 0.65
    else:
        fraction = 1.0
    return max(floor, min(normal, int(normal * fraction)))

# ---------- discard decision ----------
def do_discard(req, state):
    cards = req["hole_cards"]
    board = req["board"]
    signature = (tuple(sorted(cards)), tuple(board))
    if signature == state.flop_cache_signature and state.flop_cached_discard in cards:
        return {"type": "discard", "card": state.flop_cached_discard}
    time_left = req.get("time_left_ms", 30000)
    equities = estimate_flop_equities(
        cards, board, state.latest_history, state.seat,
        state.opp_bluff_rate, num_sims=response_sim_budget(req, 12),
    )
    best_card = cards[max(range(len(equities)), key=lambda i: equities[i])]
    return {"type": "discard", "card": best_card}

# ---------- main loop ----------
def main():
    state = BotState()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        t = msg.get("type")
        if t == "action_request":
            reply = choose_action(msg, state)
            print(json.dumps(reply), flush=True)
        elif t == "discard_request":
            reply = do_discard(msg, state)
            print(json.dumps(reply), flush=True)
        elif t == "new_hand":
            state.on_new_hand(msg)
        elif t == "discard_result":
            state.on_discard_result(msg)
        elif t == "hand_result":
            state.on_hand_result(msg)
        elif t == "match_end":
            break

if __name__ == "__main__":
    main()
