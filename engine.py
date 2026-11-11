#!/usr/bin/env python3
"""
Pokerbots local match runner: Discard Hold'em, protocol 1.0.

    python3.11 engine.py BOT_A BOT_B [--stage 1|2] [--hands N] [--seed S]
                         [--log-dir DIR] [--verbose] [--no-time-limits]

Runs a full match between two bots under the same rules as the official engine:
duplicate deals, alternating seats, no-limit betting with a 4-raise cap, the
simultaneous exposed discard, time banks, error corrections and forfeits.

A match is played in two legs. Each leg launches fresh bot processes and plays
every deal exactly once; the second leg replays the same deals with the hole
cards swapped between the bots. No bot process ever sees both hands of a
duplicate deal.
Standard library only. Teams may reuse any code in this file (Rulebook 10.1).
"""

import argparse
import json
import math
import os
import queue
import random
import subprocess
import sys
import threading
import time
from collections import Counter
from itertools import combinations

# ------------------------------------------------------------------ constants

RANKS = "23456789TJQKA"
SUITS = "shdc"  # also the tie-break order for the default discard
RANK_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}
SUIT_ORDER = {s: i for i, s in enumerate(SUITS)}
VALID_CARDS = {r + s for r in RANKS for s in SUITS}
STREETS = ("preflop", "flop", "turn", "river")
BOARD_SEEN = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
SEAT_NAMES = ("button", "big_blind")

STARTING_STACK = 200
SMALL_BLIND = 1
BIG_BLIND = 2
MAX_RAISES = 4  # bets or raises per street; the big blind counts preflop
PER_RESPONSE_MS = 5000
STARTUP_S = 5.0
STDERR_LIMIT = 64 * 1024
LEGS = 2

# "hands" is the whole match (both legs); time bank and wall clock are per leg.
STAGES = {
    1: {"hands": 400, "time_bank_ms": 30_000, "wallclock_s": 90, "cap": 400},
    2: {"hands": 1000, "time_bank_ms": 75_000, "wallclock_s": 225, "cap": 1000},
}

HAND_NAMES = ("high card", "one pair", "two pair", "three of a kind", "straight",
              "flush", "full house", "four of a kind", "straight flush")


# ------------------------------------------------------------- hand evaluator

def rank5(cards):
    """Strength of exactly five cards as a comparable tuple (higher is better)."""
    values = sorted((RANK_VALUE[c[0]] for c in cards), reverse=True)
    flush = len({c[1] for c in cards}) == 1
    uniq = sorted(set(values), reverse=True)
    straight_high = 0
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_high = uniq[0]
        elif uniq == [14, 5, 4, 3, 2]:  # A-2-3-4-5, the only low-ace straight
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
    """Best five of hole + board, using two, one or none of the hole cards."""
    return max(rank5(combo) for combo in combinations(list(hole) + list(board), 5))


def default_discard(hole):
    """Lowest rank; among equal ranks, spades, hearts, diamonds, clubs first."""
    return min(hole, key=lambda c: (RANK_VALUE[c[0]], SUIT_ORDER[c[1]]))


# ------------------------------------------------------------ reply parsing

def parse_action(line, req):
    """Return (action, amount, error_code). error_code is None if the reply is legal."""
    try:
        msg = json.loads(line)
    except Exception:
        return None, None, "malformed"
    if not isinstance(msg, dict) or msg.get("type") != "action" \
            or not isinstance(msg.get("action"), str):
        return None, None, "malformed"
    action = msg["action"]
    if action not in req["legal_actions"]:
        return None, None, "illegal_action"
    if action == "raise":
        amount = msg.get("amount")
        if isinstance(amount, bool) or not isinstance(amount, int):
            return None, None, "malformed"
        if not req["min_raise"] <= amount <= req["max_raise"]:
            return None, None, "illegal_action"
        return action, amount, None
    return action, None, None


def parse_discard(line, hole):
    """Return (card, error_code). error_code is None if the discard is legal."""
    try:
        msg = json.loads(line)
    except Exception:
        return None, "malformed"
    if not isinstance(msg, dict) or msg.get("type") != "discard":
        return None, "malformed"
    card = msg.get("card")
    if not isinstance(card, str) or card not in VALID_CARDS:
        return None, "malformed"
    if card not in hole:
        return None, "illegal_discard"
    return card, None


# ---------------------------------------------------------------- bot process

class Forfeit(Exception):
    def __init__(self, failures):
        super().__init__(failures)
        self.failures = failures  # list of (bot, reason)


class Bot:
    def __init__(self, name, path, python, log_dir, time_bank_ms, per_response_ms,
                 log_name=None):
        self.name = name
        self.path = os.path.abspath(path)
        try:
            self.label = os.path.relpath(self.path)
        except ValueError:  # bot on a different drive (Windows)
            self.label = self.path
        self.time_left_ms = float(time_bank_ms)
        self.per_response_ms = float(per_response_ms)
        self.time_used_s = 0.0
        self.notices = 0
        self.crashed = False
        self.eof = False
        self.sent_at = 0.0
        self.lines = queue.Queue()
        self.stderr_bytes = 0
        self.stderr_file = None
        if log_dir:
            fname = f"{log_name or name}_stderr.txt"
            self.stderr_file = open(os.path.join(log_dir, fname), "wb")
        self.launch_time = time.monotonic()
        self.startup_deadline = self.launch_time + STARTUP_S
        self.proc = subprocess.Popen(
            [python, self.path],
            cwd=os.path.dirname(self.path),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()

    def _read_stdout(self):
        for raw in self.proc.stdout:
            self.lines.put((time.monotonic(), raw.decode("utf-8", "replace").rstrip("\r\n")))
        self.lines.put((time.monotonic(), None))  # end of file

    def _read_stderr(self):
        while True:
            chunk = self.proc.stderr.read1(4096)
            if not chunk:
                break
            if self.stderr_file and self.stderr_bytes < STDERR_LIMIT:
                keep = chunk[:STDERR_LIMIT - self.stderr_bytes]
                self.stderr_file.write(keep)
                self.stderr_file.flush()
                self.stderr_bytes += len(keep)

    def send(self, msg):
        if self.crashed:
            return False
        try:
            data = json.dumps(msg, separators=(",", ":")) + "\n"
            self.proc.stdin.write(data.encode("utf-8"))
            self.proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError, ValueError):
            self.crashed = True
            return False

    def _drain(self):
        """Discard lines sent while no request was outstanding (they are ignored)."""
        while True:
            try:
                _, line = self.lines.get_nowait()
            except queue.Empty:
                return
            if line is None:
                self.eof = True

    def begin(self, msg):
        self._drain()
        msg["time_left_ms"] = int(min(self.time_left_ms, 10 ** 9))
        self.sent_at = time.monotonic()
        self.send(msg)

    def wait_reply(self, wall_deadline):
        """Return (line, None) or (None, forfeit_reason)."""
        if self.crashed or self.eof:
            return None, "crashed or closed stdout"
        start = max(self.sent_at, self.startup_deadline)  # start-up time is not charged
        bank_limited = self.time_left_ms < self.per_response_ms
        deadline = start + min(self.per_response_ms, self.time_left_ms) / 1000.0
        hard = min(deadline, wall_deadline)

        def timeout_reason():
            if wall_deadline < deadline:
                return "wall-clock kill"
            return "time bank exhausted" if bank_limited else "reply took longer than 5 s"

        while True:
            try:
                ts, line = self.lines.get(timeout=max(0.0, min(hard - time.monotonic(), 0.5)))
            except queue.Empty:
                if time.monotonic() >= hard:
                    return None, timeout_reason()
                continue
            if line is None:
                self.eof = True
                return None, "crashed or closed stdout"
            if ts < self.sent_at:
                continue  # sent before this request was outstanding: ignored
            if ts > hard:
                return None, timeout_reason()
            charged = max(0.0, ts - start)
            self.time_left_ms -= charged * 1000.0
            self.time_used_s += charged
            return line, None

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self._stderr_thread.join(timeout=1)
        if self.stderr_file:
            self.stderr_file.close()


# ----------------------------------------------------------------------- hand

class Hand:
    def __init__(self, match, hand_id, button_bot, seat_cards, board):
        self.match = match
        self.id = hand_id
        self.bots = (match.bots[button_bot], match.bots[1 - button_bot])  # by seat
        self.dealt = (list(seat_cards[0]), list(seat_cards[1]))
        self.hole = [list(seat_cards[0]), list(seat_cards[1])]
        self.board = list(board)  # fixed before any action
        self.board_seen = 0
        self.stacks = [STARTING_STACK - SMALL_BLIND, STARTING_STACK - BIG_BLIND]
        self.committed = [SMALL_BLIND, BIG_BLIND]
        self.history = []  # (street, seat, action, amount or None)
        self.discards = [None, None]
        self.result = None
        self.scores = None

    def history_for(self, seat):
        out = []
        for street, actor, action, amount in self.history:
            entry = {"street": street, "actor": "me" if actor == seat else "opp", "action": action}
            if amount is not None:
                entry["amount"] = amount
            out.append(entry)
        return out

    def play(self):
        for seat in (0, 1):
            self.bots[seat].send({
                "type": "new_hand", "hand_id": self.id, "seat": SEAT_NAMES[seat],
                "hole_cards": list(self.hole[seat]),
                "stack": STARTING_STACK, "opp_stack": STARTING_STACK,
            })
        folder = None
        for street in STREETS:
            self.board_seen = BOARD_SEEN[street]
            folder = self.betting_round(street)
            if folder is not None:
                break
            if street == "flop":
                self.discard()
        return self.finish(folder)

    def ask_action(self, seat, req):
        bot = self.bots[seat]
        bot.begin(req)
        line, err = bot.wait_reply(self.match.wall_deadline)
        if err:
            raise Forfeit([(bot, err)])
        action, amount, code = parse_action(line, req)
        if code:
            action = "check" if "check" in req["legal_actions"] else "fold"
            amount = None
            bot.send({"type": "notice", "code": code, "applied": action})
            bot.notices += 1
        return action, amount

    def betting_round(self, street):
        """Play one street. Return the seat that folded, or None."""
        if street == "preflop":
            bets, last_raise, raises, actor = [SMALL_BLIND, BIG_BLIND], BIG_BLIND, 1, 0
        else:
            bets, last_raise, raises, actor = [0, 0], 0, 0, 1
        if 0 in self.stacks:  # someone is all-in: no more betting this hand
            return None
        acted = [False, False]
        while True:
            opp = 1 - actor
            to_call = bets[opp] - bets[actor]
            max_raise = min(bets[actor] + self.stacks[actor], bets[opp] + self.stacks[opp])
            legal = ["fold", "call"] if to_call > 0 else ["check"]
            can_raise = raises < MAX_RAISES and max_raise > bets[opp]
            if can_raise:
                legal.append("raise")
            req = {
                "type": "action_request", "hand_id": self.id, "street": street,
                "board": self.board[:self.board_seen], "pot": sum(self.committed),
                "my_bet": bets[actor], "opp_bet": bets[opp], "to_call": to_call,
                "my_stack": self.stacks[actor], "opp_stack": self.stacks[opp],
                "legal_actions": legal,
            }
            if can_raise:
                req["min_raise"] = min(bets[opp] + max(last_raise, 2), max_raise)
                req["max_raise"] = max_raise
            req["raises_left"] = MAX_RAISES - raises
            req["history"] = self.history_for(actor)

            action, amount = self.ask_action(actor, req)

            if action == "fold":
                self.history.append((street, actor, "fold", None))
                return actor
            if action == "check":
                self.history.append((street, actor, "check", None))
            elif action == "call":
                self.stacks[actor] -= to_call
                self.committed[actor] += to_call
                bets[actor] = bets[opp]
                self.history.append((street, actor, "call", bets[actor]))
            else:  # raise to `amount`
                add = amount - bets[actor]
                self.stacks[actor] -= add
                self.committed[actor] += add
                last_raise = amount - bets[opp]
                bets[actor] = amount
                raises += 1
                acted[opp] = False
                self.history.append((street, actor, "raise", amount))
            acted[actor] = True
            if acted[0] and acted[1] and bets[0] == bets[1]:
                return None
            actor = opp

    def discard(self):
        for seat in (0, 1):
            self.bots[seat].begin({
                "type": "discard_request", "hand_id": self.id,
                "hole_cards": list(self.hole[seat]), "board": self.board[:3],
                "pot": sum(self.committed),
                "my_stack": self.stacks[seat], "opp_stack": self.stacks[1 - seat],
            })
        replies, failures = [None, None], []
        for seat in (0, 1):  # both clocks run at the same time
            line, err = self.bots[seat].wait_reply(self.match.wall_deadline)
            if err:
                failures.append((self.bots[seat], err))
            replies[seat] = line
        if failures:
            raise Forfeit(failures)
        for seat in (0, 1):
            card, code = parse_discard(replies[seat], self.hole[seat])
            if code:
                card = default_discard(self.hole[seat])
                self.bots[seat].send({"type": "notice", "code": code, "applied": card})
                self.bots[seat].notices += 1
            self.discards[seat] = card
        for seat in (0, 1):
            self.hole[seat].remove(self.discards[seat])
        for seat in (0, 1):
            self.bots[seat].send({
                "type": "discard_result", "hand_id": self.id,
                "my_discard": self.discards[seat], "opp_discard": self.discards[1 - seat],
                "hole_cards": list(self.hole[seat]),
            })

    def finish(self, folder):
        if folder is not None:
            self.result = "fold"
            lost = self.committed[folder]
            nets = [0, 0]
            nets[folder], nets[1 - folder] = -lost, lost
        else:
            self.result = "showdown"
            self.board_seen = 5
            assert self.committed[0] == self.committed[1]
            self.scores = (best_hand(self.hole[0], self.board), best_hand(self.hole[1], self.board))
            amount = self.committed[0]
            if self.scores[0] > self.scores[1]:
                nets = [amount, -amount]
            elif self.scores[1] > self.scores[0]:
                nets = [-amount, amount]
            else:
                nets = [0, 0]  # split pot
        for seat in (0, 1):
            self.bots[seat].send({
                "type": "hand_result", "hand_id": self.id, "result": self.result,
                "net": nets[seat], "board": self.board[:self.board_seen],
                "my_hole_cards": list(self.hole[seat]),
                "opp_hole_cards": list(self.hole[1 - seat]) if self.result == "showdown" else None,
                "opp_discard": self.discards[1 - seat],
                "history": self.history_for(seat),
            })
        return nets


# ---------------------------------------------------------------------- match

class Match:
    def __init__(self, args):
        stage = STAGES[args.stage]
        self.hands = args.hands or stage["hands"]
        if self.hands <= 0 or self.hands % 2:
            sys.exit("--hands must be a positive even number (every deal is played twice)")
        self.per_leg = self.hands // LEGS  # deals per leg; every deal once per leg
        self.cap = stage["cap"]
        self.time_bank_ms = stage["time_bank_ms"]  # per leg
        self.seed = args.seed if args.seed is not None else str(random.SystemRandom().randrange(10 ** 9))
        self.verbose = args.verbose
        self.log_dir = args.log_dir
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        self.log = open(os.path.join(self.log_dir, "match.log"), "w", encoding="utf-8") if self.log_dir else None

        self.args = args
        self.leg_bank = math.inf if args.no_time_limits else self.time_bank_ms
        self.per_reply = math.inf if args.no_time_limits else PER_RESPONSE_MS
        self.leg_wall = math.inf if args.no_time_limits else stage["wallclock_s"]
        self.labels = (self._label(args.bot_a), self._label(args.bot_b))
        self.used = [0.0, 0.0]        # time used by A and B, summed over legs
        self.notice_count = [0, 0]    # notices to A and B, summed over legs
        self.bots = ()
        self.leg = 0
        self.wall_deadline = math.inf

    @staticmethod
    def _label(path):
        try:
            return os.path.relpath(os.path.abspath(path))
        except ValueError:
            return os.path.abspath(path)

    def launch(self, leg):
        """Start fresh processes for both bots. Nothing survives from the previous leg."""
        a = Bot("A", self.args.bot_a, self.args.python, self.log_dir,
                self.leg_bank, self.per_reply, f"A_leg{leg + 1}")
        try:
            b = Bot("B", self.args.bot_b, self.args.python, self.log_dir,
                    self.leg_bank, self.per_reply, f"B_leg{leg + 1}")
        except Exception:
            a.proc.kill()
            raise
        return (a, b)

    def write(self, text):
        if self.log:
            self.log.write(text + "\n")
        if self.verbose:
            print(text)

    def log_hand(self, hand, seat_nets):
        n = (hand.bots[0].name, hand.bots[1].name)
        abbrev = {"fold": "f", "check": "x", "call": "c", "raise": "r"}
        acts, street_now = [], None
        for street, seat, action, amount in hand.history:
            if street != street_now:
                acts.append(street + ":")
                street_now = street
            acts.append(f"{n[seat]}{abbrev[action]}{'' if amount is None else amount}")
        disc = f" | discards {n[0]}:{hand.discards[0]} {n[1]}:{hand.discards[1]}" if hand.discards[0] else ""
        shown = ""
        if hand.scores:
            shown = f" ({n[0]} {HAND_NAMES[hand.scores[0][0]]}, {n[1]} {HAND_NAMES[hand.scores[1][0]]})"
        self.write(
            f"L{self.leg + 1} #{hand.id:<4} button {n[0]} | {n[0]}: {' '.join(hand.dealt[0])}  "
            f"{n[1]}: {' '.join(hand.dealt[1])} | board {' '.join(hand.board[:hand.board_seen]) or '-'}"
            f"{disc} | {' '.join(acts)} | {hand.result}{shown} {n[0]} {seat_nets[0]:+d}, {n[1]} {seat_nets[1]:+d}"
        )

    def run(self):
        totals, played, forfeits = [0, 0], 0, []
        forfeit_at = None  # (leg, hand) where a forfeit happened
        for leg in range(LEGS):
            self.leg = leg
            self.bots = self.launch(leg)
            self.wall_deadline = time.monotonic() + self.leg_wall
            for bot in self.bots:
                bot.send({
                    "type": "init", "protocol_version": "1.0", "total_hands": self.per_leg,
                    "starting_stack": STARTING_STACK, "small_blind": SMALL_BLIND,
                    "big_blind": BIG_BLIND, "time_bank_ms": self.time_bank_ms,
                    "per_response_ms": PER_RESPONSE_MS,
                })
            hand_no = 0
            try:
                for deal in range(self.per_leg):
                    hand_no = deal + 1
                    deck = [r + s for r in RANKS for s in SUITS]
                    random.Random(f"{self.seed}:{deal}").shuffle(deck)  # same deal in both legs
                    # Leg 1: bot A holds deck[0:3] on even deals. Leg 2 flips every
                    # deal, so each bot gets the other set of hole cards, same board.
                    button = (deal + leg) % 2
                    hand = Hand(self, hand_no, button, (deck[0:3], deck[3:6]), deck[6:11])
                    seat_nets = hand.play()
                    totals[button] += seat_nets[0]
                    totals[1 - button] += seat_nets[1]
                    played += 1
                    self.log_hand(hand, seat_nets)
            except Forfeit as f:
                forfeits = f.failures
                forfeit_at = (leg + 1, hand_no)
            finally:
                failed = {bot.name for bot, _ in forfeits}
                for bot in self.bots:
                    if not forfeits:
                        reason = "complete"
                    elif bot.name in failed:
                        reason = "forfeit"
                    else:
                        reason = "opp_forfeit"
                    bot.send({"type": "match_end", "reason": reason})
                for k, bot in enumerate(self.bots):
                    bot.close()
                    self.used[k] += bot.time_used_s
                    self.notice_count[k] += bot.notices
                self.bots = ()
            if forfeits:  # a forfeit in either leg ends the match
                break
        self.summary(totals, played, forfeits, forfeit_at)
        if self.log:
            self.log.close()

    def summary(self, totals, played, forfeits, forfeit_at):
        lines = ["", f"Seed {self.seed} | {played} of {self.hands} hands played "
                     f"({LEGS} legs of {self.per_leg})"]
        for i, name in enumerate(("A", "B")):
            lines.append(
                f"  {name}  {self.labels[i]:<32} net {totals[i]:+7d}   "
                f"time used {self.used[i]:6.2f} s (bank {self.time_bank_ms / 1000:.0f} s per leg)   "
                f"notices {self.notice_count[i]}"
            )
        if not forfeits:
            if totals[0] == totals[1]:
                lines.append("Result: draw")
            else:
                winner = "A" if totals[0] > totals[1] else "B"
                lines.append(f"Result: {winner} wins")
            capped = max(-self.cap, min(self.cap, totals[0]))
            lines.append(f"Capped delta: A {capped:+d}, B {-capped:+d} (C = {self.cap})")
        else:
            where = f"leg {forfeit_at[0]}, hand {forfeit_at[1]}"
            if len(forfeits) == 2:
                for bot, reason in forfeits:
                    lines.append(f"{bot.name} forfeited at {where}: {reason}")
                lines.append("Result: draw (both forfeited in the same hand), delta 0 for both")
            else:
                bot, reason = forfeits[0]
                other = "B" if bot.name == "A" else "A"
                lines.append(f"{bot.name} forfeited at {where}: {reason}")
                lines.append(f"Result: {other} wins by forfeit; {other}'s delta is set to +{self.cap}")
        if self.log_dir:
            lines.append(f"Logs: {os.path.join(self.log_dir, 'match.log')}, "
                         f"A_leg1_stderr.txt, A_leg2_stderr.txt, B_leg1_stderr.txt, B_leg2_stderr.txt")
        text = "\n".join(lines)
        print(text)
        if self.log:
            self.log.write(text + "\n")


def main():
    p = argparse.ArgumentParser(description="Pokerbots local match runner (Discard Hold'em)")
    p.add_argument("bot_a", help="path to the first bot's bot.py")
    p.add_argument("bot_b", help="path to the second bot's bot.py")
    p.add_argument("--stage", type=int, choices=(1, 2), default=1,
                   help="match length and limits to use (default: 1)")
    p.add_argument("--hands", type=int,
                   help="total hands over both legs, even (default: 400 in stage 1, 1000 in stage 2)")
    p.add_argument("--seed", help="deal seed; the same seed gives the same deals (default: random)")
    p.add_argument("--log-dir", default="logs", help="folder for match.log and stderr files ('' to disable)")
    p.add_argument("--python", default=sys.executable, help="interpreter used to run the bots (default: this one)")
    p.add_argument("--verbose", action="store_true", help="also print every hand to the console")
    p.add_argument("--no-time-limits", action="store_true",
                   help="disable the time bank, 5 s limit and wall-clock kill (debugging only)")
    args = p.parse_args()
    for path in (args.bot_a, args.bot_b):
        if not os.path.isfile(path):
            sys.exit(f"bot not found: {path}")
    match = Match(args)
    try:
        match.run()
    except KeyboardInterrupt:
        for bot in match.bots:
            try:
                bot.proc.kill()
            except Exception:
                pass
        sys.exit("interrupted")


if __name__ == "__main__":
    main()