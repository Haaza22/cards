import random
import Card_Manip as CM
import Hearts
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time
from collections import defaultdict


# import keras
# import tensorflow
# from keras.models import Sequential
# from keras.layers import Dense
def run_time_calc():
    deckt = CM.Deck()
    handt1 = CM.Hand()
    handt2 = CM.Hand()
    handt3 = CM.Hand()
    handt4 = CM.Hand()
    CM.deal_deck(deckt, handt1, handt2, handt3, handt4, 13)
    belief_state = BeliefState(4)
    start_time = time.time()
    belief_state.choose_card(handt1, [0, 1, 2, 3, 4, 5, 6], True)
    end_time = time.time()
    run_time = end_time - start_time
    return run_time


def win_order(points):
    # returns an array with the placment od 1st 2nd and so on in the index of the behaviour that got that place

    # count how many other entries are higher or equal than item. then minue form 5
    positions = []
    for i in range(0, 4):
        more = 0
        for e in range(0, 4):
            if points[e] >= points[i]:
                more = more + 1
        positions.append(5 - more)

    return positions


def choose_card_base(behaviour, current_hand, options, state):
    if behaviour == "R":
        playing = current_hand.remove_card(options[random.randint(0, len(options) - 1)])

    elif behaviour == "L":
        low = 15
        pos = ""
        for i in range(0, len(options)):
            if current_hand.cards[options[i]].val < low:
                low = current_hand.cards[options[i]].val
                pos = options[i]
        playing = current_hand.remove_card(pos)

    elif behaviour == "Hi":
        top = -1
        pos = ""
        for i in range(0, len(options)):
            if current_hand.cards[options[i]].val > top:
                top = current_hand.cards[options[i]].val
                pos = options[i]
        playing = current_hand.remove_card(pos)

    # avoid hearts, then random, highest or lowest using new method
    elif behaviour == "He1" or behaviour == "He2" or behaviour == "He3":
        new_options = []

        for i in range(0, len(options)):
            if current_hand.cards[options[i]].suit != "H":
                new_options.append(options[i])

        if len(new_options) == 0:
            new_options = options

        if behaviour == "He1":
            playing = choose_card_base("R", current_hand, new_options, state)
        elif behaviour == "He2":
            playing = choose_card_base("L", current_hand, new_options, state)
        else:
            playing = choose_card_base("Hi", current_hand, new_options, state)
    else:
        playing = None
        print("Behaviour should be Q, R, L, Hi, He1, He2 or He3")
    return playing


def count_hearts(table):
    hearts = 0
    for card in table:
        if card.suit == 'H':
            hearts = hearts + 1
    return hearts


class BeliefState:
    def __init__(self, num_players):
        self.num_players = num_players
        self.deck = set(range(52))  # Represent cards as integers (0-51)
        self.beliefs = defaultdict(float)  # Key: (player_id, card), Value: probability

    def update_card_played(self, card, player_id):
        self.deck.remove(card)
        if (player_id, card) in self.beliefs:
            del self.beliefs[(player_id, card)]  # Remove played card from beliefs

    def update_void(self, suit_given, player_id):
        if suit_given == 'C':
            lower = 0
            upper = 12
        elif suit_given == 'D':
            lower = 13
            upper = 25
        elif suit_given == 'H':
            lower = 26
            upper = 38
        else:  # Spade
            lower = 29
            upper = 51

        for player in range(self.num_players):
            if player != player_id:
                for card in self.deck:
                    if lower <= card <= upper:  # if suit then remove it
                        if (player, card) in self.beliefs:
                            del self.beliefs[(player, card)]

    def choose_card(self, hand_class, options, following):
        # Prioritize following suit if possible
        hand = hand_class.cards

        if following:
            # start by calculating the firsty cards score
            best_score = self.evaluate_card(hand[options[0]])
            best_pos = 0
            for i in range(1, len(options)):
                cur_score = self.evaluate_card(hand[options[i]])
                if cur_score >= best_score:
                    best_score = cur_score
                    best_pos = i
            return hand_class.remove_card(options[best_pos])
        else:
            best_heart = [-1, -1]
            other = [-1, -1]
            for i in range(0, len(options)):
                if hand[options[i]].suit == "S" and hand[options[i]].val == 12:
                    return hand_class.remove_card(options[i])
                elif hand[options[i]].suit == "H" and hand[options[i]].val > best_heart[0]:
                    best_heart[0] = hand[options[i]].val
                    best_heart[1] = i
                else:
                    if hand[options[i]].val > other[0]:
                        other[0] = hand[options[i]].val
                        other[1] = i
            if best_heart[0] != [-1]:
                return hand_class.remove_card(options[best_heart[1]])
            return hand_class.remove_card(options[other[1]])

    def evaluate_card(self, card):
        score = 0
        if card.suit == 'H':
            score = score + self.calculate_heart_capture_risk(card)
        elif card.suit == 'S' and card.val == 12:  # Queen of Spades
            score = score + self.calculate_queen_capture_risk()
        return score

    def calculate_heart_capture_risk(self, card):
        capture_risk = 0
        for unseen_card in self.beliefs:
            if unseen_card != card and unseen_card.suit == 'H':
                # Consider the probability of the unseen card being a heart
                capture_risk = capture_risk + self.beliefs[unseen_card]
        if card.suit == 'H':
            # if you play a high heart, youre likely to score
            capture_risk = capture_risk * (1 - card.val / 13)
        return capture_risk

    def calculate_queen_capture_risk(self):
        # need to see if king and ace are left
        return 0


def train_bayes():
    learn_rate = 0.1
    discount_rate = 0.9
    explore_element = 0.1
    training = True
    training_rounds = 10000
    current_training_round = 0
    behaviour = ['B', 'B', 'B', 'B']

    while training:
        belief_state = BeliefState(4)
        # Heart setup
        played_cards = []
        hands, current_player = Hearts.setup(behaviour)
        won_cards = [[], [], [], []]
        turn_record = []
        cur_score = [0, 0, 0, 0]

        # Q setup
        # Statespace: hearts on table, current winning card
        state = [[], [], [], []]
        for i in range(0, 4):
            state[i] = [0, 0]

        # Round setup
        for i in range(0, 13):
            # Hearts setup
            starter = True
            current_turn_record = ["", 0, 0, 0, 0, 0]
            who_played = []
            on_table = []
            # Q setup
            action = [0, 0, 0, 0]
            # Players actions
            for e in range(0, 4):
                # get current state for player
                state[current_player][0] = count_hearts(on_table)
                if len(on_table) != 0:
                    state[current_player][1] = CM.card_number(Hearts.find_wining_card(on_table))
                else:
                    state[current_player][1] = 0

                # Find possible to play cards
                # Options is a list contianing all the positions in the hand which are valid cards to play
                if starter:
                    options = list(range(0, len(hands[current_player].cards)))
                    following = True
                else:
                    options, following = Hearts.calc_options_B(hands[current_player], current_turn_record[0])

                playing = belief_state.choose_card(hands[current_player], options, following)

                # Take action
                action[current_player] = str(playing.suit) + str(playing.val)
                on_table, who_played, current_turn_record = Hearts.one_loop(current_player, starter, on_table,
                                                                            who_played,
                                                                            current_turn_record, playing)

                # Bayes update
                if following:
                    belief_state.update_card_played(CM.card_number(action[current_player]), current_player)
                else:
                    belief_state.update_void(on_table[0].suit, current_player)

                current_player = Hearts.next(current_player)
                starter = False

            # Round end
            won_cards, turn_record, current_player, played_cards = Hearts.round_over(who_played, on_table, won_cards,
                                                                                     turn_record,
                                                                                     current_turn_record, played_cards)

            new_score = Hearts.count_points(won_cards)
            cur_score = new_score

        current_training_round = current_training_round + 1
        if current_training_round >= training_rounds:
            training = False


def make_game(behaviour):
    hands, current_player = Hearts.setup(behaviour)
    won_cards = [[], [], [], []]
    # an array of 6 items, [lead suit, lead player, what player 1 played, what player 2 played, what player 3 played, what player 4 played,]
    turn_record = []
    played_cards = []
    current_scores = [0, 0, 0, 0]
    belief_state = BeliefState(4)

    state = [[0, 0], [0, 0], [0, 0], [0, 0]]
    for i in range(0, 13):
        starter = True
        current_turn_record = ["", 0, 0, 0, 0, 0]
        who_played = []
        on_table = []

        for e in range(0, 4):
            # get current state for player
            state[current_player][0] = count_hearts(on_table)
            if len(on_table) != 0:
                state[current_player][1] = CM.card_number(Hearts.find_wining_card(on_table))
            else:
                state[current_player][1] = 0
            # Options is a list contianing all the positions in the hand which are valid cards to play
            if starter:
                options = list(range(0, len(hands[current_player].cards)))
                following = True
            else:
                options, following = Hearts.calc_options_B(hands[current_player], current_turn_record[0])

            if behaviour == "B":
                playing = choose_card_base(behaviour[current_player], hands[current_player], options,
                                           state[current_player])
            else:
                playing = belief_state.choose_card(hands[current_player], options, following)

            on_table, who_played, current_turn_record = Hearts.one_loop(current_player, starter, on_table, who_played,
                                                                        current_turn_record, playing)

            if following:
                belief_state.update_card_played(CM.card_number(str(playing.suit) + str(playing.val)), current_player)
            else:
                belief_state.update_void(on_table[0].suit, current_player)

            starter = False
            current_player = Hearts.next(current_player)
            current_scores = Hearts.count_points(won_cards)

        won_cards, turn_record, current_player, played_cards = Hearts.round_over(who_played, on_table, won_cards,
                                                                                 turn_record,
                                                                                 current_turn_record, played_cards)

    points = Hearts.count_points(won_cards)
    positions = win_order(points)

    return points, positions


def scoring_analysis(scores, games_played):
    # Takes the scores and gives a total fitnees score relating to the other games
    # 5 poitns for first, 3 for 2nd 1 for 3rd and none for last
    totals = [0, 0, 0, 0, 0, 0, 0]
    for i in range(0, 7):
        totals[i] = (totals[i] + 5 * scores[i][0] + 3 * scores[i][1] + scores[i][2]) / games_played[i]
    return totals


# Initialise score tracking
scoring = np.zeros((7, 4))
games_played = [0, 0, 0, 0, 0, 0, 0]

action_space = []
# initialise Q learning
# Define action space
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
suits = ['H', 'D', 'C', 'S']
for val in ranks:
    for suit in suits:
        action_space.append(str(suit) + str(val))
# for action in action_space:
#     print(action)

training_route = True

print("Train start")
start_time = time.time()
train_bayes()
end_time = time.time()
train_time = end_time - start_time
print("Trained in", train_time, "seconds")

for game in range(0, 1000):
    b_ops = ["R", "L", "Hi", "He1", "He2", "He3", "B"]
    behaviour_pre = []
    behaviour_num = []

    # select random behaviours
    for i in range(0, 4):
        chosen = random.randint(0, 6)
        behaviour_pre.append(b_ops[chosen])
        behaviour_num.append(chosen)

    # Set behaviour tests
    # behaviour_pre = ["Hi", "R", "He1", "Q"]
    # behaviour_num = [2, 0, 3, 6]

    points, positions = make_game(behaviour_pre)

    # increase the score of the behaviour by 1 in its position, positions needs to be -1 becayse its 1-4 and not 0-3
    for i in range(0, 4):
        scoring[behaviour_num[i]][positions[i] - 1] = scoring[behaviour_num[i]][positions[i] - 1] + 1
        # increase how many games its in
        games_played[behaviour_num[i]] = games_played[behaviour_num[i]] + 1

print(scoring)
print(scoring_analysis(scoring, games_played))
win_anal = scoring[6][0] + scoring[6][1] / 2
percent_win = (win_anal / games_played[6]) * 100
run_time = run_time_calc()
print("Fitness function:", CM.fitness(percent_win, train_time, run_time))  # Fitness is: 43.6974341419476

# Plotting graph
placements = ["1", "2", "3", "4"]
scores = {
    'Random': scoring[0],
    'Lowest': scoring[1],
    'Highest': scoring[2],
    'Random avoid hearts': scoring[3],
    'Lowest avoid hearts': scoring[4],
    'Highest avoid hearts': scoring[5],
    'Genetic': scoring[6],
}
x = np.arange(len(placements))  # the label locations
width = 0.12  # the width of the bars

fig, ax = plt.subplots()
offset = 0

for attribute, measurement in scores.items():
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=6)
    offset += width  # Increase the offset for the next group of bars

# Improve readability
ax.set_ylabel('Games in this position', fontsize=12)
ax.set_title('Average placements', fontsize=14)
ax.set_xticks(x + width * 3, placements)
ax.legend(loc='upper left', fontsize=8, ncol=4)

# Add grid lines
ax.grid(axis='y')

# Adjust plot borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()
