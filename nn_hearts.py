import random
import Card_Manip
import Hearts
import numpy as np
import itertools
import matplotlib.pyplot as plt


# import keras
# import tensorflow
# from keras.models import Sequential
# from keras.layers import Dense


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


def choose_card_base(behaviour, current_hand, options, on_table, played_cards):
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
            playing = choose_card_base("R", current_hand, new_options, on_table, played_cards)
        elif behaviour == "He2":
            playing = choose_card_base("L", current_hand, new_options, on_table, played_cards)
        else:
            playing = choose_card_base("Hi", current_hand, new_options, on_table, played_cards)
    # elif behaviour == "Q":
    #     return q_learn(current_hand, options, on_table)
    else:
        playing = None
        print("Behaviour should be Q, R, L, Hi, He1, He2 or He3")
    return playing


def update_q_table(state, action, reward, next_state, learn_rate, discount_rate):
    state_index = get_state_index(state_space, state)
    next_state_index = get_state_index(state_space, next_state)

    # Get the index of the action in the action space
    action_index = action_space.index(action)

    # Update the Q-value using the Q-learning update rule
    q_table[state_index, action_index] += learn_rate * (
            reward + discount_rate * np.max(q_table[next_state_index]) - q_table[state_index, action_index])


def initialize_state_space():
    # statspace is cards on table(52 bool), previously played (52 bool) and current score (27)
    on_table_vals = [0, 1]  # 0: Card not in hand, 1: Card in hand
    prev_played_val = [0, 1]  # 0: Card not played, 1: Card played
    player_scores_val = list(range(27))  # Possible scores from 0 to max_score

    # Generate all possible combinations of player's hand, trick history, and player scores
    on_table_combo = list(itertools.product(on_table_vals, repeat=52))
    prev_play_combo = list(itertools.product(prev_played_val, repeat=52))
    player_scores_combo = list(itertools.product(player_scores_val, repeat=4))

    # Cartesian product of player's hand, trick history, and player scores
    state_space = list(itertools.product(on_table_combo, prev_play_combo, player_scores_combo))

    return state_space

def get_state_index(state_space, state):
    return state_space.index(state)


def train_q():
    learn_rate = 0.1
    discount_rate = 0.9
    explore_element = 0.1
    training = True
    training_rounds = 100
    current_training_round = 0
    behaviour=['Q', 'Q', 'Q', 'Q']

    while training:
        print(current_training_round)
        # Heart setup
        played_cards = []
        hands, current_player = Hearts.setup(behaviour)
        won_cards = [[], [], [], []]

        # Q setup
        # Statespace: hand played card, on table and current score
        # Trying statspace of cards on table, previously played and current score
        state = [[], [], [], []]
        for i in range(0, 4):
            state[i] = [[], [], 0]

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
                # Set the state space
                state[current_player][0] = on_table
                state[current_player][1] = played_cards
                state[current_player][3] = Hearts.count_points(won_cards)[current_player]

                # Find possible to play cards
                # Options is a list contianing all the positions in the hand which are valid cards to play
                if starter:
                    options = list(range(0, len(hands[current_player].cards)))
                else:
                    options = Hearts.calc_options(hands[current_player], current_turn_record[0])

                # Choose action
                if np.random.rand() < explore_element:
                    # Random action
                    playing = hands[current_player].remove_card(options[random.randint(0, len(options))])
                else:
                    # Optimal action
                    action_options = []
                    # Make options in form of 'H2' so can be checked agaianst actions
                    for option in options:
                        cur_card = hands[current_player].cards[option]
                        action_options.append(itertools.product(cur_card.suit, cur_card.val))

                    # Get all q values
                    q_values = q_table[get_state_index(state)]
                    ordered_actions = [action_space[i] for j in np.argsort(q_values)]
                    # Action index list! order them, 0 best, 51 worst

                    for j in range(0, len(ordered_actions)):
                        # Loop cards to find highest valid option
                        if ordered_actions[j] in action_options:
                            playing = hands[current_player].remove_card(options[j])
                            break

                # Take action
                action[current_player] = playing
                on_table, who_played, current_turn_record = Hearts.one_loop(current_player, starter, on_table,
                                                                            who_played,
                                                                            current_turn_record, playing)

                current_player = Hearts.next(current_player)
                starter = False

            # Round end
            won_cards, turn_record, current_player, played_cards = Hearts.round_over(who_played, on_table, won_cards,
                                                                                     turn_record,
                                                                                     current_turn_record, played_cards)
            for j in range(0, 4):
                ## Making next state on_table by blank is, a bit upsetting, but i cant think of a way around it
                # Check reward of new stats
                next_state = [[], played_cards, Hearts.count_points(won_cards)[j]]
                reward = calc_reward(state[j], next_state[j], current_player, j)
                # Update Q-table
                update_q_table(state[j], action[j], reward, next_state, learn_rate, discount_rate)
                # Update state
                state[j] = next_state[j]

        current_training_round = current_training_round + 1
        print(current_training_round)
        if current_training_round >= training_rounds:
            training = False

    print(q_table)


def calc_reward(state, next_state, start_player, player_evaluated):
    reward = 0
    if state[2] <= next_state[2]:
        # If didn't gain points gain rewards
        reward = reward + 1
        if start_player == player_evaluated:
            # If in control and didn't gain points gain reward
            reward = reward + 1
    else:
        # If gained poitns loose reward, points euqal to score gained
        reward = reward - (next_state[i][2] - state[i][2])
    return reward


def make_game(behaviour):
    hands, current_player = Hearts.setup(behaviour)
    won_cards = [[], [], [], []]
    # an array of 6 items, [lead suit, lead player, what player 1 played, what player 2 played, what player 3 played, what player 4 played,]
    turn_record = []
    played_cards = []
    current_scores = [0, 0, 0, 0]
    for i in range(0, 13):
        starter = True
        current_turn_record = ["", 0, 0, 0, 0, 0]
        who_played = []
        on_table = []

        for e in range(0, 4):
            # Options is a list contianing all the positions in the hand which are valid cards to play
            if starter:
                options = list(range(0, len(hands[current_player].cards)))
            else:
                options = Hearts.calc_options(hands[current_player], current_turn_record[0])
            playing = choose_card_base(behaviour[current_player], hands[current_player], options, on_table,
                                       played_cards)

            on_table, who_played, current_turn_record = Hearts.one_loop(current_player, starter, on_table, who_played,
                                                                        current_turn_record, playing)

            current_scores = Hearts.count_points(won_cards)
            current_player = Hearts.next(current_player)
            starter = False

        won_cards, turn_record, current_player, played_cards = Hearts.round_over(who_played, on_table, won_cards,
                                                                                 turn_record,
                                                                                 current_turn_record, played_cards)
    points = Hearts.count_points(won_cards)
    positions = win_order(points)

    return points, positions


# Initialise score tracking
scoring = np.zeros((4, 4))
# initialise Q learning
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
suits = ['H', 'D', 'C', 'S']
action_space = list(itertools.product(suits, ranks))
state_space = initialize_state_space()
# for action in action_space:
#     print(action)

# Trying statspace of cards on table, previously played and current score
# So that's one 52 size bool, another 52 size bool, and a number between 0 and 26, so a 27. so 73008 states...
num_states = 73008
num_actions = len(action_space)
q_table = np.zeros((num_states, num_actions))
# print(q_table)

training_route = False

if training_route:
    train_q()
else:
    for game in range(0, 100):
        b_ops = ["R", "L", "Hi", "He1", "He2", "He3", "Q"]
        behaviour_pre = []
        behaviour_num = []

        # select random behaviours
        for i in range(0, 4):
            chosen = random.randint(0, 5)
            behaviour_pre.append(b_ops[chosen])
            behaviour_num.append(chosen)

        # Set behaviour tests
        behaviour_pre = ["Hi", "R", "He1", "L"]
        behaviour_num = [2, 0, 3, 1]

        points, positions = make_game(behaviour_pre)

        # increase the score of the behaviour by 1 in its position, positions needs to be -1 becayse its 1-4 and not 0-3
        for i in range(0, 4):
            scoring[behaviour_num[i], positions[i] - 1] = scoring[behaviour_num[i], positions[i] - 1] + 1

    print(scoring)

# Plotting graph
# placments=["1","2","3","4"]
# scores={
#     'Random': scoring[0],
#     'Lowest': scoring[1],
#     'Highest': scoring[2],
#     'Random avoid hearts': scoring[3],
#     'Lowest avoid hearts': scoring[4],
#     'Highest avoid hearts': scoring[5],
# }
# x = np.arange(len(placments))  # the label locations
# width = 0.125  # the width of the bars
# multiplier = 0
# fig, ax = plt.subplots(layout='constrained')
# for attribute, measurement in scores.items():
#     offset = width * multiplier * 2
#     rects = ax.bar(x + offset, measurement, width, label=attribute)
#     ax.bar_label(rects, padding=6)
#     multiplier += 1
# ax.set_ylabel('Games in this positions')
# ax.set_title('Avergae placments')
# ax.set_xticks(x + width, placments)
# ax.legend(loc='upper left', ncols=6)
# plt.show()
