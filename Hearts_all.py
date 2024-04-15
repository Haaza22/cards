# idea to get all agents play against each other

import random
import Card_Manip as CM
import Hearts
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time
import Hearts_Bay as BAY
import Hearts_NN as NN
import Hearts_Q as QL


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
    start_time = time.time()
    choose_card_base("Q", handt1, [0, 1, 2, 3, 4, 5, 6], [2, 7])
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
    elif behaviour == 'Q':
        # Optimal action
        action_options = []
        # Make options in form of 'H2' so can be checked agaianst actions
        for option in options:
            cur_card = current_hand.cards[option]
            action_options.append(str(cur_card.suit) + str(cur_card.val))

        # Get all q values
        index = get_state_index(state)
        q_values = QL.q_table[index]
        ordered_actions = [action_space[j] for j in np.argsort(q_values)]
        # Action index list! order them, 0 best, 51 worst

        for j in range(0, len(ordered_actions)):
            # Loop cards to find highest valid option
            if ordered_actions[j] in action_options:
                position_in_hand = action_options.index(ordered_actions[j])
                playing = current_hand.remove_card(options[position_in_hand])
                break
    elif behaviour == "NN":
        playing, actions_, valid_actions_ = NN.agent.select_action(state, options, current_hand)
    else:
        playing = None
        print("Behaviour should be NN, B, Q, R, L, Hi, He1, He2 or He3")
    return playing


def initialize_state_space():
    # statspace is cards on table(52 bool), previously played (52 bool) and current score (27)
    hearts_on_table_vals = list(range(4))  # 0-3, only numbers of possible hearts on table
    curr_win_val = list(
        range(53))  # 0-52, each number being a card (0 as no cards on table,1 at 2 of clubs, 51 as ace of spades)
    # player_scores_val = list(range(27))  # Possible scores from 0 to max_score (26)

    state_space = list(itertools.product(hearts_on_table_vals, curr_win_val))

    return state_space, len(state_space)


def get_state_index(state):
    # Convert the state tuple to a hashable representation
    index = 53 * state[0] + state[1]
    return index


def count_hearts(table):
    hearts = 0
    for card in table:
        if card.suit == 'H':
            hearts = hearts + 1
    return hearts


def make_game(behaviour):
    hands, current_player = Hearts.setup(behaviour)
    won_cards = [[], [], [], []]
    # an array of 6 items, [lead suit, lead player, what player 1 played, what player 2 played, what player 3 played, what player 4 played,]
    turn_record = []
    played_cards = []
    current_scores = [0, 0, 0, 0]
    belief_state = BAY.BeliefState(4)

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

            if behaviour[current_player] != "B":
                playing = choose_card_base(behaviour[current_player], hands[current_player], options, state[current_player])
            else:
                playing = belief_state.choose_card(hands[current_player], options, following)

            on_table, who_played, current_turn_record = Hearts.one_loop(current_player, starter, on_table, who_played,
                                                                        current_turn_record, playing)

            if following:
                belief_state.update_card_played(CM.card_number(str(playing.suit) + str(playing.val)), current_player)
            else:
                belief_state.update_void(on_table[0].suit, current_player)

            current_scores = Hearts.count_points(won_cards)
            current_player = Hearts.next(current_player)
            starter = False

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
scoring = np.zeros((9, 4))
games_played = [0, 0, 0, 0, 0, 0, 0, 0, 0]

action_space = []
# initialise Q learning
# Define action space
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
suits = ['H', 'D', 'C', 'S']
for val in ranks:
    for suit in suits:
        action_space.append(str(suit) + str(val))

for game in range(0, 1000):
    b_ops = ["R", "L", "Hi", "He1", "He2", "He3", "Q", "NN", "B"]
    behaviour_pre = []
    behaviour_num = []

    # select random behaviours
    for i in range(0, 4):

        # chosen = 7
        # while chosen == 7:
        #     chosen = random.randint(0, 8)

        chosen = random.randint(0, 8)
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
run_time = QL.run_time_calc()
print("Fitness function Q:", CM.fitness(percent_win, QL.train_time, run_time))  #
win_anal = scoring[7][0] + scoring[7][1] / 2
percent_win = (win_anal / games_played[7]) * 100
run_time = NN.run_time_calc()
print("Fitness function NN:", CM.fitness(percent_win, NN.train_time, run_time))
win_anal = scoring[8][0] + scoring[8][1] / 2
percent_win = (win_anal / games_played[8]) * 100
run_time = BAY.run_time_calc()
print("Fitness function BAY:", CM.fitness(percent_win, BAY.train_time, run_time))


# plotting
placements = ["1", "2", "3", "4"]
scores = {
    'Random': scoring[0],
    'Lowest': scoring[1],
    'Highest': scoring[2],
    'Random avoid hearts': scoring[3],
    'Lowest avoid hearts': scoring[4],
    'Highest avoid hearts': scoring[5],
    'Q learning': scoring[6],
    'Neural Network': scoring[7],
    'Bayesian Learning': scoring[8],
}

# Calculate the number of groups and categories
n_groups = len(placements)
n_categories = len(scores)

# Define narrower bar width to prevent overlapping
bar_width = 0.9  # Adjust based on desired spacing

# Create an index for each category within each group
index = np.arange(n_groups) * n_categories  # Creates an array like [0, 1, 2, 3, ...]

fig, ax = plt.subplots()

# Loop through categories and plot bars with specific offsets
for i, (attribute, measurement) in enumerate(scores.items()):
    rects = ax.bar(index + i * bar_width, measurement, width=bar_width, label=attribute)
    ax.bar_label(rects, padding=8)

# Improve readability
ax.set_ylabel('Games in this position', fontsize=12)
ax.set_title('Average placements', fontsize=14)

# Set x-axis ticks and labels at correct positions
ax.set_xticks(index + n_categories / 2, placements)
ax.set_xlabel('Strategy', fontsize=12)

# Rotate x-axis labels for better readability if many categories
if n_categories > 4:
    plt.xticks(rotation=45)  # Adjust rotation angle if needed

# Add legend and grid lines
ax.legend(loc='upper left', fontsize=8, ncol=4)
ax.grid(axis='y')

# Adjust plot borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()
