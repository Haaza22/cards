import random
import Card_Manip as CM
import Hearts
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time


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
        q_values = q_table[index]
        ordered_actions = [action_space[j] for j in np.argsort(q_values)]
        # Action index list! order them, 0 best, 51 worst

        for j in range(0, len(ordered_actions)):
            # Loop cards to find highest valid option
            if ordered_actions[j] in action_options:
                position_in_hand = action_options.index(ordered_actions[j])
                playing = current_hand.remove_card(options[position_in_hand])
                break
    else:
        playing = None
        print("Behaviour should be Q, R, L, Hi, He1, He2 or He3")
    return playing


def update_q_table(state, action, reward, next_state, learn_rate, discount_rate):
    state_index = get_state_index(state)
    next_state_index = get_state_index(next_state)

    # Get the index of the action in the action space
    action_index = action_space.index(action)

    # Update the Q-value using the Q-learning update rule
    q_table[state_index, action_index] += learn_rate * (
            reward + discount_rate * np.max(q_table[next_state_index]) - q_table[state_index, action_index])


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


def train_q():
    learn_rate = 0.1
    discount_rate = 0.9
    explore_element = 0.1
    training = True
    training_rounds = 10000
    current_training_round = 0
    behaviour = ['Q', 'Q', 'Q', 'Q']

    while training:
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
                else:
                    options = Hearts.calc_options(hands[current_player], current_turn_record[0])

                # Choose action
                if np.random.rand() < explore_element:
                    # Random action
                    position_in_hand = random.randint(0, len(options) - 1)
                    action_form = str(hands[current_player].cards[position_in_hand].suit) + str(
                        hands[current_player].cards[position_in_hand].val)
                    playing = hands[current_player].remove_card(options[position_in_hand])
                else:
                    # Optimal action
                    action_options = []
                    # Make options in form of 'H2' so can be checked agaianst actions
                    for option in options:
                        cur_card = hands[current_player].cards[option]
                        action_options.append(str(cur_card.suit) + str(cur_card.val))

                    # Get all q values
                    index = get_state_index(state[current_player])
                    q_values = q_table[index]
                    ordered_actions = [action_space[j] for j in np.argsort(q_values)]
                    # Action index list! order them, 0 best, 51 worst

                    for j in range(0, len(ordered_actions)):
                        # Loop cards to find highest valid option
                        if ordered_actions[j] in action_options:
                            position_in_hand = action_options.index(ordered_actions[j])
                            action_form = str(hands[current_player].cards[position_in_hand].suit) + str(
                                hands[current_player].cards[position_in_hand].val)
                            playing = hands[current_player].remove_card(options[position_in_hand])
                            break

                # Take action
                action[current_player] = action_form
                on_table, who_played, current_turn_record = Hearts.one_loop(current_player, starter, on_table,
                                                                            who_played,
                                                                            current_turn_record, playing)

                current_player = Hearts.next(current_player)
                starter = False

            # Round end
            won_cards, turn_record, current_player, played_cards = Hearts.round_over(who_played, on_table, won_cards,
                                                                                     turn_record,
                                                                                     current_turn_record, played_cards)

            new_score = Hearts.count_points(won_cards)
            for j in range(0, 4):
                ## Making next state on_table by blank is, a bit upsetting, but i cant think of a way around it
                # Check reward of new stats
                next_state = [0, 0]
                reward = calc_reward(cur_score, new_score, current_player, j)
                # Update Q-table
                update_q_table(state[j], action[j], reward, next_state, learn_rate, discount_rate)
            cur_score = new_score

        current_training_round = current_training_round + 1
        if current_training_round >= training_rounds:
            training = False


def calc_reward(pre_points, cur_points, start_player, player_evaluated):
    reward = 0
    if pre_points <= cur_points:
        # If didn't gain points gain rewards
        reward = reward + 1
        if start_player == player_evaluated:
            # If in control and didn't gain points gain reward
            reward = reward + 1
    else:
        # If gained poitns loose reward, points euqal to score gained
        reward = reward - (cur_points - pre_points)
    return reward


def make_game(behaviour):
    hands, current_player = Hearts.setup(behaviour)
    won_cards = [[], [], [], []]
    # an array of 6 items, [lead suit, lead player, what player 1 played, what player 2 played, what player 3 played, what player 4 played,]
    turn_record = []
    played_cards = []
    current_scores = [0, 0, 0, 0]

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
            else:
                options = Hearts.calc_options(hands[current_player], current_turn_record[0])

            playing = choose_card_base(behaviour[current_player], hands[current_player], options, state[current_player])

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

# Trying statspace of cards on table, previously played and current score
# So that's one 52 size bool, another 52 size bool, and a number between 0 and 26, so a 27. so 73008 states...
state_space, num_states = initialize_state_space()
num_actions = len(action_space)
q_table = np.zeros((num_states, num_actions))
# print(q_table)

training_route = True

print("Train start")
start_time = time.time()
train_q()
end_time = time.time()
train_time = end_time - start_time
print("Trained in", train_time, "seconds")

for game in range(0, 1000):
    b_ops = ["R", "L", "Hi", "He1", "He2", "He3", "Q"]
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
print("Run time:", run_time, "seconds")
print("Train time:", train_time, "seconds")
print("Fitness function:", CM.fitness(percent_win, train_time, run_time))  # fittness is: 47.548330248657024

# Plotting graph
placements = ["1", "2", "3", "4"]
scores = {
    'Random': scoring[0],
    'Lowest': scoring[1],
    'Highest': scoring[2],
    'Random avoid hearts': scoring[3],
    'Lowest avoid hearts': scoring[4],
    'Highest avoid hearts': scoring[5],
    'Q learning': scoring[6],
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
