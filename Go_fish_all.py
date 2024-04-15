import random
import Card_Manip as CM
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
from collections import defaultdict
import Go_Fish_Bay as BAY
import Go_fish_NN as NN
import Go_fish_Q as QL


def advance_turn(turn, in_game):
    if turn == 3:
        next_turn = 0
    else:
        next_turn = turn + 1
    return next_turn


def find_values(cards):
    values = []
    for card in cards:
        values.append(card.val)
    values_unique = list(dict.fromkeys(values))
    return values_unique


def valid_opponents(in_game, turn):
    opponents = []
    for i in range(0, 4):
        if in_game[i] and i != turn:
            opponents.append(i)
    return opponents


def guess(target_hand, player_hand, target_val):
    if target_val in find_values(target_hand.cards):

        values = []
        for i in range(0, len(target_hand.cards)):
            if target_hand.cards[i].val == target_val:
                values.append(i)

        for position in values:
            player_hand.add_card(target_hand.cards[position])
        looping = True
        i = 0
        while looping:
            if target_hand.cards[i].val == target_val:
                target_hand.remove_card(i)
            else:
                i = i + 1
            if i >= target_hand.size:
                looping = False

        return target_hand, player_hand, True
    return target_hand, player_hand, False


def set_check(cards):
    is_in = []
    amount = []
    for card in cards:
        if card.val not in is_in:
            is_in.append(card.val)
            amount.append(1)
        else:
            place = is_in.index(card.val)
            amount[place] = amount[place] + 1
    for i in range(0, len(amount)):
        if amount[i] == 4:
            return is_in[i]
    return -1


def remove_set(hand, remove):
    looping = True
    i = 0
    while looping:
        if hand.cards[i].val == remove:
            hand.remove_card(i)
        else:
            i = i + 1
        if i >= hand.size:
            looping = False
    return hand


def main_game(behaviour):
    # Base game needs
    playing = True
    start_cards = 5
    turn = random.randint(0, 3)
    players = 4
    in_game = [True, True, True, True]
    books = [0, 0, 0, 0]
    what_booked = []
    pond_exists = True
    states = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

    # Each item a 5 item array [whos turn it is, target, card asked, successful, if draw from deck is successdul?]
    turn_record = []

    # setting starting card logic
    deck = CM.Deck()
    hand1 = CM.Hand()
    hand2 = CM.Hand()
    hand3 = CM.Hand()
    hand4 = CM.Hand()
    deck, hand1, hand2, hand3, hand4 = CM.deal_deck(deck, hand1, hand2, hand3, hand4, start_cards)
    hands = [hand1, hand2, hand3, hand4]

    belief_state = BAY.BeliefState(4)

    while playing:
        extra_turn = False
        current_turn_record = [turn, -1, -1, False, False]
        # get right hand
        current_hand = hands[turn]

        alive = 0
        for player in in_game:
            if player:
                alive = alive + 1

        # make sure can get back into game
        if ((current_hand.size == 0 and pond_exists) or current_hand.size > 0) and alive > 1:
            if current_hand.size == 0 and pond_exists:
                current_hand.add_card(deck.draw_card())
                belief_state.update_fish(turn)
                in_game[turn] = True
                if deck.size == 0:
                    pond_exists = False

            # getting cards can ask for and opponents can ask
            # Options is between 2 and 14, a number
            options = find_values(current_hand.cards)
            opponents_main = valid_opponents(in_game, turn)

            # Make guess
            if behaviour[turn] == "BB":
                if np.random.rand() < 0.05:
                    guess_val, target = belief_state.choose_card(current_hand, options, turn)
                else:
                    guess_val, target = choose_action("RR", options, opponents_main, current_hand, hands, states[turn],
                                                      turn)
            else:
                guess_val, target = choose_action(behaviour[turn], options, opponents_main, current_hand, hands,
                                                  states[turn], turn)

            # Setup
            target_hand = hands[int(target)]
            current_turn_record[1] = target
            current_turn_record[2] = guess_val

            # Do guess
            target_hand, current_hand, correct = guess(target_hand, current_hand, guess_val + 2)

            belief_state.update_card_played(guess_val, turn, target)

            if correct:
                extra_turn = True
                current_turn_record[3] = True
            elif pond_exists:
                # Go fish
                new_card = deck.draw_card()
                current_hand.add_card(new_card)
                belief_state.update_fish(turn)
                if deck.size == 0:
                    pond_exists = False
                if new_card.val == guess_val:
                    extra_turn = True
                    current_turn_record[4] = True

            # check if targets last card was taken
            if target_hand.size == 0:
                in_game[int(target)] = False

            # check to see if game is over
            alive = 0
            for player in in_game:
                if player:
                    alive = alive + 1

        elif alive == 1 and pond_exists:
            # if no one else is a live but the pond eixsts
            current_hand.add_card(deck.draw_card())
            if deck.size == 0:
                pond_exists = False

        # check for set
        set = set_check(current_hand.cards)
        if set != -1:
            # remove full set and mark point
            current_hand = remove_set(current_hand, set)
            books[turn] = books[turn] + 1
            what_booked.append(set)
            belief_state.update_book(set)

        if current_hand.size == 0:
            in_game[turn] = False

        hands[turn] = current_hand

        if alive > 1 or pond_exists:
            if not extra_turn:
                turn = advance_turn(turn, in_game)
        else:
            playing = False
        turn_record.append(current_turn_record)
    return books


def count_vals_in_hand(hand):
    base = np.zeros(13)
    for card in hand:
        base[card.val - 2] = base[card.val - 2] + 1
    return base


def choose_action(behaviour, options, opponents, current_hand, hands, state, turn):
    if behaviour == "QQ" and np.random.rand() < 0.05:
        # Optimal action
        action_options = []
        # Make options in form of '22' so can be checked agaianst action space

        # PROBLEM IS OPPONENT CHOSEN ISNT ALEAYS CORREST
        # IT CHOOSES OPPOENENT 3 WHEN IT SHOULDNT
        for opponent in opponents:
            for val in options:
                action_temp = opponent
                if int(turn) < int(opponent):
                    action_temp = int(action_temp) - 1
                action_options.append(str(action_temp) + str(val))

        # Get all q values
        index = QL.get_state_index(state)
        q_values = QL.q_table[index]
        ordered_actions = [action_space[j] for j in np.argsort(q_values)]

        for j in range(0, len(ordered_actions)):
            # Loop cards to find highest valid option
            if ordered_actions[j] in action_options:
                target_guessing = opponents[ordered_actions[j][0].index(ordered_actions[j][0])]
                if len(ordered_actions[j]) == 2:
                    guessing_val = ordered_actions[j][1]
                else:
                    guessing_val = ordered_actions[j][1:]
                guessing_val = int(guessing_val) + 2
                break
    elif behaviour == "NN" and np.random.rand() < 0.05:
        guessing_val, target_guessing = NN.agent.select_action(state, options, current_hand, hands, turn, opponents)
    else:
        # Card Choice
        if behaviour[0] == 'R' or random.randint(1, 4) == 1 or behaviour[0] == 'Q' or behaviour[0] == 'N' or behaviour[0] == 'B':
            guessing_val = options[random.randint(0, len(options) - 1)]
        elif behaviour[0] == 'M':
            # Find the value with the most occurrences in the hand
            arr = count_vals_in_hand(current_hand.cards)
            guessing_val = np.argmax(arr) + 2  # Convert index to value
        elif behaviour[0] == 'L':
            # Find the value with the least occurrences in the hand (excluding 0 occurrences)
            arr = count_vals_in_hand(current_hand.cards)
            least_occurrences = min(filter(lambda x: x > 0, arr))
            guessing_val = np.argmin(arr) + 2 if least_occurrences > 0 else random.choice(options)
        else:
            print("Behavour pt 0 doesnt exist")
        # Opponent Choice
        if behaviour[1] == 'R' or random.randint(1, 4) == 1 or behaviour[1] == 'Q' or behaviour[0] == 'N' or behaviour[0] == 'B':
            target_guessing = opponents[random.randint(0, len(opponents) - 1)]
        elif behaviour[1] == 'M':
            target_guessing = max(opponents, key=lambda x: hands[x].size)
        elif behaviour[1] == 'L':
            # Filter opponents with non-empty hands
            non_empty_opponents = [o for o in opponents if hands[o].size > 0]
            target_guessing = min(non_empty_opponents,
                                  key=lambda x: hands[x].size) if non_empty_opponents else random.choice(
                opponents)
        else:
            print("Behavour pt 1 doesnt exist")
    return (int(guessing_val) - 2), target_guessing


def score_analysis(books):
    # returns an array with the placment of 1st 2nd and so on in the index of the behaviour that got that place
    # count how many other entries are higher or equal than item. then minue form 5
    positions = []
    for i in range(0, 4):
        more = 0
        for e in range(0, 4):
            if books[e] <= books[i]:
                more = more + 1
        positions.append(5 - more)

    return positions


def scoring_analysis(scores, games_played):
    totals = np.zeros(10)
    for i in range(0, 10):
        totals[i] = (totals[i] + 5 * scores[i][0] + 3 * scores[i][1] + scores[i][2]) / games_played[i]
    return totals


scoring = np.zeros((12, 4))
games_played = np.zeros(12)

action_space = []
# initialise Q learning
# Define action space
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
opponents = ['0', '1', '2']
for val in ranks:
    for opp in opponents:
        action_space.append(str(opp) + str(val))

for games in range(0, 1000):
    # Options are split into 2. R, M and L. them being Random, Most and Least.
    # Frist pos is card from hand, then oppoennent. so RR is random card, random opponent. RL is random card and oppoennt with least cards
    b_ops = ["RR", "RM", "RL", "MR", "MM", "ML", "LR", "LM", "LL", "QQ", "NN", "BB"]
    behaviour = []
    behaviour_num = []
    for i in range(0, 4):
        chosen = random.randint(0, 11)
        behaviour.append(b_ops[chosen])
        behaviour_num.append(chosen)

    # Run game
    # print(behaviour)
    books = main_game(behaviour)

    # do overall algorithm scoring
    positions = score_analysis(books)
    for i in range(0, 4):
        scoring[behaviour_num[i]][positions[i] - 1] = scoring[behaviour_num[i]][positions[i] - 1] + 1
        # increase how many games its in
        games_played[behaviour_num[i]] = games_played[behaviour_num[i]] + 1
    if games % 100 == 0:
        print("Test game:", games)

print(scoring)
print(scoring_analysis(scoring, games_played))
win_anal = scoring[9][0] + scoring[9][1] / 2
percent_win = (win_anal / games_played[9]) * 100
run_time = QL.run_time_calc()
print("Run time:", run_time, "seconds")
print("Train time:", QL.train_time, "seconds")
print("Fitness function Q:", CM.fitness(percent_win, QL.train_time, run_time))  # Fitness
win_anal = scoring[10][0] + scoring[10][1] / 2
percent_win = (win_anal / games_played[10]) * 100
run_time = NN.run_time_calc()
print("Run time:", run_time, "seconds")
print("Train time:", NN.train_time, "seconds")
print("Fitness function NN:", CM.fitness(percent_win, NN.train_time, run_time))  # Fitness
win_anal = scoring[11][0] + scoring[11][1] / 2
percent_win = (win_anal / games_played[11]) * 100
run_time = BAY.run_time_calc()
print("Run time:", run_time, "seconds")
print("Train time:", BAY.train_time, "seconds")
print("Fitness function BAY:", CM.fitness(percent_win, BAY.train_time, run_time))  # Fitness

# Plotting graph
placements = ["1", "2", "3", "4"]
scores = {
    'Random Random': scoring[0],
    'Random Most': scoring[1],
    'Random Least': scoring[2],
    'Most Random': scoring[3],
    'Most Most': scoring[4],
    'Most Least': scoring[5],
    'Least Random': scoring[6],
    'Last Most': scoring[7],
    'Least Least': scoring[8],
    'Q': scoring[9],
    'Neural Network': scoring[10],
    'Bayesian learning': scoring[11],
}
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
# if n_categories > 4:
#     plt.xticks(rotation=45)  # Adjust rotation angle if needed

# Add legend and grid lines
ax.legend(loc='upper left', fontsize=8, ncol=4)
ax.grid(axis='y')

# Adjust plot borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()

