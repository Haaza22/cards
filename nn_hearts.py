import random
import Card_Manip
import Hearts
import numpy as np


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


def choose_card_base(behaviour, current_hand, options, on_table):
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
        # for card in options:
        #     if current_hand.cards[card].suit != "H":
        #         new_options.append(current_hand.cards[card])

        if len(new_options) == 0:
            new_options = options

        if behaviour == "He1":
            playing = choose_card_base("R", current_hand, new_options, on_table)
        elif behaviour == "He2":
            playing = choose_card_base("L", current_hand, new_options, on_table)
        else:
            playing = choose_card_base("Hi", current_hand, new_options, on_table)
    else:
        print("Behaviour should be R, L, Hi, He1, He2 or He3")
    return playing


def make_game(behaviour):
    hands, current_player = Hearts.setup(behaviour)
    won_cards = [[], [], [], []]
    # an array of 6 items, [lead suit, lead player, what player 1 played, what player 2 played, what player 3 played, what player 4 played,]
    turn_record = []
    for i in range(0, 13):
        starter = True
        current_turn_record = ["", 0, 0, 0, 0, 0]
        who_played = []
        on_table = []

        for e in range(0, 4):
            # Options is a list contianing all the positions in the hand which are valid cards to play
            if starter:
                options = Hearts.list_hand(len(hands[current_player].cards))
            else:
                options = Hearts.calc_options(hands[current_player], current_turn_record[0])
            playing = choose_card_base(behaviour[current_player], hands[current_player], options, on_table)

            on_table, who_played, current_turn_record = Hearts.one_loop(current_player, starter, on_table, who_played,
                                                                        current_turn_record, playing)

            current_player = Hearts.next(current_player)
            starter = False

        won_cards, turn_record, current_player = Hearts.round_over(who_played, on_table, won_cards, turn_record,
                                                                   current_turn_record)
    points = Hearts.count_points(won_cards)
    positions = win_order(points)

    return points, positions


def score_hearts(state):
    print("WIP")
    # as rounds go on score goes up, as you gain points, loose score. +1 score per round, -1 score per point



scoring = np.zeros((6, 4))

for game in range(0, 100):
    b_ops = ["R", "L", "Hi", "He1", "He2", "He3"]
    behaviour = []
    behaviour_num = []

    # select random behaviours
    for i in range(0, 4):
        chosen = random.randint(0, 5)
        behaviour.append(b_ops[chosen])
        behaviour_num.append(chosen)

    points, positions = make_game(behaviour)

    # increase the score of the behaviour by 1 in its position, positions needs to be -1 becayse its 1-4 and not 0-3
    for i in range(0, 4):
        scoring[behaviour_num[i], positions[i] - 1] = scoring[behaviour_num[i], positions[i] - 1] + 1

print(scoring)
