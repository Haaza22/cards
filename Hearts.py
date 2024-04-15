import random
import Card_Manip as CM


def next(turn):
    if turn == 3:
        return 0
    return turn + 1


def calc_options(hand, given_suit):
    options = []
    for i in range(0, hand.size):
        if hand.cards[i].suit == given_suit:
            options.append(i)
    if len(options) == 0:
        return list_hand(len(hand.cards))
    else:
        return options


def calc_options_B(hand, given_suit):
    options = []
    for i in range(0, hand.size):
        if hand.cards[i].suit == given_suit:
            options.append(i)
    if len(options) == 0:
        return list_hand(len(hand.cards)), False
    else:
        return options, True


def list_hand(hand_size):
    options = []
    for i in range(0, hand_size):
        options.append(i)
    return options


def find_winner(table):
    given_suit = table[0].suit
    highest = 0
    winner = 0
    for i in range(0, len(table)):
        if table[i].suit == given_suit and table[i].val > highest:
            highest = table[i].val
            winner = i
    return winner


def find_wining_card(table):
    given_suit = table[0].suit
    highest = 0
    for i in range(0, len(table)):
        if table[i].suit == given_suit and table[i].val > highest:
            highest = table[i].val
    return str(given_suit) + str(highest)


def count_points(won_cards):
    points = [0, 0, 0, 0]
    for i in range(0, len(points)):
        for card in won_cards[i]:
            if card.suit == "H":
                points[i] = points[i] + 1
            elif card.suit == "C" and card.val == 12:
                points[i] = points[i] + 13
    return points


def setup(behaviour_passing):
    start_cards = 13
    leader = random.randint(0, 3)

    # setting starting card logic
    deck = CM.Deck()
    hand1 = CM.Hand()
    hand2 = CM.Hand()
    hand3 = CM.Hand()
    hand4 = CM.Hand()
    deck, hand1, hand2, hand3, hand4 = CM.deal_deck(deck, hand1, hand2, hand3, hand4, start_cards)
    hands = [hand1, hand2, hand3, hand4]

    # pre game 3 card move 1 up
    moving = [[], [], [], []]

    # Select removing
    for i in range(0, 4):
        for e in range(0, 3):
            if behaviour_passing[i] == "R":
                moving[i].append(hands[i].remove_card(random.randint(0, hands[i].size - 1)))
            else:
                moving[i].append(hands[i].remove_card(random.randint(0, hands[i].size - 1)))

    for i in range(0, 4):
        adding_to = next(i)
        for e in range(0, 3):
            hands[adding_to].add_card(moving[i][e])
    # End moving
    return hands, leader


def one_loop(current_player, start, on_table, who_played, current_turn_record, playing):
    if start:
        # leader chosen card
        current_turn_record = ["", 0, 0, 0, 0, 0]

        current_turn_record[0] = playing.suit
        current_turn_record[1] = current_player
        current_turn_record[2 + current_player] = playing
        on_table = [playing]
        who_played = [current_player]

        return on_table, who_played, current_turn_record

    else:
        on_table.append(playing)
        who_played.append(current_player)
        current_turn_record[2 + current_player] = playing
        return on_table, who_played, current_turn_record


def round_over(who_played, on_table, won_cards, turn_record, current_turn_record, played_cards):
    round_winner = who_played[find_winner(on_table)]

    for card in on_table:
        won_cards[round_winner].append(card)

    turn_record.append(current_turn_record)

    for i in range(2, 6):
        played_cards.append((current_turn_record[i].val, current_turn_record[i].suit))

    return won_cards, turn_record, round_winner, played_cards


def bonus(won_cards, turn_record):
    points = count_points(won_cards)
    print("Points", points)

    print(turn_record)
