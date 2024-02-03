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
        return hand.cards
    else:
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


def count_points(won_cards):
    points = [0, 0, 0, 0]
    for i in range(0, len(points)):
        for card in won_cards[i]:
            if card.suit == "H":
                points[i] = points[i] + 1
            elif card.suit == "C" and card.val == 12:
                points[i] = points[i] + 13
    return points


playing = True
start_cards = 13
leader = random.randint(0, 3)
players = 4

# an array of 6 items, [lead suit, lead, what player 1 played, what player 2 played, what player 3 played, what player 4 played,]
turn_record = []

# setting starting card logic
deck = CM.Deck()
hand1 = CM.Hand()
hand2 = CM.Hand()
hand3 = CM.Hand()
hand4 = CM.Hand()
deck, hand1, hand2, hand3, hand4 = CM.deal_deck(deck, hand1, hand2, hand3, hand4, start_cards)
hands = [hand1, hand2, hand3, hand4]
won_cards = [[], [], [], []]

behaviour_passing = ["R", "R", "R", "R"]
behaviour = ["R", "R", "R", "R"]

# pre game 3 card move 1 up
moving=[[],[],[],[]]

# Select removing
for i in range(0,4):
    for e in range(0,3):
        if behaviour_passing[i] == "R":
            moving[i].append(hands[i].remove_card(random.randint(0,hands[i].size-1)))
        else:
            print("Should be R P")


for i in range(0,4):
    adding_to = next(i)
    for e in range(0,3):
        hands[adding_to].add_card(moving[i][e])

while playing:
    current_turn_record = ["", 0, 0, 0, 0, 0]
    current_hand = hands[leader]
    options = current_hand.cards

    # leader chosen card
    if behaviour[leader] == "R":
        playing = current_hand.remove_card(random.randint(0, len(options) - 1))
    else:
        print("Should be R 1")

    current_turn_record[0] = playing.suit
    current_turn_record[1] = leader
    current_turn_record[2+leader] = playing


    current_player = next(leader)
    on_table = [playing]
    who_played = [leader]
    for i in range(0, 3):
        options = calc_options(hands[current_player], playing.suit)

        if behaviour[current_player] == "R":
            random_card = random.randint(0, len(options) - 1)
            on_table.append(hands[current_player].remove_card(random_card))
            who_played.append(current_player)
        else:
            print("Should be R 2")

        current_turn_record[2+current_player] = on_table[i]
        current_player = next(current_player)

    round_winner = who_played[find_winner(on_table)]

    for card in on_table:
        won_cards[round_winner].append(card)
    leader = round_winner

    if current_hand.size == 0:
        playing = False

    turn_record.append(current_turn_record)

points = count_points(won_cards)
print("Points", points)

print(turn_record)