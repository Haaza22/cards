import random
import Card_Manip as CM


# def advance_turn(turn, in_game):
#     if turn == 3:
#         next_turn = 0
#     else:
#         next_turn = turn + 1
#     if in_game[next_turn]:
#         return next_turn
#     else:
#         return advance_turn(next_turn, in_game)

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


# Base game needs
playing = True
start_cards = 5
turn = random.randint(0, 3)
players = 4
in_game = [True, True, True, True]
books = [0, 0, 0, 0]
what_booked = []
pond_exists = True

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

# set agents (only hand random rn)
behaviour = ["R", "R", "R", "R"]

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
            in_game[turn] = True
            if deck.size == 0:
                pond_exists = False

        # getting cards can ask for and opponents can ask
        options = find_values(current_hand.cards)
        opponents = valid_opponents(in_game, turn)

        # Find guess
        if behaviour[turn] == "R":
            guess_val = options[random.randint(0, len(options) - 1)]
            target = opponents[random.randint(0, len(opponents) - 1)]
            target_hand = hands[target]
            current_turn_record[1] = target
            current_turn_record[2] = guess_val

        else:
            print("Should be using random")

        # Do guess
        target_hand, current_hand, correct = guess(target_hand, current_hand, guess_val)

        if correct:
            extra_turn == True
            current_turn_record[3] = True
        elif pond_exists:
            # go fish
            new_card = deck.draw_card()
            current_hand.add_card(new_card)
            if deck.size == 0:
                pond_exists = False
            if new_card.val == guess_val:
                extra_turn == True
                current_turn_record[4] = True

        # check if targets last cars was taken
        if target_hand.size == 0:
            in_game[target] = False

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

    if current_hand.size == 0:
        in_game[turn] = False

    hands[turn] = current_hand

    if alive > 1 or pond_exists:
        if not extra_turn:
            turn = advance_turn(turn, in_game)
    else:
        playing = False
    turn_record.append(current_turn_record)

print(books)
