import random


class Deck:
    def __init__(self):
        self.top = random.randint(0, 51)
        self.size = 52
        self.in_deck = []
        # create deck
        for suit in ['S', 'C', 'D', 'H']:
            for num in range(2, 15):
                self.in_deck.append(Card(suit, num))

    def draw_card(self):
        if self.size > 0:
            ret = self.in_deck.pop(self.top)
            self.size = self.size - 1
            if self.size == 0:
                self.top = 0
            else:
                self.top = random.randint(0, self.size - 1)
            return ret
        return 0


class Card:
    def __init__(self, given_suit, given_val):
        self.suit = given_suit
        self.val = given_val

    def print_card(self):
        to_print = self.suit + " "
        if self.val == 11:
            to_print = to_print + "J"
        elif self.val == 12:
            to_print = to_print + "Q"
        elif self.val == 13:
            to_print = to_print + "K"
        elif self.val == 14:
            to_print = to_print + "A"
        else:
            to_print = to_print + str(self.val)
        print(to_print)


class Hand:
    def __init__(self):
        self.size = 0
        self.cards = []

    def add_card(self, card):
        self.cards.append(card)
        self.size = self.size + 1

    def remove_card(self, position):
        discarding = self.cards.pop(position)
        self.size = self.size - 1
        return discarding

def print_deck(deck):
    for card in deck:
        card.print_card()


def deal_deck(deck, hand1, hand2, hand3, hand4, how_many):
    for i in range(0, how_many):
        hand1.add_card(deck.draw_card())
        hand2.add_card(deck.draw_card())
        hand3.add_card(deck.draw_card())
        hand4.add_card(deck.draw_card())
    return deck, hand1, hand2, hand3, hand4

# testing
# deck = Deck()
# hand1 = Hand()
# hand2 = Hand()
# hand3 = Hand()
# hand4 = Hand()
#
# deck, hand1, hand2, hand3, hand4 = deal_deck(deck, hand1, hand2, hand3, hand4, 13)
# print_deck(hand1.cards)
# print(hand1.size)