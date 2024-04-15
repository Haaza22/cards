import random
import Card_Manip as CM
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
import tensorflow as tf


def get_state_index(state):
    return int(state[0]) * 15 * 15 + int(state[1]) * 15 + int(state[2])
def run_time_calc():
    deckt = CM.Deck()
    handt1 = CM.Hand()
    handt2 = CM.Hand()
    handt3 = CM.Hand()
    handt4 = CM.Hand()
    handst = [handt1, handt2, handt3, handt4]
    CM.deal_deck(deckt, handt1, handt2, handt3, handt4, 5)
    start_time = time.time()
    # def choose_action(behaviour, options, opponents, current_hand, hands, state, turn):
    choose_action("NN", find_values(handt1.cards), [0, 1, 2], handt1, handst, [0, 4, 7], 0)
    end_time = time.time()
    run_time = end_time - start_time
    return run_time

def create_q_network(state_dim, action_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(state_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(action_dim)
    ])
    return model


# Define the epsilon-greedy policy
def epsilon_greedy_policy(q_values, epsilon, options, hand, hands, state, turn, opponents):
    # Make guess
    if random.random() < epsilon:
        # Make random move
        return choose_action("RR", options, opponents, hand, hands, state, turn)
    else:
        # Optimal action
        action_options = []
        # Make options in form of '22' so can be checked agaianst action space
        for opponent in opponents:
            for val in options:
                action_temp = opponent
                if int(turn) < int(opponent):
                    action_temp = int(action_temp) - 1
                action_options.append(str(action_temp) + str(val))

        ordered_actions = [action_space[j] for j in np.argsort(q_values)]

        for j in range(0, len(ordered_actions)):
            # Loop cards to find highest valid option
            if ordered_actions[j] in action_options:
                target = ordered_actions[j][0]
                if len(ordered_actions[j]) == 2:
                    guess_val = ordered_actions[j][1]
                else:
                    guess_val = ordered_actions[j][1:]
                return guess_val, target


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, experience):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


class DQNAgent:
    def __init__(self, state_dim, action_dim, capacity, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay,
                 target_update):
        self.q_network = create_q_network(state_dim, action_dim)
        self.target_network = create_q_network(state_dim, action_dim)
        self.target_network.set_weights(self.q_network.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.q_network.compile(optimizer=self.optimizer, loss='mse')
        self.memory = ReplayBuffer(capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def update_model(self):
        if len(self.memory.buffer) < self.batch_size:
            return
        batch = self.memory.sample(self.batch_size)
        # states, actions, rewards, next_states, dones = batch

        states = np.array([data[0] for data in batch])
        actions = np.array([data[1] for data in batch])
        rewards = np.array([data[2] for data in batch])
        next_states = np.array([data[3] for data in batch])
        dones = np.array([data[4] for data in batch])

        states = np.array(states)
        next_states = np.array(next_states)
        q_values = self.q_network.predict(states, verbose=0)
        next_q_values = self.target_network.predict(next_states, verbose=0)
        for i in range(self.batch_size):
            if dones[i]:
                q_values[i][actions[i]] = rewards[i]
            else:
                q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        self.q_network.fit(states, q_values, verbose=0, use_multiprocessing=True)

    def select_action(self, state, options, hand, hands, turn, opponents):
        state_array = np.array(state)
        input_shape = (3,)  # Represents two features
        reshaped_state = np.array(state_array).reshape(input_shape)
        reshaped_state = np.expand_dims(reshaped_state, axis=0)
        q_values = self.q_network.predict(reshaped_state, verbose=0)
        return epsilon_greedy_policy(q_values, self.epsilon, options, hand, hands, state, turn, opponents)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def train_nn():
    learn_rate = 0.1
    discount_rate = 0.9
    explore_element = 0.1
    training = True
    training_rounds = 100
    current_training_round = 0
    behaviour = ['NN', 'NN', 'NN', 'NN']
    states = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]

    while training:
        # Base game needs
        playing = True
        start_cards = 5
        turn = random.randint(0, 3)
        players = 4
        in_game = [True, True, True, True]
        books = [0, 0, 0, 0]
        what_booked = []
        pond_exists = True
        episode_reward = [0, 0, 0, 0]

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

        while playing:
            start_books = books[turn]
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
                # Options is between 2 and 14, a number
                options = find_values(current_hand.cards)
                opponents = valid_opponents(in_game, turn)

                guess_val, target = agent.select_action(states[turn], options, current_hand, hands, turn, opponents)

                action_target = target
                if int(turn) < int(target):
                    action_target = int(target) - 1

                action = str(action_target) + str(guess_val)
                action_num = int(action_target) * 13 + int(guess_val)
                # Setup
                target_hand = hands[int(target)]
                current_turn_record[1] = target
                current_turn_record[2] = guess_val

                # Do guess
                target_hand, current_hand, correct = guess(target_hand, current_hand, (guess_val + 2))

                if correct:
                    extra_turn = True
                    current_turn_record[3] = True
                elif pond_exists:
                    # Go fish
                    new_card = deck.draw_card()
                    current_hand.add_card(new_card)
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

            if current_hand.size == 0:
                in_game[turn] = False

            hands[turn] = current_hand

            # Update Q
            next_state = states[turn]
            reward = calc_reward(start_books, books[turn], extra_turn)
            # Update Q-table
            agent.memory.push((states[turn], action_num, reward, next_state, False))
            episode_reward[player] += reward
            agent.update_model()

            # update_states
            update_pos = 0
            for i in range(0, 4):
                if i != turn:
                    # Update the right one. so if 0 guess, then 1 0, 2 0 and 3 0 would need it
                    # If 1 guessed then 0 0, 2 1 and 3 2
                    # If 2 guessed then 0 2, 1 2 and 3 3
                    # If 3 guessed then 0 3, 1 3 and 2 3
                    states[i][update_pos] = guess_val
                    update_pos = update_pos + 1

            if alive > 1 or pond_exists:
                if not extra_turn:
                    turn = advance_turn(turn, in_game)
            else:
                playing = False
            turn_record.append(current_turn_record)

        current_training_round = current_training_round + 1
        if current_training_round % 10 == 0:
            print("Round", current_training_round)
        if current_training_round >= training_rounds:
            training = False


def calc_reward(start_points, end_points, extra_turn):
    if extra_turn:
        points = 4
    else:
        points = 0
    if start_points > end_points:
        points = points + 1
    return points


def initialize_state_space():
    # State is 3 most recent asked for. order it to lower size
    permutations = list(itertools.product(range(0, 15), repeat=3))
    # Print the state space
    return len(permutations)


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
            # Options is between 2 and 14, a number
            options = find_values(current_hand.cards)
            opponents_main = valid_opponents(in_game, turn)

            # Make guess
            guess_val, target = choose_action(behaviour[turn], options, opponents_main, current_hand, hands,
                                              states[turn], turn)

            # Setup
            target_hand = hands[int(target)]
            current_turn_record[1] = target
            current_turn_record[2] = guess_val

            if target not in opponents_main:
                print(behaviour[turn])
                print("Tar", target)
                print("Opp", opponents_main)
                print()

            # Do guess
            target_hand, current_hand, correct = guess(target_hand, current_hand, (guess_val + 2))

            if correct:
                extra_turn = True
                current_turn_record[3] = True
            elif pond_exists:
                # Go fish
                new_card = deck.draw_card()
                current_hand.add_card(new_card)
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


def choose_action(behaviour, options, opponents_action_choice, current_hand, hands, state, turn):
    if behaviour == "NN" and np.random.rand() < 0.05:
        guessing_val, target_guessing = agent.select_action(state, options, current_hand, hands, turn, opponents_action_choice)
    else:
        # Card Choice
        if behaviour[0] == 'R' or random.randint(1, 4) == 1 or behaviour[0] == 'N':
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
        if behaviour[1] == 'R' or random.randint(1, 4) == 1 or behaviour[1] == 'N':
            target_guessing = opponents_action_choice[random.randint(0, len(opponents_action_choice) - 1)]
        elif behaviour[1] == 'M':
            target_guessing = max(opponents_action_choice, key=lambda x: hands[x].size)
        elif behaviour[1] == 'L':
            # Filter opponents with non-empty hands
            non_empty_opponents = [o for o in opponents_action_choice if hands[o].size > 0]
            target_guessing = min(non_empty_opponents,
                                  key=lambda x: hands[x].size) if non_empty_opponents else random.choice(
                opponents_action_choice)
        else:
            print("Behavour pt 1 doesnt exist")
    return (guessing_val - 2), target_guessing


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


scoring = np.zeros((10, 4))
games_played = np.zeros(10)

action_space = []
# initialise Q learning
# Define action space
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
opponents = ['0', '1', '2']
for val in ranks:
    for opp in opponents:
        action_space.append(str(opp) + str(val))

# Satetes are, most recent asked card
num_states = initialize_state_space()
num_actions = len(action_space)
q_table = np.zeros((num_states, num_actions))

# train paramiters
state_dim = 3
action_dim = num_actions
capacity = 10000
batch_size = 64
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
target_update = 10

agent = DQNAgent(state_dim, action_dim, capacity, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay,
                 target_update)

print("Start training")
start_time = time.time()
train_nn()
end_time = time.time()
train_time = end_time - start_time
print("Trained done in", train_time, "seconds")

for games in range(0, 1000):
    # Options are split into 2. R, M and L. them being Random, Most and Least.
    # Frist pos is card from hand, then oppoennent. so RR is random card, random opponent. RL is random card and oppoennt with least cards
    b_ops = ["RR", "RM", "RL", "MR", "MM", "ML", "LR", "LM", "LL", "NN"]
    behaviour = []
    behaviour_num = []
    for i in range(0, 4):
        chosen = random.randint(0, 9)
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
run_time = run_time_calc()
print("Run time:", run_time, "seconds")
print("Train time:", train_time, "seconds")
print("Fitness function:", CM.fitness(percent_win, train_time, run_time)) # Fitness 41.6654877302435

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
    'Neural Network': scoring[9],
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

# Scores
# 200 rounds training, 1000 test!
# --
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(state_dim,)),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dense(action_dim)
#     ])
# Time taken: 1389.0456783771515 seconds
# Performance:3.43099274 [0.1 off best]
# Action speed: ?
# Main fitness: ?
# --
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(state_dim,)),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(action_dim)
#     ])
# Time taken: 1388.698704957962 seconds
# Performance: 3.45588235 [best by 0.3]
# Action speed: ?
# Main fitness: ?
# --
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(state_dim,)),
#         tf.keras.layers.Dense(16, activation='sigmoid'),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(52, activation='sigmoid'),
#         tf.keras.layers.Dense(action_dim)
#     ])
# Time taken: 1372.82106590271 seconds
# Performance: 3.49636804
# Action speed: 0.0
# Main fitness: 41.28429781203313
# --

# 100 training  rounds, 1000 test:
# --
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(state_dim,)),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dense(action_dim)
#     ])
# Time taken: 685.2835369110107 seconds
# Performance:3.38518519 (0.5 off best)
# Action speed: ?
# Main fitness: ?
# --
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(state_dim,)),
#         tf.keras.layers.Dense(16, activation='sigmoid'),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(52, activation='sigmoid'),
#         tf.keras.layers.Dense(action_dim)
#     ])
# Time taken:
# Performance:
# Action speed:
# Main fitness:
