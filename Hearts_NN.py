import random
import Card_Manip as CM
import Hearts
import numpy as np
import itertools
import matplotlib.pyplot as plt
import time
import tensorflow as tf
# from keras import layers
# from keras import activations
from sklearn.model_selection import train_test_split


def run_time_calc():
    deckt = CM.Deck()
    handt1 = CM.Hand()
    handt2 = CM.Hand()
    handt3 = CM.Hand()
    handt4 = CM.Hand()
    CM.deal_deck(deckt, handt1, handt2, handt3, handt4, 13)
    start_time = time.time()
    choose_card_base("NN", handt1, [0, 1, 2, 3, 4, 5, 6], [2, 7])
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
    elif behaviour == 'NN':
        playing, actions_, valid_actions_ = agent.select_action(state, options, current_hand)
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


def create_q_network(state_dim, action_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(16, activation='sigmoid'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(52, activation='sigmoid'),
        tf.keras.layers.Dense(action_dim)
    ])
    return model


# Define the epsilon-greedy policy
def epsilon_greedy_policy(q_values, epsilon, options, hand):
    action_options = []
    for option in options:
        cur_card = hand.cards[option]
        action_options.append(str(cur_card.suit) + str(cur_card.val))
    valid_actions = action_options
    if random.random() < epsilon:
        position_in_hand = random.randint(0, len(options) - 1)
        action_form = str(hand.cards[position_in_hand].suit) + str(
            hand.cards[position_in_hand].val)
        playing = hand.remove_card(options[position_in_hand])
        return playing, action_form, valid_actions
    else:
        ordered_actions = [action_space[j] for j in np.argsort(q_values)]
        # Action index list! order them, 0 best, 51 worst
        for j in range(0, len(ordered_actions)):
            # Loop cards to find highest valid option
            if ordered_actions[j] in action_options:
                position_in_hand = action_options.index(ordered_actions[j])
                action_form = str(hand.cards[position_in_hand].suit) + str(
                    hand.cards[position_in_hand].val)
                playing = hand.remove_card(options[position_in_hand])
                return playing, action_form, valid_actions


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
                action_form = CM.card_number(actions[i])
                q_values[i][action_form] = rewards[i] + self.gamma * np.max(next_q_values[i])
        self.q_network.fit(states, q_values, verbose=0, use_multiprocessing=True)

    def select_action(self, state, options, hand):
        state_array = np.array(state)
        input_shape = (2,)  # Represents two features
        reshaped_state = np.array(state_array).reshape(input_shape)
        reshaped_state = np.expand_dims(reshaped_state, axis=0)
        q_values = self.q_network.predict(reshaped_state, verbose=0)
        return epsilon_greedy_policy(q_values, self.epsilon, options, hand)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


def train_nn():
    training_data = []
    training = True
    training_rounds = 100
    current_round = 0
    while training:
        played_cards = []
        hands, current_player = Hearts.setup(['Q', 'Q', 'Q', 'Q'])
        won_cards = [[], [], [], []]
        turn_record = []
        cur_score = [0, 0, 0, 0]
        state = [[], [], [], []]
        valid_in_loop = []
        episode_reward = [0, 0, 0, 0]
        for i in range(0, 4):
            state[i] = [0, 0]

        # Do game
        for i in range(0, 13):
            episode_data = [0, 0, 0, 0]
            # Hearts setup
            starter = True
            current_turn_record = ["", 0, 0, 0, 0, 0]
            who_played = []
            on_table = []
            # nn setup
            actions = [0, 0, 0, 0]
            valid_actions = [0, 0, 0, 0]
            rewards = [0, 0, 0, 0]  # Not sure if need

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

                playing, actions[current_player], valid_actions[current_player] = agent.select_action(
                    state[current_player], options, hands[current_player])

                on_table, who_played, current_turn_record = Hearts.one_loop(current_player, starter, on_table,
                                                                            who_played,
                                                                            current_turn_record, playing)
                current_player = Hearts.next(current_player)
                starter = False

            # Round end
            won_cards, turn_record, current_player, played_cards = Hearts.round_over(who_played, on_table, won_cards,
                                                                                     turn_record,
                                                                                     current_turn_record, played_cards)

            for player in range(0, 4):
                # stats needs to be hashed
                # action is number 1 to 52, also currently in 'S2'
                # valid action is an array of all valid actions in action form (1-52), currently in form 'S2'
                # reward is not needed rn?
                next_state = [0, 0]
                agent.memory.push((state[player], actions[player], rewards[player], next_state, False))
                episode_reward[player] += rewards[player]
                agent.update_model()

        # Training round over
        current_round = current_round + 1
        if current_round % 10 == 0:
            print("Training round:", current_round)
        if current_round >= training_rounds:
            training = False
    print("Training Rounds:", current_round)


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

# train paramiters
state_dim = 2
action_dim = 52
capacity = 10000
batch_size = 64
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
target_update = 10

agent = DQNAgent(state_dim, action_dim, capacity, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay,
                 target_update)

print("Train NN start")
start_time_nn = time.time()
train_nn()
end_time_nn = time.time()
train_time = end_time_nn - start_time_nn
print("NN trained in", train_time, "seconds")

for game in range(0, 1000):
    b_ops = ["R", "L", "Hi", "He1", "He2", "He3", "NN"]
    behaviour_pre = []
    behaviour_num = []

    # select random behaviours
    for i in range(0, 4):
        chosen = random.randint(0, 6)
        behaviour_pre.append(b_ops[chosen])
        behaviour_num.append(chosen)

    points, positions = make_game(behaviour_pre)

    # increase the score of the behaviour by 1 in its position, positions needs to be -1 becayse its 1-4 and not 0-3
    for i in range(0, 4):
        scoring[behaviour_num[i]][positions[i] - 1] = scoring[behaviour_num[i]][positions[i] - 1] + 1
        # increase how many games its in
        games_played[behaviour_num[i]] = games_played[behaviour_num[i]] + 1

    if game % 100 == 0:
        print("game:", game)

# Outpts
# Scores n jazz
print(scoring)
print(scoring_analysis(scoring, games_played))
win_anal = scoring[6][0] + scoring[6][1] / 2
percent_win = (win_anal / games_played[6]) * 100
run_time = run_time_calc()
print("Run time:", run_time, "seconds")
print("Train time:", train_time, "seconds")
print("Fitness function:", CM.fitness(percent_win, train_time, run_time))

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
ax.grid(axis='y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()

tf.get_logger().setLevel('INFO')

# Scores
# 200 training  rounds, 1000 test:
# --
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(2,)),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dense(action_dim)
#     ])
# Time taken: 1038.2910697460175 seconds
# Performance: 2.5631399317406145
# Action speed: 0.027246475219726562 seconds
# Main fitness: 27.64452708878208
# --
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(2,)),
#         tf.keras.layers.Dense(16, activation='sigmoid'),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(52, activation='sigmoid'),
#         tf.keras.layers.Dense(action_dim)
#     ])
# Time taken: 1023.0342795848846 seconds
# Performance: 2.607142857142857
# Action speed: 0.02358078956604004 seconds
# Main fitness: 28.338620777924856
# --
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(2,)),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(action_dim)
#     ])
# Time taken: 1016.3453371524811 seconds
# Performance: 2.557142857142857 (0.1 off best)
# Action speed: 0.024005413055419922 seconds
# Main fitness: 25.795309471648974
# --
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(2,)),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(action_dim)
#     ])
# Time taken: 1018.9866826534271 seconds
# Performance: 2.625 (The Best!)
# Action speed: 0.024296283721923828 seconds
# Main fitness: 25.030236609494533


# 100 rounds training
# --
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input(shape=(2,)),
#         tf.keras.layers.Dense(16, activation='sigmoid'),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(52, activation='sigmoid'),
#         tf.keras.layers.Dense(action_dim)
#     ])
# Time taken: 509.76108980178833 seconds
# Performance: 2.6196213425129087
# Action speed: 0.02400517463684082 seconds
# Main fitness: 37.73553077873661

# could optimise action making speed by once finished training it will predice table for all states. Would increase training a fair bit tho
