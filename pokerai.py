import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import to_categorical

suits = ['hearts', 'diamonds', 'clubs', 'spades']
values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
deck = [f'{value} of {suit}' for value in values for suit in suits]


def print_intro():
    print("============================================================")
    print("PokerAI")
    print("Created by: Corvus Codex")
    print("Github: https://github.com/CorvusCodex/")
    print("Licence : MIT License")
    print("Support my work:")
    print("BTC: bc1q7wth254atug2p4v9j3krk9kauc0ehys2u8tgg3")
    print("ETH & BNB: 0x68B6D33Ad1A3e0aFaDA60d6ADf8594601BE492F0")
    print("Buy me a coffee: https://www.buymeacoffee.com/CorvusCodex")
    print("============================================================")

print_intro()

def shuffle_and_deal(deck, num_cards):
    random.shuffle(deck)
    return deck[:num_cards]

def encode_card(card):
    return to_categorical(deck.index(card), num_classes=len(deck))

# Simulate 1 million games
data = []
for _ in range(1000000):
    hand = shuffle_and_deal(deck, 2)
    table = shuffle_and_deal(deck, 4)
    next_card = shuffle_and_deal(deck, 1)[0]
    data.append((hand + table, next_card))

X = np.array([[encode_card(card) for card in game[0]] for game in data[:-1]])  # all but the last game
X = X.reshape((X.shape[0], 6, len(deck)))

y = to_categorical(np.array([deck.index(data[i][1]) for i in range(1, len(data))]), num_classes=len(deck))  # all but the first game

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(len(deck), activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy')

model.fit(X, y, epochs=3, verbose=0)

user_hand = input("Enter your cards (comma-separated): ").split(',')
user_table = input("Enter the table cards (comma-separated): ").split(',')

while len(user_hand + user_table) < 6:
    user_table.append('2 of hearts')

user_input = np.array([encode_card(card.strip()) for card in user_hand + user_table]).reshape(1, -1, len(deck))

prediction = model.predict(user_input)

for card in user_hand + user_table:
    prediction[0, deck.index(card.strip())] = 0

prediction /= np.sum(prediction)

next_card = np.argmax(prediction)
probability = np.max(prediction)

probability_percentage = probability * 100

print(f"The next card is likely to be {deck[next_card]} with a probability of {probability_percentage}%.")
