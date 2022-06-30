from math import *
import SimConnect
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os
import time

AIRSPEED = 'AIRSPEED_INDICATED'
ALTITUDE = 'PLANE_ALT_ABOVE_GROUND'
VERTICAL_SPEED = 'VERTICAL_SPEED'
ELEVATOR = 'ELEVATOR_POSITION'
AILERON = 'AILERON_POSITION'
RUDDER = 'RUDDER_POSITION'
HEADING = 'PLANE_HEADING_DEGREES_TRUE'
FLAPS = 'FLAPS_HANDLE_INDEX'
ANGLE_OF_ATTACK = 'ANGLE_OF_ATTACK_INDICATOR'
ENGINE = 'ENGINE'
THROTTLE = ['GENERAL_ENG_THROTTLE_LEVER_POSITION:1', 'GENERAL_ENG_THROTTLE_LEVER_POSITION:2']
BRK = 'BRAKE'
BRAKE = ['BRAKE_LEFT_POSITION', 'BRAKE_RIGHT_POSITION']
SPOILERS = 'SPOILERS_HANDLE_POSITION'
PITCH = 'PLANE_PITCH_DEGREES'
BANK = 'PLANE_BANK_DEGREES'

actions = [ELEVATOR + ':0', ELEVATOR + ':1', ELEVATOR + ':2', ELEVATOR + ':3',
           ELEVATOR + ':4', ELEVATOR + ':5', ELEVATOR + ':6', ELEVATOR + ':7',
           ELEVATOR + ':8', ELEVATOR + ':9', ELEVATOR + ':A', ELEVATOR + ':B',
           ELEVATOR + ':C', ELEVATOR + ':D', ELEVATOR + ':E', ELEVATOR + ':F',
           AILERON + ':0', AILERON + ':1', AILERON + ':2', AILERON + ':3',
           AILERON + ':4', AILERON + ':5', AILERON + ':6', AILERON + ':7',
           AILERON + ':8', AILERON + ':9', AILERON + ':A', AILERON + ':B',
           AILERON + ':C', AILERON + ':D', AILERON + ':E', AILERON + ':F',
           RUDDER + ':0', RUDDER + ':1', RUDDER + ':2', RUDDER + ':3',
           RUDDER + ':4', RUDDER + ':5', RUDDER + ':6', RUDDER + ':7',
           RUDDER + ':8', RUDDER + ':9', RUDDER + ':A', RUDDER + ':B',
           RUDDER + ':C', RUDDER + ':D', RUDDER + ':E', RUDDER + ':F',
           SPOILERS + ':0', SPOILERS + ':1', SPOILERS + ':2', SPOILERS + ':3',
           SPOILERS + ':4', SPOILERS + ':5', SPOILERS + ':6', SPOILERS + ':7',
           SPOILERS + ':8', SPOILERS + ':9', SPOILERS + ':A', SPOILERS + ':B',
           SPOILERS + ':C', SPOILERS + ':D', SPOILERS + ':E', SPOILERS + ':F',
           FLAPS + ':0', FLAPS + ':1', FLAPS + ':2', FLAPS + ':3', FLAPS + ':4',
           ENGINE + ':REV', ENGINE + ':IDLE', ENGINE + ':A/THR', ENGINE + ':CL', ENGINE + ':FLX', ENGINE + ':TOGA',
           BRK]

state_size = 10
action_size = len(actions)
batch_size = 32
n_episodes = 100000
game_speed = 1
output_dir = "model_output/airborne/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

sc = SimConnect.SimConnect()
aq = SimConnect.AircraftRequests(sc, _time=2000)
distance_x = 0
distance_y = 0
RWY_lat = 50.040059
RWY_lon = 8.586546
RWY_head = 249.5

earth_radius_feet = 20902000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, activation="relu", input_dim=self.state_size))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward  # if done
            if not done:
                target = (reward +
                          self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def save(self, name):
        f = open(name[0:-5] + ".txt", 'w')
        f.write(str(self.epsilon))
        f.close()
        self.model.save_weights(name)

    def load(self, name):
        f = open(name[0:-5] + ".txt", 'r')
        self.epsilon = float(f.read())
        f.close()
        self.model.load_weights(name)


def reset_pos():
    x = random.randint(1000, 50000)
    print(x)
    alt = tan(radians(3)) * (x - 1000)
    brng = radians(RWY_head - 180)
    lat1 = radians(RWY_lat)
    lon1 = radians(RWY_lon)

    lat2 = asin(sin(lat1) * cos(x / earth_radius_feet) + cos(lat1) * sin(x / earth_radius_feet) * cos(brng))
    lon2 = lon1 + atan2(sin(brng) * sin(x / earth_radius_feet) * cos(lat1),
                        cos(x / earth_radius_feet) - sin(lat1) * sin(lat2))

    lat2 = degrees(lat2)
    lon2 = degrees(lon2)

    aq.set(ALTITUDE, alt)
    time.sleep(0.2)
    aq.set('PLANE_LATITUDE', lat2)
    aq.set('PLANE_LONGITUDE', lon2)


def reset():
    reset_pos()
    time.sleep(0.2)
    aq.set(HEADING, radians(249.5))
    aq.set(PITCH, 0)
    aq.set(BANK, 0)
    aq.set(AIRSPEED, 145)
    aq.set(THROTTLE[0], 28)
    aq.set(THROTTLE[1], 28)
    aq.set('FUEL_TOTAL_QUANTITY', 7061)
    time.sleep(0.1)
    return get_state()


def get_state():
    return [aq.get(AIRSPEED), aq.get(ALTITUDE), aq.get(VERTICAL_SPEED), aq.get(HEADING), aq.get(FLAPS),
            aq.get(ANGLE_OF_ATTACK), aq.get(BANK), aq.get(PITCH), distance_x, distance_y]


def random_action():
    return random.choice(actions)


def distance_feet_GPS_coordinates(lat1, lon1, lat2, lon2):
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)

    lat1 = radians(lat1)
    lat2 = radians(lat2)

    a = sin(d_lat / 2) * sin(d_lat / 2) + sin(d_lon / 2) * sin(d_lon / 2) * cos(lat1) * cos(lat2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return earth_radius_feet * c


def heading_GPS_coordinates(lat1, lon1, lat2, lon2):
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    lon1 = radians(lon1)
    lon2 = radians(lon2)

    d_lon = (lon2 - lon1)
    y = sin(d_lon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(d_lon)

    brng = degrees(atan2(y, x))
    brng = (brng + 360) % 360
    return brng


def calc_glideslope_reward():
    global distance_x
    global distance_y
    lat = None
    lon = None
    while lat is None or lon is None:
        lat = aq.get('PLANE_LATITUDE')
        lon = aq.get('PLANE_LONGITUDE')
    phi = round(249.5 - (heading_GPS_coordinates(lat, lon, RWY_lat, RWY_lon)))
    r = distance_feet_GPS_coordinates(lat, lon, RWY_lat, RWY_lon)
    x = r * cos(radians(phi))
    y = r * sin(radians(phi))
    distance_x = x
    distance_y = y
    glideslope_alt = tan(radians(3)) * (x - 1000)
    if glideslope_alt < 0:
        glideslope_alt = 0
    plane_alt = aq.get(ALTITUDE)
    diff_alt = abs(glideslope_alt - plane_alt)
    distance = sqrt(pow(diff_alt, 2) + pow(y, 2))
    return sigmoid(distance / 2000 - 3) * -2 + 1


def calc_localizer_reward():
    lat = None
    lon = None
    while lat is None or lon is None:
        lat = aq.get('PLANE_LATITUDE')
        lon = aq.get('PLANE_LONGITUDE')
    phi = round(249.5 - (heading_GPS_coordinates(lat, lon, RWY_lat, RWY_lon)))
    diff = abs(phi)
    return sigmoid(diff / -2.5) * 2


def calc_bank_pitch_reward():
    pitch = aq.get(PITCH)
    bank = aq.get(BANK)
    pitch_reward = sigmoid(pitch / -5 + 5) - 0.5
    bank_reward = sigmoid(bank / -5 + 5) - 0.5
    return pitch_reward + bank_reward


def sigmoid(x):
    return 1 / (1 + exp(-x))


def calc_reward():
    reward = calc_glideslope_reward() * 6 + calc_localizer_reward() * 3 + calc_bank_pitch_reward()
    print(reward)
    return reward


def do_action(action):
    act = action.split(':')[0]
    if act == ELEVATOR or act == AILERON or act == RUDDER:
        # Elevator, Aileron or Rudder
        level = (int(action.split(':')[1], 16) - 8) * 100 / 8
        aq.set(act, level)
    elif act == SPOILERS:
        # Spoilers
        level = int(action.split(':')[1], 16) * 100 / 16
        aq.set(SPOILERS, level)
    elif act == FLAPS:
        # Flaps
        level = int(action.split(':')[1])
        aq.set(act, level)
    elif act == ENGINE:
        # Engine
        level = action.split(':')[1]
        if level == 'REV':
            # Reverse Thrust
            aq.set(THROTTLE[0], -20)
            aq.set(THROTTLE[1], -20)
        elif level == 'IDLE':
            # IDLE
            aq.set(THROTTLE[0], 0)
            aq.set(THROTTLE[1], 0)
        elif level == 'A/THR':
            # Auto-throttle
            aq.set(THROTTLE[0], 50)
            aq.set(THROTTLE[1], 50)
        elif level == 'CL':
            # Climb
            aq.set(THROTTLE[0], 90)
            aq.set(THROTTLE[1], 90)
        elif level == 'FLX':
            # Takeoff Power
            aq.set(THROTTLE[0], 95)
            aq.set(THROTTLE[1], 95)
        elif level == 'TOGA':
            # Takeoff / Go around
            aq.set(THROTTLE[0], 100)
            aq.set(THROTTLE[1], 100)
    elif act == BRK:
        # Brake
        aq.set(BRAKE[0], 0x8000)
        aq.set(BRAKE[1], 0x8000)

    next_state = get_state()
    reward = calc_reward()
    done = aq.get('SIM_ON_GROUND') and (round(aq.get('GPS_GROUND_SPEED')) < 10)
    return next_state, reward, done


def main():
    agent = DQNAgent(state_size, action_size)
    try:
        agent.load(output_dir + "weights.hdf5")
    except:
        pass
    for e in range(n_episodes):
        timeStop = time.time() + 120

        state = reset()
        np.reshape(state, (1, state_size))

        done = False
        timeStep = 0
        while not done:
            try:
                action = agent.act(state)
                next_state, reward, done = do_action(actions[action])
                reward = reward if not done else -10
                next_state = np.reshape(next_state, (1, state_size))
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, n_episodes - 1, timeStep, agent.epsilon))
                timeStep += 1
                time.sleep(0.1 * game_speed)
                if timeStop < time.time():
                    break
            except:
                pass

        if len(agent.memory) > batch_size:
            try:
                agent.train(batch_size)
            except:
                pass
        if e % 10 == 0:
            agent.save(output_dir + "weights.hdf5")


if __name__ == '__main__':
    main()
