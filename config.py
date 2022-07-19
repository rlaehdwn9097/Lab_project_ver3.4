import math
from platform import node

# runtime info
# 유저가 요청 횟수 
# 요청이 많은 시간대 고려 안함
# 라운드 사이의 간격은 랜덤함
# 따라서 라운드를 통해서 요일을 계산하고 해당 요일에 요청이 많은 컨테츠를 학습하여 BS 미리 해당 컨텐츠를 캐싱하는 것을 목표
TOTAL_PRIOD = 100 #(week)
MAX_ROUNDS = 30000   #70000 2만이나 3만으로 늘리고 기간을 한달이라고 하면 총 24 * 30일 
MAX_REQ_PER_ROUND = 1

# info of node
NB_NODES = 300 # 300개 
TX_RANGE = 30 # meters

# area definition
AREA_WIDTH = 5000.0
AREA_LENGTH = 5000.0


NUM_microBS = [5,5] # 36개
NUM_BS=[2,2] # 9개

# storage size
microBS_SIZE = 100
BS_SIZE = 100
CENTER_SIZE = 200
#scenario info

#1.test


#잠깐 정리
# network latency = propagation delay + transmission delay + processing delay https://www.rfwireless-world.com/calculators/Network-Latency-Calculator.html
# propagation delay = distance / speed
# transmission delay = 
# serialization delay = packet size (bits) / Transmission Rate (bps)

DLthroughput = 1386000000
ULthroughput = 380000000
DLpackets_per_second = 108281.25
ULpackets_per_second = 29687.5

LIGHT_SPEAD = 299792458
PACKET_SIZE = 12800 #(1500byte)
#packet size embb 1500byte
#packet size urllc 32bytes
#pacekt size mMTC 32bytes
#throughput https://5g-tools.com/5g-nr-throughput-calculator/ 

LATENCY_INTERNET = 0.01#0.0422 #ms

# Episode Number
MAX_EPISODE_NUM = 500

# DQN structure
DROPOUT_RATE = 0.2
H1 = 3
H2 = 9
H3 = 6
H4 = 31
q = 4

# DQN 하이퍼파라미터
GAMMA = 0.95
BATCH_SIZE = 4 #64
BUFFER_SIZE = 20000
DQN_LEARNING_RATE = 0.001
TAU = 0.001
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01


#action 
NB_ACTION = 7000

# reward parameter
a = 1
b = 0.5
c = 0.5
d = 0.1
e = 10





