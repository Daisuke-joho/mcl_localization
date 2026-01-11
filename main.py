# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import random
# import math

# NUM_PARTICLES = 100
# WORLD_SIZE = 50.0
# LANDMARK_POS = 30.0  # ランドマークの位置


# sigma_move = 0.2     # 移動誤差
# sigma_sensor = 0.3   # 観測誤差

# # 1. 初期状態
# particles = [random.uniform(0, WORLD_SIZE) for _ in range(NUM_PARTICLES)]
# true_pos = 0.0 # ロボットの初期位置

# fig, ax = plt.subplots(figsize=(12, 2))

# def update(frame):
#     global true_pos, particles
#     ax.clear()

#     # 2. ロボットの移動とパーティクルの更新
#     u = 0.5  # 制御指令（右に0.5m）
    
#     # ロボットの移動
#     true_pos += u + random.gauss(0, 0.05)
    
#     # 各パーティクルの移動
#     for i in range(NUM_PARTICLES):
#         particles[i] += u + random.gauss(0, sigma_move)

#     # 3. 観測後のセンサ値の反映
#     z = abs(LANDMARK_POS - true_pos) + random.gauss(0, 0.1)
    
#     weights = []
#     for p in particles:

#         z_hat = abs(LANDMARK_POS - p)
#         error = z - z_hat
#         likelihood = (1.0 / math.sqrt(2.0 * math.pi * sigma_sensor**2)) * \
#                      math.exp(- (error**2) / (2.0 * sigma_sensor**2))
#         weights.append(likelihood)

#     # 4. リサンプリング
#     if sum(weights) > 0:
#         particles = random.choices(particles, weights=weights, k=NUM_PARTICLES)

#     # 5. 描画
#     # ランドマーク
#     ax.scatter([LANDMARK_POS], [0], color='green', s=200, marker='*', label='Landmark')
#     # パーティクル
#     ax.scatter(particles, [0] * NUM_PARTICLES, color='blue', alpha=0.3, s=20, label='Particles')
#     # ロボット
#     ax.scatter([true_pos], [0], color='red', s=100, label='Robot(True)', marker='x', zorder=5)
    
#     ax.set_xlim(0, WORLD_SIZE)
#     ax.set_ylim(-0.5, 0.5)
#     ax.set_yticks([])
#     ax.legend(loc='upper right')
#     ax.set_title(f"MCL Step {frame}: Moving and Sensing Point Landmark")

# ani = animation.FuncAnimation(fig, update, frames=30, interval=300, repeat=False)
# ani.save('mcl_result.gif', writer='pillow')
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math

# --- 設定 ---
NUM_PARTICLES = 100
WORLD_SIZE = 60.0
LANDMARK_POS = 30.0  # ランドマーク
SENSOR_RANGE = 10.0  

sigma_move = 0.3     # 移動誤差
sigma_sensor = 0.3   # 観測誤差

# 1. 初期状態
particles = [random.uniform(0, 10.0) for _ in range(NUM_PARTICLES)] # 最初は0~10mに固めておく
true_pos = 5.0 

fig, ax = plt.subplots(figsize=(12, 2))

def update(frame):
    global true_pos, particles
    ax.clear()

    # 2. ロボットの移動
    u = 1.0 
    true_pos += u + random.gauss(0, 0.05)
    for i in range(NUM_PARTICLES):
        particles[i] += u + random.gauss(0, sigma_move)

    # 3. 観測
    dist_to_lm = abs(LANDMARK_POS - true_pos)
    weights = [1.0] * NUM_PARTICLES

    # ランドマークが観測範囲内のとき重みを計算
    if dist_to_lm < SENSOR_RANGE:
        z = dist_to_lm + random.gauss(0, 0.1)
        for i in range(NUM_PARTICLES):
            z_hat = abs(LANDMARK_POS - particles[i])
            error = z - z_hat
            weights[i] = (1.0 / math.sqrt(2.0 * math.pi * sigma_sensor**2)) * \
                         math.exp(- (error**2) / (2.0 * sigma_sensor**2))
        status = "IN RANGE: Sensing..."
    else:
        status = "OUT OF RANGE: Error accumulating..."

    # 4. リサンプリング
    if sum(weights) > 0:
        particles = random.choices(particles, weights=weights, k=NUM_PARTICLES)

    # 5. 描画
    ax.scatter([LANDMARK_POS], [0], color='green', s=200, marker='*', label='Landmark')
    
    ax.scatter(particles, [0] * NUM_PARTICLES, color='blue', alpha=0.3, s=20, label='Particles')
    ax.scatter([true_pos], [0], color='red', s=100, label='Robot(True)', marker='x', zorder=5)
    
    ax.set_xlim(0, WORLD_SIZE)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.legend(loc='upper right')
    ax.set_title(f"Step {frame}: {status}")

ani = animation.FuncAnimation(fig, update, frames=50, interval=150, repeat=False)
ani.save('mcl_result.gif', writer='pillow')
plt.show()