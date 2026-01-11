import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math

NUM_PARTICLES = 100
WORLD_SIZE = 20.0
LANDMARK_POS = 15.0  # ランドマークの位置


sigma_move = 0.2     # 移動誤差
sigma_sensor = 0.3   # 観測誤差

# 1. 初期状態
particles = [random.uniform(0, WORLD_SIZE) for _ in range(NUM_PARTICLES)]
true_pos = 0.0 # ロボットの初期位置

fig, ax = plt.subplots(figsize=(12, 2))

def update(frame):
    global true_pos, particles
    ax.clear()

    # --- 2. 移動後のパーティクルの姿勢更新 (Motion Update) ---
    # 教科書 5.2.1項：状態遷移モデル p(x | x_{t-1}, u)
    u = 0.5  # 制御指令（右に0.5m）
    
    # ロボット（真値）の移動
    true_pos += u + random.gauss(0, 0.05)
    
    # 各パーティクルの移動（ノイズを加えることで不確かさを表現）
    for i in range(NUM_PARTICLES):
        particles[i] += u + random.gauss(0, sigma_move)

    # --- 3. 観測後のセンサ値の反映 (Sensor Update) ---
    # 教科書 5.3.1項：センサ値によるパーティクルの評価
    z = abs(LANDMARK_POS - true_pos) + random.gauss(0, 0.1) # 実際の観測値
    
    weights = []
    for p in particles:
        # 教科書 5.3.2項：パーティクルの重み（尤度計算）
        # 予測される観測値
        z_hat = abs(LANDMARK_POS - p)
        # 教科書 5.3.3項：尤度関数の実装（ガウス分布）
        error = z - z_hat
        likelihood = (1.0 / math.sqrt(2.0 * math.pi * sigma_sensor**2)) * \
                     math.exp(- (error**2) / (2.0 * sigma_sensor**2))
        weights.append(likelihood)

    # --- 4. リサンプリング (Resampling) ---
    # 教科書 5.4.2項：単純なリサンプリング（系統サンプリングの簡易版）
    if sum(weights) > 0:
        # 重みに比例した確率でパーティクルを次世代にコピーする
        particles = random.choices(particles, weights=weights, k=NUM_PARTICLES)

    # --- 5. 描画 (y軸は常に0の1次元表示) ---
    # ランドマーク (点)
    ax.scatter([LANDMARK_POS], [0], color='green', s=200, marker='*', label='Landmark')
    # パーティクル
    ax.scatter(particles, [0] * NUM_PARTICLES, color='blue', alpha=0.3, s=20, label='Particles')
    # 真のロボット
    ax.scatter([true_pos], [0], color='red', s=100, label='Robot(True)', marker='x', zorder=5)
    
    ax.set_xlim(0, WORLD_SIZE)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.legend(loc='upper right')
    ax.set_title(f"MCL Step {frame}: Moving and Sensing Point Landmark")

ani = animation.FuncAnimation(fig, update, frames=30, interval=300, repeat=False)
ani.save('mcl_result.gif', writer='pillow')
plt.show()