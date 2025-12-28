# Hướng dẫn cài đặt môi trường MARL-Snake

**1. Clone repository**
```
git clone https://github.com/tranthai189765/MARL-Snake.git
```
**2. Di chuyển vào thư mục môi trường**
```
cd marlenv
```
**3. Cài đặt môi trường (Python 3.10)**
```
pip install -e . --use-pep517
```
**4. Cài đặt phiên bản Gym tương thích**
```
pip install gym==0.23.1
```
**5. Kiểm tra cài đặt**

Sau khi hoàn tất, chạy file test_env.py để kiểm tra môi trường hoạt động đúng.

**Ghi chú**

Nếu bạn muốn cài lại môi trường (ví dụ sau khi update render hoặc code mới), chỉ cần thực hiện:
```
pip uninstall marlenv
```

Sau đó làm lại các bước 2 → 4 ở trên.

**Lưu ý**

Phần render sẽ được Thái cập nhật để hiển thị đẹp và trực quan hơn trong thời gian tới nhé. 

**Có vấn đề gì nhắn Thái nhé mng!**


# Luật chơi (Rules)

Nhiều rắn (snakes) sẽ chiến đấu trên một bản đồ dạng lưới có kích thước cố định.

Mỗi con rắn được sinh ra tại một vị trí ngẫu nhiên với hướng di chuyển ban đầu ngẫu nhiên khi gọi reset().

Bản đồ có thể được khởi tạo với các bức tường khác nhau tùy vào cài đặt môi trường.

Rắn sẽ chết nếu đầu của nó va chạm với tường hoặc thân của rắn khác.

Rắn gây ra cái chết sẽ nhận thưởng “kill”,

Rắn chết sẽ nhận phạt “lose”.

Nếu nhiều rắn va đầu vào nhau cùng lúc, tất cả đều chết mà không nhận điểm kill.

Khi chỉ còn một rắn sống sót, nó sẽ nhận thưởng “win” cho mỗi đơn vị thời gian sống sót thêm.

Rắn lớn thêm 1 pixel khi ăn được trái cây (fruit).

**Dạng quan sát (Observation Types)**

Quan sát được thể hiện dưới dạng lưới hình ảnh (image grid) theo thứ tự NHWC. (Batch_size - Height - Width - Channel)

# Ví dụ khởi tạo môi trường
```
import gym
import marlenv
from marlenv.marlenv.wrappers import make_snake, RenderGUI
env, obs_shape, action_shape, properties = make_snake(
    num_envs=1,      # Số lượng môi trường
    height=20,       # Chiều cao bản đồ
    width=20,        # Chiều rộng bản đồ
    num_snakes=4,    # Số lượng rắn trên bản đồ
    snake_length=3,  # Độ dài khởi tạo của rắn
    vision_range=5,  # Tầm nhìn (nếu None thì trả về toàn bản đồ)
    frame_stack=1,   # Số lượng khung quan sát được stack lại
)

env = RenderGUI(env)
```

**Các giá trị trả về gồm:**

env: đối tượng môi trường

observation_space: không gian quan sát đã được xử lý

action_space: không gian hành động

properties: dict chứa thông tin:

+ high: giá trị quan sát tối đa

+ low: giá trị quan sát tối thiểu

+ num_envs: số lượng môi trường

+ num_snakes: số rắn được sinh ra

+ discrete: True nếu không gian hành động là rời rạc

+ action_info: {action_high, action_low} nếu là liên tục hoặc {action_n} nếu là rời rạc

**Lưu ý: Để render được môi trường thì mng cần bọc env như này nhé, cái này Thái mới thêm:** 
```
env = RenderGUI(env)
```

# Hàm thưởng tùy chỉnh (Custom Reward Function)

Người dùng có thể tùy chỉnh cấu trúc hàm thưởng khi khởi tạo môi trường.

Ví dụ:
```
custom_reward_func = {
    'fruit': 1.0,  # Thưởng khi ăn trái cây
    'kill': 0.0,   # Thưởng khi giết rắn khác
    'lose': 0.0,   # Phạt khi chết
    'time': 0.0,   # Thưởng theo thời gian sống sót
    'win': 0.0     # Thưởng khi là rắn cuối cùng còn sống
}

env = gym.make('snake-v1', reward_func=custom_reward_func)
```

Mỗi phần thưởng có thể là số thực dương hoặc âm, tùy ý người dùng.




**Render hiện tại:**

<img width="619" height="677" alt="image" src="https://github.com/user-attachments/assets/24cb4833-27b2-4b07-bd41-8cec943e6f7f" />

---

# DQN Training

Baseline Deep Q-Learning (DQN) cho môi trường Multi-Agent Snake. Mỗi agent (rắn) có một mạng DQN riêng và học độc lập (Independent Q-Learning).

## Cài đặt thêm

```bash
pip install torch numpy
```

## Các lệnh chạy

### Training

```bash
# Training mới từ đầu
python train_dqn.py --mode train --episodes 5000

# Resume training từ checkpoint (preset mặc định)
python train_dqn.py --mode train --episodes 10000 --resume 4000

# Resume với late_training preset (giảm tự hủy đầu game)
python train_dqn.py --mode train --episodes 10000 --resume 5000 --reward-preset late_training
```

### Evaluation

```bash
# Đánh giá model tốt nhất (có render)
python train_dqn.py --mode eval --checkpoint best

# Đánh giá checkpoint cụ thể
python train_dqn.py --mode eval --checkpoint 4000 --eval-episodes 10

# Đánh giá không render
python train_dqn.py --mode eval --checkpoint best --no-render
```

### Train + Eval

```bash
python train_dqn.py --mode both --episodes 5000
```

## Tham số dòng lệnh

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--mode` | `train` | Chế độ: `train`, `eval`, hoặc `both` |
| `--episodes` | `5000` | Tổng số episodes training |
| `--resume` | `None` | Resume từ episode cụ thể |
| `--reward-preset` | `default` | Reward preset: `default` hoặc `late_training` (giảm tự hủy) |
| `--checkpoint` | `final` | Checkpoint để eval: `best`, `final`, hoặc số episode |
| `--eval-episodes` | `10` | Số episodes đánh giá |
| `--max-eval-steps` | `500` | Giới hạn steps mỗi episode khi eval |
| `--no-render` | `False` | Tắt render khi eval |

---

## Cấu hình (Config class trong train_dqn.py)

### Environment

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `NUM_SNAKES` | 4 | Số lượng rắn trong game |
| `HEIGHT` | 20 | Chiều cao bản đồ |
| `WIDTH` | 20 | Chiều rộng bản đồ |
| `SNAKE_LENGTH` | 5 | Độ dài ban đầu của rắn |
| `VISION_RANGE` | 5 | Tầm nhìn của rắn. `None` = toàn bản đồ, `5` = vùng 11x11 |

### Training Hyperparameters

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `NUM_EPISODES` | 5000 | Tổng số episodes training |
| `MAX_STEPS_PER_EPISODE` | 500 | Số bước tối đa mỗi episode |
| `BATCH_SIZE` | 64 | Kích thước batch khi sample từ buffer |
| `GAMMA` | 0.99 | Discount factor - độ quan trọng của reward tương lai |
| `LR` | 1e-4 | Learning rate của optimizer |

### Epsilon-Greedy

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `EPSILON_START` | 1.0 | Epsilon ban đầu (100% random) |
| `EPSILON_END` | 0.05 | Epsilon tối thiểu (5% random) |
| `EPSILON_DECAY` | 0.9995 | Hệ số giảm epsilon mỗi episode |

**Công thức:** `epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)`

### Replay Buffer

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `BUFFER_SIZE` | 100000 | Dung lượng tối đa của buffer |
| `MIN_BUFFER_SIZE` | 1000 | Số sample tối thiểu trước khi bắt đầu học |

### Target Network

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `TARGET_UPDATE_FREQ` | 100 | Số episodes giữa mỗi lần cập nhật target network |

### Reward Shaping

**Default Preset** (cho early training 0-5000 eps):

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `fruit` | +1.0 | Thưởng khi ăn trái cây |
| `kill` | +2.0 | Thưởng khi giết rắn khác |
| `lose` | -1.0 | Phạt khi chết |
| `win` | +0.5 | Thưởng khi thắng game |
| `time` | +0.01 | Thưởng nhỏ mỗi step sống sót |

**Late Training Preset** (cho late training 5000+ eps, giảm tự hủy):

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `fruit` | +1.0 | Thưởng khi ăn trái cây |
| `kill` | +2.0 | Thưởng khi giết rắn khác |
| `lose` | -1.5 | Phạt chết nặng hơn |
| `win` | +0.5 | Thưởng khi thắng game |
| `time` | 0.0 | Tắt time reward - tập trung vào quality |

**Early Death Penalty:**
- `EARLY_DEATH_THRESHOLD`: 10 steps
- `EARLY_DEATH_PENALTY`: -1.0 (phạt thêm nếu chết trong 10 bước đầu)

### Checkpoint Strategy

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `SAVE_FREQ` | 500 | Lưu checkpoint định kỳ mỗi N episodes |
| `SAVE_BEST_ONLY` | True | Chỉ lưu best model khi có cải thiện |
| `KEEP_LAST_N` | 3 | Giữ lại N checkpoint gần nhất, xóa cũ |
| `SAVE_DIR` | `checkpoints` | Thư mục lưu checkpoint |
| `RESUME_FROM` | None | Episode để resume (set qua `--resume`) |

---


