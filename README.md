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


