import sys
import time
import multiprocessing as mp
import numpy as np
import cv2
import gym
from enum import Enum
from gym.vector.async_vector_env import AsyncVectorEnv
from gym.error import AlreadyPendingCallError, NoAsyncCallError
from gym.vector.utils import write_to_shared_memory
 
# --- PHẦN ĐỊNH NGHĨA CLASS (Giữ nguyên logic của bạn, sắp xếp lại vị trí) ---
 
class AsyncState(Enum):
    DEFAULT = 'default'
    WAITING_RESET = 'reset'
    WAITING_STEP = 'step'
    WAITING_RENDER = 'render'
 
class RenderGUI(gym.Wrapper):
    def __init__(
        self,
        env,
        window_name="Snake AI",
        save_video=False,
        video_path="output.mp4",
        fps=20
    ):
        super().__init__(env)
        self.window_name = window_name
        self.render_size = 30
        self.window_initialized = False

        # ---- VIDEO ----
        self.save_video = save_video
        self.video_path = video_path
        self.fps = fps
        self.video_writer = None

    def render(self):
        img_rgb = self.env.render_fancy(cell_size=self.render_size)
        if img_rgb is None:
            return None

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # ---- INIT WINDOW ----
        if not self.window_initialized:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                self.window_name,
                img_bgr.shape[1],
                img_bgr.shape[0]
            )
            self.window_initialized = True

        cv2.imshow(self.window_name, img_bgr)
        cv2.waitKey(1)

        # ---- INIT VIDEO WRITER ----
        if self.save_video and self.video_writer is None:
            h, w, _ = img_bgr.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                self.video_path,
                fourcc,
                self.fps,
                (w, h)
            )

        # ---- WRITE FRAME ----
        if self.save_video and self.video_writer is not None:
            self.video_writer.write(img_bgr)

        return img_rgb

    def close(self):
        if self.video_writer is not None:
            self.video_writer.release()
        if self.window_initialized:
            cv2.destroyWindow(self.window_name)
        super().close()
        
class SingleAgent(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.num_snakes == 1, "Number of player must be one"
        self.action_space = gym.spaces.Discrete(len(self.env.action_dict))
        # Logic Observation space...
        if getattr(self.env, 'vision_range', None):
            h = w = self.env.vision_range * 2 + 1
            shape = (h, w, self.env.obs_ch)
        else:
            shape = (*self.env.grid_shape, self.env.obs_ch)
       
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8)
 
    def reset(self, **kwargs):
        wrapped_obs = self.env.reset(**kwargs)
        return wrapped_obs[0]
 
    def step(self, action, **kwargs):
        obs, rews, dones, infos = self.env.step([action], **kwargs)
        return obs[0], rews[0], dones[0], {}
 
class SingleMultiAgent(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Lưu ý: Action space ở đây khai báo là Discrete(N) nhưng thực tế env cần List actions
        self.action_space = gym.spaces.Discrete(len(self.env.action_dict))
       
        # Sửa logic lấy obs shape để tránh lỗi nếu attribute không tồn tại
        vision_range = getattr(self.env, 'vision_range', None)
        obs_ch = getattr(self.env, 'obs_ch', 3) # Mặc định 3 kênh màu
       
        if vision_range:
            h = w = vision_range * 2 + 1
            shape = (self.env.num_snakes, h, w, obs_ch)
        else:
            shape = (self.env.num_snakes, *self.env.grid_shape, obs_ch)
           
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8)
 
def _worker_shared_memory(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == 'reset':
                observation = env.reset()
                write_to_shared_memory(index, observation, shared_memory, observation_space)
                pipe.send((None, True))
            elif command == 'step':
                observation, reward, done, info = env.step(data)
                # Auto-reset logic
                all_done = done if isinstance(done, bool) else all(done)
                if all_done:
                    observation = env.reset()
               
                write_to_shared_memory(index, observation, shared_memory, observation_space)
                pipe.send(((None, reward, done, info), True))
            elif command == 'close':
                pipe.send((None, True))
                break
            elif command == 'render':
                # Render trả về ảnh RGB
                img = env.render(mode='rgb_array')
                pipe.send((img, True))
            # ... (các command khác)
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()
 
class AsyncVectorMultiEnv(AsyncVectorEnv):
    def __init__(self, env_fns, **kwargs):
        super().__init__(env_fns, worker=_worker_shared_memory, **kwargs)
        self.default_state = None
 
    def render_async(self):
        self._assert_is_running()
        self.parent_pipes[0].send(('render', None))
        self._state = AsyncState.WAITING_RENDER
 
    def render_wait(self, timeout=None):
        self._assert_is_running()
        if not self._poll(timeout):
            raise mp.TimeoutError('Render timed out')
       
        results = []
        # Thu thập kết quả từ tất cả worker (không chỉ pipe[0])
        # AsyncVectorEnv gốc của gym xử lý việc gửi lệnh cho tất cả pipes trong step_async
        # Nhưng ở đây ta đang gửi thủ công. Để đơn giản, ta chỉ lấy từ pipe 0 hoặc phải loop send.
        # Sửa nhanh: Gửi lệnh render cho TẤT CẢ workers
        for pipe in self.parent_pipes:
            pipe.send(('render', None))
           
        for pipe in self.parent_pipes:
            result, success = pipe.recv()
            if success: results.append(result)
           
        self._state = AsyncState.DEFAULT # Reset state
        return results # Trả về list các ảnh
 
    def render(self, mode='human'):
        # Override render chuẩn
        self.render_async() # Lưu ý: Logic gốc class này hơi sai ở render_async chỉ gửi pipe[0]
        return self.render_wait()
 
# Di chuyển hàm tạo env ra ngoài để multiprocessing có thể pickle được
def _single_env_factory(env_id, num_snakes, env_wrapper, **kwargs):
    import gym
    import marlenv # Đảm bảo import bên trong worker
    env = gym.make(env_id, num_snakes=num_snakes, **kwargs)
    return env_wrapper(env)
 
def make_snake(num_envs=1, num_snakes=4, env_id="Snake-v1", **kwargs):
    env_wrapper = SingleMultiAgent if num_snakes > 1 else SingleAgent
 
    def make_env():
        import gym, marlenv
        env = gym.make(env_id, num_snakes=num_snakes, **kwargs)
        return env_wrapper(env)
 
    if num_envs > 1:
        env = AsyncVectorEnv([make_env for _ in range(num_envs)])
    else:
        env = make_env()
 
    dummy = make_env()
    properties = {
        'action_info': {'action_n': dummy.action_space.n},
        'num_envs': num_envs,
        'num_snakes': num_snakes
    }
    return env, None, None, properties