**Hướng dẫn cài đặt môi trường MARL-Snake**

**1. Clone repository**
```
git clone https://github.com/tranthai189765/MARL-Snake.git
```
**2. Di chuyển vào thư mục môi trường**
```
cd marlenv
```
**3. Cài đặt môi trường**
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

Có vấn đề gì nhắn Thái nhé mng <3 !

Render hiện tại:

<img width="680" height="733" alt="image" src="https://github.com/user-attachments/assets/8187ca73-6eb7-44c5-b882-3a48f255e57c" />

