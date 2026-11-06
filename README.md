# AI Meeting Intelligence & Task Management System

Hệ thống thông minh hỗ trợ phân tích nội dung cuộc họp, bóc băng tự động (Speech-to-Text), tóm tắt thông tin quan trọng và tự động trích xuất danh sách công việc (Action Items/Tasks) kèm deadline và người phụ trách bằng AI.

##  Công nghệ sử dụng
- **Ngôn ngữ:** Python 3.12+
- **Giao diện Web:** Streamlit
- **Quản lý quy trình AI:** LangGraph
- **Mô hình ngôn ngữ & Speech-to-Text:** Whisper, Hugging Face, LLM (OpenAI / Gemini / Groq)

##  Cấu trúc dự án (Kiến trúc phân tầng - Decoupled Architecture)
- `config/`: Cấu hình hệ thống, nạp biến môi trường `.env`.
- `src/schemas/`: Định nghĩa hợp đồng dữ liệu chuẩn bằng Pydantic (`meeting.py`, `task.py`).
- `src/services/`: Tầng nghiệp vụ (`MeetingService`), điều phối giữa UI, AI Core và Database.
- `src/ai/`: Tầng AI Core (Speech-to-Text, Diarization, LangGraph Workflows), độc lập 100% với UI.
- `src/db/`: Quản lý lưu trữ SQLite (`connection.py`, bảng meetings & tasks).
- `src/ui/`: Các thành phần giao diện mở rộng cho Streamlit.
- `src/utils/`: Các hàm tiện ích hỗ trợ xuất báo cáo.
- `data/`: Dữ liệu âm thanh mẫu, exports và database `meetings.db`.

##  Hướng dẫn kích hoạt & khởi chạy ứng dụng

### 1. Kích hoạt môi trường ảo
Trên Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Cài đặt các thư viện cần thiết
```powershell
pip install -r requirements.txt
```

### 3. Khởi chạy ứng dụng Streamlit
```powershell
streamlit run app.py
```
Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ `http://localhost:8501`.

