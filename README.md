# AI Meeting Intelligence & Task Management System

Hệ thống thông minh hỗ trợ doanh nghiệp tải lên tệp ghi âm cuộc họp, chuyển giọng nói tiếng Việt sang văn bản (`PhoWhisper`), phân tách & nhận diện người nói (`Speaker Diarization & Voiceprint`), tổng hợp biên bản và trích xuất công việc (`Action Items`) qua `Gemini API`, lưu trữ trên **PostgreSQL 17** và đóng gói hoàn chỉnh bằng **Docker Compose + Adminer**.

---

## Kiến trúc & Cây thư mục dự án

```text
AI Meeting Intelligence/
├── .vscode/
│   └── settings.json                # Ẩn bytecode/cache trong IDE & cấu hình Python .venv
├── config/
│   └── settings.py                  # Cấu hình đường dẫn & biến môi trường tập trung
├── data/
│   ├── samples/
│   │   └── meeting_test.wav         # Tệp âm thanh mẫu chuẩn 1 phút (16kHz Mono WAV)
│   ├── uploads/
│   │   └── .gitkeep                 # Thư mục lưu tệp ghi âm cuộc họp tải lên
│   └── voiceprints/
│       └── .gitkeep                 # Thư mục lưu mẫu giọng nói nhân sự
├── notebooks/
│   ├── day3_speaker_diarization.ipynb
│   └── results/                     # Dữ liệu chuẩn đối chiếu Diarization & Alignment
│       ├── aligned_transcript.json
│       └── diarization_result.json
├── src/
│   ├── ai/                          # Tầng AI Core (Độc lập với giao diện & DB)
│   │   ├── llm/
│   │   │   └── analyzer.py          # Tóm tắt biên bản & trích xuất công việc (Gemini)
│   │   ├── transcription/
│   │   │   └── phowhisper.py        # Nhận diện giọng nói tiếng Việt (PhoWhisper ASR)
│   │   ├── alignment.py             # Đồng bộ mốc thời gian Diarization & ASR
│   │   └── voiceprint.py            # Trích xuất đặc trưng & đối sánh giọng nói
│   ├── db/                          # Tầng Cơ sở dữ liệu (PostgreSQL 17 / SQLite fallback)
│   │   ├── connection.py            # PostgresConnectionWrapper, Migration & Admin Seeding
│   │   └── repository.py            # MeetingRepository, TaskRepository, SpeakerRepository
│   ├── schemas/                     # Hợp đồng dữ liệu Pydantic
│   │   ├── meeting.py               # MeetingAnalysis, TranscriptSegment, SpeakerProfile
│   │   └── task.py                  # TaskItem, TaskStatus, TaskConfirmationStatus
│   └── services/                    # Tầng Nghiệp vụ (Service Layer)
│       └── meeting_service.py       # Điều phối xử lý âm thanh, AI và Repository
├── tests/                           # Bộ kiểm thử tự động (Dùng Test DB cô lập)
│   ├── test_alignment.py
│   ├── test_asr_integration.py
│   ├── test_database_crud.py
│   └── test_llm_and_hitl.py
├── web/                             # Giao diện Web
│   ├── css/
│   │   └── styles.css
│   ├── js/
│   │   ├── admin.js                 # Logic trang Quản trị Hệ thống (/admin)
│   │   └── app.js                   # Logic trang Biên bản & Công việc (/)
│   ├── admin.html
│   └── index.html
├── .dockerignore                    # Loại trừ thư mục phát triển khi build Docker Image
├── .env.example                     # Mẫu cấu hình biến môi trường Production
├── .gitignore                       # Quy tắc loại trừ dữ liệu nhạy cảm & cache
├── Dockerfile                       # Build FastAPI AI Server (Python 3.10 + FFmpeg)
├── docker-compose.yml               # 3 dịch vụ: PostgreSQL 17, Adminer & FastAPI App
├── requirements.txt                 # Thư viện Python phụ thuộc
└── server.py                        # Máy chủ FastAPI Backend & Static Web Server
```

---

## Hướng dẫn triển khai & Khởi chạy

### Cách 1: Chạy bằng Docker Compose (Khuyến nghị — PostgreSQL 17 + Adminer)

1. Sao chép file biến môi trường và điền `GEMINI_API_KEY`:
```powershell
Copy-Item .env.example .env
```
2. Khởi chạy toàn bộ 3 dịch vụ (`db`, `adminer`, `app`):
```powershell
docker compose up -d --build
```
- **Ứng dụng chính (Workspace):** `http://localhost:8000`
- **Trang Quản trị Hệ thống:** `http://localhost:8000/admin` (Tài khoản mặc định: `nguyenhoang` / `123456`)
- **Quản trị CSDL Adminer:** `http://localhost:8080` (Server: `db`, User: `aimeeting`, Password: `aimeeting123`, DB: `aimeeting_db`)

### Cách 2: Chạy trực tiếp trên máy cục bộ (Local Development)

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python server.py
```
