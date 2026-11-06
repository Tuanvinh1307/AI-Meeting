"""
AI Meeting Intelligence & Task Management System
Điểm khởi chạy chính của ứng dụng web Streamlit.
Chỉ tương tác với hệ thống thông qua Service Layer (MeetingService).
"""
import streamlit as st
import pandas as pd
from config.settings import APP_NAME, APP_VERSION
from src.services.meeting_service import MeetingService
from src.schemas.task import TaskConfirmationStatus, TaskStatus

# Khởi tạo Service Layer (Không gọi trực tiếp DB hay AI Core từ UI)
meeting_service = MeetingService()

# ==============================================================================
# Cấu hình trang (Page Configuration)
# Phải là lệnh Streamlit đầu tiên được gọi trong file!
# ==============================================================================
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Thanh điều hướng bên trái (Sidebar)
# ==============================================================================
with st.sidebar:
    st.title("⚙️ Bảng điều khiển")
    st.caption(f"Phiên bản: v{APP_VERSION} (Repository Layer Enabled)")
    st.divider()

    st.subheader("1. Chọn nguồn dữ liệu")
    input_source = st.radio(
        "Hình thức nạp nội dung cuộc họp:",
        options=["Tải lên file ghi âm", "Dán trực tiếp văn bản (Transcript)"],
        index=1,
    )

    st.divider()
    st.subheader("2. Thống kê Database (SQLite)")
    recent_meetings = meeting_service.get_recent_meetings(limit=5)
    st.success(f"📊 Đã lưu: **{len(recent_meetings)}** cuộc họp")

    all_tasks = meeting_service.get_all_tasks()
    pending_tasks = [t for t in all_tasks if t.confirmation_status == TaskConfirmationStatus.PENDING]
    confirmed_tasks = [t for t in all_tasks if t.confirmation_status == TaskConfirmationStatus.CONFIRMED]
    
    st.info(f"📋 Tổng số Task: **{len(all_tasks)}**\n- ⏳ Chờ duyệt: **{len(pending_tasks)}**\n- ✅ Đã duyệt: **{len(confirmed_tasks)}**")

    st.divider()
    st.caption("🛡️ Kiến trúc phân tầng: UI -> Service -> Repository -> SQLite")

# ==============================================================================
# Giao diện chính (Main Area)
# ==============================================================================
st.title(f"🎙️ {APP_NAME}")
st.write(
    "Hệ thống phân tích biên bản cuộc họp thông minh: Tự động bóc băng, phân vai người nói, "
    "tóm tắt quyết định và trích xuất công việc (Task) với bằng chứng cụ thể."
)

st.divider()

# Khối hiển thị theo lựa chọn của người dùng ở sidebar
if input_source == "Tải lên file ghi âm":
    st.subheader("📤 Tải lên file ghi âm cuộc họp")
    st.write("Hệ thống hỗ trợ các định dạng âm thanh phổ biến như MP3, WAV, M4A, MP4.")

    uploaded_file = st.file_uploader(
        "Chọn file âm thanh từ máy tính của bạn:",
        type=["mp3", "wav", "m4a", "mp4"],
        help="Giới hạn dung lượng tùy thuộc cấu hình hệ thống.",
    )

    if uploaded_file is not None:
        st.success(f"Đã nhận file: **{uploaded_file.name}** ({uploaded_file.size / 1024 / 1024:.2f} MB)")
        st.audio(uploaded_file, format=uploaded_file.type)
        st.info("💡 Bước tiếp theo: Tích hợp mô hình Whisper trong `src/ai/transcription/` để bóc băng file này.")
else:
    st.subheader("📝 Dán nội dung cuộc họp (Transcript)")
    meeting_title = st.text_input("Tiêu đề cuộc họp:", value="Họp rà soát tiến độ & báo cáo tài chính")
    sample_text = st.text_area(
        "Nhập nội dung biên bản cuộc họp thô vào đây:",
        placeholder="Ví dụ: Giám đốc Tuấn phát biểu: 'Anh A hoàn thành báo cáo tài chính trước thứ Sáu nhé...'",
        height=140,
        value="Trần Giám Đốc: Tuần này chúng ta cần ưu tiên hoàn thành báo cáo tài chính quý trước.\nNguyễn Văn A: Tôi sẽ phụ trách tổng hợp số liệu doanh thu và gửi trước thứ Sáu.\nTrần Thị B: Tôi sẽ gửi email biên bản họp cho cả phòng.",
    )

    if st.button("🚀 Phân tích cuộc họp", type="primary"):
        if sample_text.strip():
            with st.spinner("Đang điều phối phân tích qua Service Layer & lưu qua Repository..."):
                try:
                    result = meeting_service.process_transcript(
                        raw_transcript=sample_text,
                        title=meeting_title,
                    )
                    st.success(f"✅ Đã xử lý & lưu thành công Cuộc họp (ID: {result.id}) vào Database!")

                    # 1. Hiển thị Tóm tắt & Quyết định
                    col_sum, col_dec = st.columns([1, 1])
                    with col_sum:
                        st.subheader("📌 Tóm tắt nội dung chính")
                        st.write(result.summary)
                    with col_dec:
                        st.subheader("🎯 Quyết định quan trọng")
                        for dec in result.key_decisions:
                            st.markdown(f"- **{dec}**")

                    st.divider()

                    # 2. Hiển thị Đoạn thoại theo người nói (Transcript Segments)
                    if result.segments:
                        st.subheader("🗣️ Phân vai người phát biểu (Speaker Segments & Timestamps)")
                        for seg in result.segments:
                            with st.chat_message("user"):
                                st.markdown(f"**{seg.speaker_name}** (`{seg.speaker_id}`) &nbsp;|&nbsp; *[{seg.start_time}s - {seg.end_time}s]*")
                                st.write(seg.text)

                    st.divider()

                    # 3. Hiển thị Bảng công việc (Tasks) với đầy đủ trường mới
                    st.subheader("📋 Danh sách công việc đề xuất (Kèm bằng chứng & Trạng thái xác nhận)")
                    tasks_view = []
                    for t in result.tasks:
                        tasks_view.append({
                            "ID": t.id,
                            "Tên công việc": t.title,
                            "Người giao": t.assigner,
                            "Người nhận": t.assignee,
                            "Deadline": t.due_date,
                            "Độ ưu tiên": t.priority.value,
                            "Lý do ưu tiên": t.priority_reason,
                            "Câu nói gốc (Bằng chứng)": t.source_quote,
                            "Xác nhận AI": t.confirmation_status.value,
                            "Tiến độ": t.status.value,
                        })
                    st.dataframe(pd.DataFrame(tasks_view), use_container_width=True)

                except Exception as e:
                    st.error(f"Đã xảy ra lỗi: {str(e)}")
        else:
            st.warning("Vui lòng nhập nội dung cuộc họp trước khi bấm phân tích!")

# ==============================================================================
# Khu vực quản lý & xác nhận Task (Human-in-the-Loop & CRUD Verification)
# ==============================================================================
st.divider()
st.subheader("⚡ Quản lý & Xác nhận công việc (Human-in-the-loop Test)")

tasks_list = meeting_service.get_all_tasks()
if tasks_list:
    selected_task_id = st.selectbox(
        "Chọn Task để kiểm tra thao tác Cập nhật / Duyệt (Update CRUD):",
        options=[t.id for t in tasks_list],
        format_func=lambda tid: f"Task #{tid}: {[t.title for t in tasks_list if t.id == tid][0]} (Hiện tại: {[t.confirmation_status.value for t in tasks_list if t.id == tid][0]} | {[t.status.value for t in tasks_list if t.id == tid][0]})",
    )

    col1, col2 = st.columns(2)
    with col1:
        new_conf = st.selectbox(
            "Cập nhật Trạng thái xác nhận:",
            options=[c.value for c in TaskConfirmationStatus],
            key=f"conf_{selected_task_id}",
        )
        if st.button("Lưu trạng thái xác nhận"):
            success = meeting_service.update_task_confirmation(selected_task_id, new_conf)
            if success:
                st.success("✅ Cập nhật trạng thái xác nhận thành công!")
                st.rerun()

    with col2:
        new_stat = st.selectbox(
            "Cập nhật Tiến độ công việc:",
            options=[s.value for s in TaskStatus],
            key=f"stat_{selected_task_id}",
        )
        if st.button("Lưu tiến độ công việc"):
            success = meeting_service.update_task_status(selected_task_id, new_stat)
            if success:
                st.success("✅ Cập nhật tiến độ thành công!")
                st.rerun()

# ==============================================================================
# Bảng lộ trình phát triển (Roadmap)
# ==============================================================================
with st.expander("📌 Lộ trình các bước phát triển tiếp theo", expanded=False):
    st.markdown("""
    - [x] **Bước 1:** Khởi tạo môi trường ảo Python 3.12 & cấu trúc thư mục chuẩn.
    - [x] **Bước 2:** Xây dựng khung kiến trúc phân tầng (UI -> Service -> Repository -> SQLite & Schemas).
    - [x] **Bước 3:** Hoàn thiện Pydantic Schemas đầy đủ (Assigner, Quote, Confirmation, Segments) & Repository Layer.
    - [ ] **Bước 4:** Xây dựng đồ thị xử lý thông minh với LangGraph (Tóm tắt, Trích xuất Task, Đánh giá Priority) kết nối LLM thật.
    - [ ] **Bước 5:** Tích hợp mô hình nhận diện giọng nói (Speech-to-Text - Whisper) & Speaker Diarization.
    - [ ] **Bước 6:** Xuất báo cáo (Export ra Markdown, PDF, Excel) và chuẩn bị migration sang FastAPI.
    """)
