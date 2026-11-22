import sys as _sys
from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parent.parent)
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

"""
Kịch bản kiểm thử độc lập Giai đoạn 4: Tích hợp PhoWhisper ASR vào MeetingService.
BẢO ĐẢM TOÀN VẸN DỮ LIỆU:
- Sử dụng database test riêng: data/test_meetings.db
- Kiểm tra số lượng bản ghi của database chính data/meetings.db trước và sau test để đảm bảo 100% không bị tác động.
"""
import os
import sqlite3
import time
from pathlib import Path
from config.settings import BASE_DIR, DB_PATH
from src.core.ai.asr.phowhisper import PhoWhisperASR
from src.services.meeting_service import MeetingService
from src.schemas.meeting import MeetingAnalysis, TranscriptSegment

TEST_DB_PATH = BASE_DIR / "data" / "test_meetings.db"
AUDIO_SAMPLE_PATH = BASE_DIR / "data" / "samples" / "meeting_test.wav"


def _get_db_stats(db_file: Path) -> dict:
    """Lấy số lượng bản ghi meetings và tasks của một database file."""
    if not db_file.exists():
        return {"meetings": 0, "tasks": 0, "segments": 0}
    conn = sqlite3.connect(db_file)
    try:
        m_count = conn.execute("SELECT COUNT(*) FROM meetings").fetchone()[0]
        t_count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        s_count = conn.execute("SELECT COUNT(*) FROM transcript_segments").fetchone()[0]
        return {"meetings": m_count, "tasks": t_count, "segments": s_count}
    except Exception:
        return {"meetings": 0, "tasks": 0, "segments": 0}
    finally:
        conn.close()


def run_tests():
    print("=" * 75)
    print("   KIỂM THỬ TÍCH HỢP PHOWHISPER ASR VÀO MEETINGSERVICE (TEST DB RIÊNG)")
    print("=" * 75)

    # 1. Ghi nhận trạng thái database chính trước khi test
    main_db_before = _get_db_stats(DB_PATH)
    print(f"📊 Trạng thái Database CHÍNH ({DB_PATH.name}) TRƯỚC test:")
    print(f"   - Meetings: {main_db_before['meetings']} | Tasks: {main_db_before['tasks']} | Segments: {main_db_before['segments']}")

    # 2. Chuẩn bị file database test riêng biệt
    if TEST_DB_PATH.exists():
        try:
            os.remove(TEST_DB_PATH)
        except Exception:
            pass

    # 3. Khởi tạo instance ASR dùng chung để kiểm tra dependency injection và caching
    print("\n🧪 Khởi tạo PhoWhisperASR instance và MeetingService với Test DB...")
    shared_asr = PhoWhisperASR()
    assert not shared_asr.is_loaded, "ASR instance phải chưa được tải (Lazy Loading)!"

    svc = MeetingService(db_path=TEST_DB_PATH, asr=shared_asr)
    assert svc.asr is shared_asr, "ASR instance phải được inject chính xác vào Service!"

    # 4. Bài test 1: Tạo meeting thông qua process_audio()
    print("\n--- BÀI TEST 1: GỌI PROCESS_AUDIO() VỚI FILE MEETING_TEST.WAV ---")
    start_time = time.time()
    analysis = svc.process_audio(AUDIO_SAMPLE_PATH, title="Cuộc họp kiểm thử tích hợp ASR")
    duration_1 = time.time() - start_time
    print(f"⏱️  Thời gian xử lý lần 1 (bao gồm tải model vào RAM): {duration_1:.2f}s")
    assert isinstance(analysis, MeetingAnalysis), "Kết quả trả về phải là đối tượng MeetingAnalysis!"
    assert analysis.id is not None, "Cuộc họp phải được tạo với ID hợp lệ!"
    print(f"-> Tạo thành công Meeting ID: {analysis.id}, Tiêu đề: {analysis.title}")

    # 5. Bài test 2: Transcript được lưu đúng
    print("\n--- BÀI TEST 2: KIỂM TRA NỘI DUNG RAW TRANSCRIPT ĐƯỢC LƯU VÀO DB ---")
    meeting_details = svc.get_meeting_details(analysis.id)
    assert meeting_details is not None, "Phải tìm thấy chi tiết cuộc họp trong database!"
    assert len(meeting_details["raw_transcript"].strip()) > 0, "Raw transcript không được rỗng!"
    print(f"📝 Raw Transcript trong DB: \"{meeting_details['raw_transcript']}\"")
    assert (
        "chúng ta cần hoàn thành dự án" in meeting_details["raw_transcript"]
        or "rà soát tiến độ hệ thống" in meeting_details["raw_transcript"]
    ), "Nội dung nhận diện phải chính xác!"

    # 6. Bài test 3: Transcript segments được lưu đúng
    print("\n--- BÀI TEST 3: KIỂM TRA TRANSCRIPT SEGMENTS ĐƯỢC LƯU VÀO DB ---")
    segments = meeting_details.get("segments", [])
    print(f"🗣️  Số lượng segments lưu trong DB: {len(segments)}")
    assert len(segments) > 0, "Phải có ít nhất 1 transcript segment được lưu!"
    for seg in segments:
        print(f"   [{seg['start_time']}s - {seg['end_time']}s] {seg['speaker_name']} ({seg['speaker_id']}): {seg['text']}")
        assert seg["text"].strip(), "Nội dung segment không được rỗng!"

    # 7. Bài test 4: Kiểm tra tính hợp lệ của timestamp (hoặc bảo toàn None nếu model không trả end_time)
    print("\n--- BÀI TEST 4: KIỂM TRA TIMESTAMP HỢP LỆ TỪ FASTER-WHISPER / BẢO TOÀN NONE ---")
    last_seg = segments[-1]
    print(f"   Last segment: start_time={last_seg['start_time']}, end_time={last_seg['end_time']}")
    assert all(
        s["end_time"] is None or (s["start_time"] is not None and s["end_time"] >= s["start_time"])
        for s in segments
    ), "Timestamp end_time phải là None hoặc >= start_time hợp lệ!"
    print("   ✅ Xác nhận: Timestamp của các segment được lưu chính xác trong Database.")

    # 8. Bài test 5: Không có tasks giả
    print("\n--- BÀI TEST 5: XÁC MINH KHÔNG CÓ TASKS GIẢ MẠO ĐƯỢC SINH RA ---")
    tasks = svc.get_all_tasks()
    print(f"📋 Tổng số tasks trong Test DB sau khi gọi process_audio: {len(tasks)}")
    assert len(tasks) == 0, "LỖI: Tuyệt đối không được sinh task giả mạo khi chưa có AI LangGraph!"
    print("   ✅ Xác nhận: Số tasks = 0 (Hoàn toàn trung thực, không có task giả).")

    # 9. Bài test 6: Model không bị tải lại khi gọi lần 2 (In-memory caching)
    print("\n--- BÀI TEST 6: TÁI SỬ DỤNG ASR INSTANCE (KHÔNG TẢI LẠI MODEL) ---")
    start_time_2 = time.time()
    analysis_2 = svc.process_audio(AUDIO_SAMPLE_PATH, title="Cuộc họp bóc băng lần 2 (Tái sử dụng)")
    duration_2 = time.time() - start_time_2
    print(f"⏱️  Thời gian xử lý lần 2: {duration_2:.2f}s (nhanh hơn lần 1 do model đã nằm trong RAM)")
    assert analysis_2.id != analysis.id, "Lần 2 phải tạo ra cuộc họp mới!"
    assert duration_2 < duration_1, "Thời gian lần 2 phải nhanh hơn lần 1 do không phải load model!"
    print("   ✅ Xác nhận: Model được tái sử dụng trong RAM thành công.")

    # 10. Bài test 7: Xử lý file không hợp lệ
    print("\n--- BÀI TEST 7: XỬ LÝ NGOẠI LỆ KHI FILE KHÔNG HỢP LỆ HOẶC KHÔNG TỒN TẠI ---")
    non_existent = Path("data/samples/file_ao_khong_ton_tai_999.wav")
    try:
        svc.process_audio(non_existent)
        assert False, "Phải ném FileNotFoundError khi file không tồn tại!"
    except FileNotFoundError:
        print("   ✅ Bắt đúng FileNotFoundError khi file âm thanh không tồn tại.")

    # 11. Dọn dẹp file test database
    del svc
    import gc
    gc.collect()
    if TEST_DB_PATH.exists():
        try:
            os.remove(TEST_DB_PATH)
            print(f"\n🧹 Đã dọn dẹp sạch sẽ file {TEST_DB_PATH.name} sau test.")
        except Exception:
            pass

    # 12. Bài test 8: Kiểm tra toàn vẹn Database chính
    print("\n--- BÀI TEST 8: ĐỐI CHIẾU KIỂM TRA BẢO TOÀN DATABASE CHÍNH ---")
    main_db_after = _get_db_stats(DB_PATH)
    print(f"📊 Trạng thái Database CHÍNH ({DB_PATH.name}) SAU test:")
    print(f"   - Meetings: {main_db_after['meetings']} (Trước test: {main_db_before['meetings']})")
    print(f"   - Tasks: {main_db_after['tasks']} (Trước test: {main_db_before['tasks']})")
    print(f"   - Segments: {main_db_after['segments']} (Trước test: {main_db_before['segments']})")

    assert main_db_after["meetings"] == main_db_before["meetings"], "LỖI: Database chính bị thay đổi meetings!"
    assert main_db_after["tasks"] == main_db_before["tasks"], "LỖI: Database chính bị thay đổi tasks!"
    assert main_db_after["segments"] == main_db_before["segments"], "LỖI: Database chính bị thay đổi segments!"

    print("\n" + "=" * 75)
    print(">>> 100% CÁC BÀI TEST TÍCH HỢP ĐỀU ĐẠT CHUẨN XUẤT SẮC! <<<")
    print("=" * 75)


if __name__ == "__main__":
    run_tests()

