"""
Module: prompts.py
Duong dan: src/core/ai/llm/prompts.py
Muc dich: Chua cac mau System Prompt, dinh dang luot thoai va chen thong tin tu sua loi
(self-correction feedback) cho Gemini / LLM.
"""

from typing import List, Optional

from src.schemas.meeting import TranscriptSegment


def build_system_prompt() -> str:
    """Tao System Prompt tieng Viet chat che cho LLM."""
    return (
        "Ban la chuyen gia phan tich bien ban cuoc hop doanh nghiep tieng Viet.\n"
        "Nhiem vu cua ban la doc danh sach cac luot phat bieu theo dinh dang:\n"
        "[Turn #i | start_time - end_time | speaker_id (speaker_name)]: text\n\n"
        "Hay tra ve DUY NHAT mot doi tuong JSON hop le theo cau truc sau:\n"
        "{\n"
        '  "cleaned_transcript": "Van ban hoi thoai da sua loi chinh ta dong am ASR '
        "(vi du: 'hop day' -> 'hop de', 'tai phai' -> 'tai file', 'mo hinh ngoi do' -> "
        "'mo hinh moi, no', 'luat noi' -> 'luot noi', 'hien hang' -> 'han', "
        "'chuot' -> 'truoc'), giu dung y nghia goc, khong bia them thong tin\",\n"
        '  "summary": "- **Tiến độ giao diện:** Hoàn thành giao diện, đang kiểm tra tải tệp và lưu biên bản.\\n- **Chủ đề ngắn gọn 2:** Nội dung chính ngắn gọn.\\n- **Chủ đề ngắn gọn 3:** Nội dung chính ngắn gọn.",\n'
        '  "key_decisions": ["Quyet dinh 1", "Quyet dinh 2"],\n'
        '  "tasks": [\n'
        "    {\n"
        '      "title": "Ten cong viec cu the",\n'
        '      "description": "Mo ta chi tiet yeu cau",\n'
        '      "assigner": "Ten nguoi giao viec hoac Chưa xác định",\n'
        '      "assignee": "Ten nguoi nhan viec hoac Chưa chỉ định",\n'
        '      "due_date": "Han chot neu co hoac null",\n'
        '      "priority": "Thấp | Trung bình | Cao | Khẩn cấp",\n'
        '      "priority_reason": "Ly do danh gia muc do uu tien",\n'
        '      "source_quote": "Cau noi goc nguyen van trong luot thoai",\n'
        '      "source_start_time": 0.0,\n'
        '      "source_end_time": 0.0\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "QUY TAC BAT BUOC:\n"
        "1. Dinh dang truong 'summary' (van la 1 chuoi string duy nhat):\n"
        "   - Viet thanh 3-5 y ngan gon, moi y nam tren mot dong theo dung mau: '- **Chủ đề ngắn gọn:** Nội dung chính ngắn gọn.' (phan tach cac dong bang '\\n').\n"
        "   - KHONG neu ten rieng tung ca nhan hay viet kieu 'Anh A bao cao..., anh B yeu cau...'. "
        "Viet trung lap, khach quan theo noi dung, tien do va van de chinh da ban trong cuoc hop.\n"
        "   - Khong lap nguyen van phan 'key_decisions' (Ket luan chinh); tap trung khai quat nhung viec chinh da trao doi.\n"
        "2. Giu nguyen logic tach 'key_decisions' va 'tasks' (phan 'tasks' van giu ro ten nguoi giao assigner va nguoi thuc hien assignee).\n"
        "3. Tuyet doi khong suy dien hoac bia cong viec neu khong co can cu trong hoi thoai.\n"
        "4. Moi task bat buoc phai trich dan dung source_quote va moc thoi gian cua Turn goc.\n"
        "5. Tach rieng tung cam ket hoac yeu cau cong viec o cac luot thoai (Turn) khac nhau thanh cac task doc lap (khong gop chung nhieu dau viec vao 1 task)."
    )


def build_formatted_turns(segments: List[TranscriptSegment]) -> str:
    """Dinh dang danh sach TranscriptSegment thanh chuoi van ban co cau truc."""
    lines: List[str] = []
    for idx, seg in enumerate(segments, start=1):
        start_str = f"{seg.start_time:.2f}" if seg.start_time is not None else "None"
        end_str = f"{seg.end_time:.2f}" if seg.end_time is not None else "None"
        spk_id = seg.speaker_id or "SPEAKER_UNASSIGNED"
        spk_name = seg.speaker_name or "Chưa phân vai"
        lines.append(
            f"[Turn #{idx} | {start_str} - {end_str} | {spk_id} ({spk_name})]: {seg.text}"
        )
    return "\n".join(lines)


def build_user_prompt_with_feedback(
    base_prompt: str,
    valid_names: Optional[List[str]] = None,
    validation_errors: Optional[List[str]] = None,
) -> str:
    """Bo sung danh sach nhan su hop le va danh sach loi validation de LLM tu sua o vong retry."""
    user_prompt = base_prompt
    if valid_names:
        user_prompt = (
            f"{user_prompt}\n\n"
            f"DANH SACH NHAN SU HOP LE TRONG HE THONG: {', '.join(valid_names)}\n"
            "(Khi gan assignee/assigner cho cong viec, uu tien su dung dung ten nhan su trong danh sach tren hoac 'Chưa chỉ định')."
        )

    if validation_errors:
        errors_block = "\n".join(f"- {err}" for err in validation_errors)
        user_prompt = (
            f"{user_prompt}\n\n"
            "=== CANH BAO LOI TU VONG KIEM TRA TRUOC (YEU CAU TU SUA) ===\n"
            "Ket qua JSON o lan sinh truoc gap cac loi sau:\n"
            f"{errors_block}\n"
            "Hay sua lai toan bo cac loi tren va tra ve JSON hop le, day du summary va dung ten nhan su."
        )
    return user_prompt
