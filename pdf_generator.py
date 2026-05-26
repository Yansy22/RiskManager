import os
from fpdf import FPDF
import datetime

class AIReportPDF(FPDF):
    def header(self):
        # 폰트가 이미 등록되어 있어야 함
        try:
            self.set_font("Malgun", "B", 10)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, "AI Portfolio Guardian Strategy Report", 0, 1, "R")
            self.ln(5)
        except:
            pass

    def footer(self):
        try:
            self.set_y(-15)
            self.set_font("Malgun", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"Page {self.page_no()} | AI Portfolio Guardian", 0, 0, "C")
        except:
            pass

def generate_pdf_report(markdown_text: str, output_path: str):
    pdf = AIReportPDF()
    
    # 여백 조정 (20mm -> 15mm)
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.set_top_margin(15)
    
    # 폰트 등록
    font_path = r"C:\Windows\Fonts\malgun.ttf"
    font_bold_path = r"C:\Windows\Fonts\malgunbd.ttf"
    
    pdf.add_font("Malgun", "", font_path)
    pdf.add_font("Malgun", "B", font_bold_path)
    pdf.add_font("Malgun", "I", font_path)
    
    pdf.add_page()
    page_width = pdf.w - 2 * pdf.l_margin
    
    lines = markdown_text.split("\n")
    
    # 테이블 데이터를 모으기 위한 임시 변수
    in_table = False
    table_data = []

    for line in lines:
        line = line.strip()
        
        # 테이블 처리 로직 개선
        if line.startswith("|"):
            if "---" in line: 
                in_table = True
                continue
            cols = [c.strip() for c in line.split("|") if c.strip()]
            if cols:
                table_data.append(cols)
            continue
        else:
            # 테이블이 끝나면 한꺼번에 출력
            if in_table and table_data:
                pdf.set_font("Malgun", "", 9)
                # fpdf2의 내장 테이블 기능 사용 (가장 확실한 해결책)
                with pdf.table(
                    borders_layout="ALL",
                    cell_fill_color=245,
                    cell_fill_mode="ROWS",
                    line_height=6,
                    text_align="CENTER",
                    width=page_width
                ) as table:
                    for row in table_data:
                        row_cells = table.row()
                        for item in row:
                            row_cells.cell(item)
                
                table_data = []
                in_table = False
                pdf.ln(5)

        if not line:
            pdf.ln(5)
            continue
            
        try:
            # 모든 출력 전 커서 위치와 정렬을 강제로 리셋 (오른쪽 짤림 방지)
            pdf.set_x(pdf.l_margin)
            
            # 마크다운 파싱
            if line.startswith("# "):
                pdf.set_font("Malgun", "B", 20)
                pdf.set_text_color(0, 50, 150)
                pdf.multi_cell(page_width, 15, line[2:], align="L")
                pdf.set_text_color(0, 0, 0)
            elif line.startswith("## "):
                pdf.set_font("Malgun", "B", 16)
                pdf.set_text_color(50, 50, 50)
                pdf.ln(5)
                pdf.set_x(pdf.l_margin) # 다시 한 번 리셋
                pdf.multi_cell(page_width, 12, line[3:], align="L")
                pdf.set_text_color(0, 0, 0)
            elif line.startswith("### "):
                pdf.set_font("Malgun", "B", 14)
                pdf.ln(3)
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(page_width, 10, line[4:], align="L")
            elif line.startswith("![") and "](" in line:
                try:
                    parts = line.split("](")
                    img_path = parts[1].replace(")", "").strip()
                    if os.path.exists(img_path):
                        pdf.ln(5)
                        img_w = 150  # mm width
                        x_pos = (pdf.w - img_w) / 2
                        pdf.image(img_path, x=x_pos, w=img_w)
                        pdf.ln(5)
                    continue
                except Exception as img_err:
                    print(f"      [PDF Warning] Image embedding failed: {img_err}")
                    continue
            else:
                pdf.set_font("Malgun", "", 11)
                # 불필요한 특수문자 제거 및 줄바꿈
                clean_line = line.replace("**", "").replace("* ", "• ").strip()
                if clean_line:
                    pdf.multi_cell(page_width, 7, clean_line, align="L")
        except Exception as e:
            continue

    pdf.output(output_path)
    print(f"      [PDF] 전략 리포트가 생성되었습니다: {output_path}")
    return output_path
