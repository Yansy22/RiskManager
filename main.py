import os
from dotenv import load_dotenv
from workflow import build_guardian_graph

def main():
    # 환경 변수 로드 (API 키 등)
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        print("🚨 Error: GOOGLE_API_KEY가 환경 변수(.env)에 설정되지 않았습니다.")
        print(".env 파일을 만들고 GOOGLE_API_KEY=당신의_키 를 입력해주세요.")
        return

    # 워크플로우 그래프 생성
    app = build_guardian_graph()
    
    # 💡 [시나리오]
    # 현재 보유 포트폴리오: 애플 60%, 마이크로소프트 40%
    # 새로운 전략 (What-If): 기존 비중을 40%씩으로 줄이고 엔비디아를 20% 편입
    initial_state = {
        "current_portfolio": {"AAPL": 0.6, "MSFT": 0.4},
        "new_portfolio": {"AAPL": 0.4, "MSFT": 0.4, "NVDA": 0.2},
        "initial_value": 10000.0,
        "start_date": "2023-01-01",
        "end_date": "2024-04-28"
    }
    
    print("="*60)
    print("🛡️ AI Portfolio Guardian 시스템을 시작합니다 🛡️")
    print("="*60 + "\n")
    
    # LangGraph 워크플로우 실행
    try:
        result = app.invoke(initial_state)
    except Exception as e:
        print(f"\n🚨 워크플로우 실행 중 오류가 발생했습니다: {e}")
        return
        
    print("\n" + "="*60)
    print("📄 AI 가디언 최종 리스크 진단 리포트")
    print("="*60 + "\n")
    print(result["final_report"])

if __name__ == "__main__":
    main()
