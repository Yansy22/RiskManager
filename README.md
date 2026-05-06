# 🛡️ RiskManager: AI Portfolio Guardian Roadmap

## Phase 1: 데이터 정비 및 기초 체력 다지기 (Data Ingestion)
- [ ] **portfolio.json 구조 고도화**
    - 기존 종목 정보에 `score`(1~10점), `confidence`(1~10점) 필드 추가
    - 현금 자산을 하나의 종목처럼 취급하여 비중 계산에 포함
- [ ] **데이터 소스 확장 (data_provider.py)**
    - 과거 수정 종가 데이터: 최근 1~2년치 일일 수익률 데이터 수집
    - 시가총액 데이터: 종목별 시가총액 수집 (시장 균형 비중 계산용)
    - 무위험 수익률: 현재 예금 금리나 단기 국채 금리 데이터 설정

## Phase 2: 동적 리스크 클러스터링 (Dynamic Risk Clustering)
- [ ] **상관계수 행렬(Correlation Matrix) 산출**
    - 모든 종목 쌍 간의 수익률 상관관계 계산
- [ ] **계층적 클러스터링(Hierarchical Clustering) 구현**
    - 상관계수를 거리(Distance)로 변환 ($d = \sqrt{2(1-\rho)}$)
    - Scipy를 사용하여 통계적 유사 그룹(리스크 그룹) 생성
- [ ] **외톨이 종목(Outlier) 처리**
    - 리스크 분산 효과 극대화를 위한 독자 종목 식별

## Phase 3: 블랙-리터만 기대수익률 엔진 (Black-Litterman Engine)
- [ ] **시장 균형 수익률($\Pi$) 역산**
    - 시가총액 비중과 과거 변동성을 기반으로 도출
- [ ] **사용자 견해(View) 수치화**
    - `score` -> 초과 수익률($Q$), `confidence` -> 오차 분산($\Omega$) 변환
- [ ] **베이지안 업데이트 실행**
    - 시장 수익률($\Pi$)과 사용자 견해($Q$)를 결합한 수정 기대수익률 산출

## Phase 4: 제약 조건 기반 포트폴리오 최적화 (Constrained Optimization)
- [ ] **목적 함수 설정**
    - 샤프 지수(Sharpe Ratio) 극대화 지점 탐색
- [ ] **클러스터별 비중 제한(Weight Constraints) 적용**
    - 특정 리스크 그룹 합계 비중 제한 (예: 30% 이하)
- [ ] **최종 최적 비중($w^*$) 산출**
    - 이차 계획법(Quadratic Programming)을 통한 종목별 비중 확정

## Phase 5: 몬테카를로 스트레스 테스트 (Monte Carlo Simulation)
- [ ] **미래 경로 생성**
    - Phase 3의 수익률과 Phase 2의 상관관계를 모수로 1만 개 시나리오 생성
- [ ] **리스크 지표 계산**
    - MDD (최대 낙폭) 및 VaR (Value at Risk) 산출

## Phase 6: AI 인사이트 및 리밸런싱 리포트 (Final Report)
- [ ] **데이터 전송 및 LLM 분석**
    - 현재 vs 최적 비중 및 리스크 지표를 Gemini API로 전송
- [ ] **리밸런싱 가이드 생성**
    - 구체적인 매수/매도 제안 및 실행 가이드 출력
- [ ] **의사결정 및 실행**
    - 매주 월요일 리포트 기반 포트폴리오 수정

github 업데이트 방법

1. 변경된 파일들을 장바구니에 담기 (스테이징)
git add .

2. 어떤 걸 수정했는지 짧은 메모 남기기 (커밋)
git commit -m "feat: 뭐든 바꿨겠지"

3. GitHub 서버로 보내기 (푸시)
git push origin main


github에서 가져오는 방법

1. 원격 저장소(GitHub)의 최신 변경 사항을 내 컴퓨터로 다운로드
git pull origin main


가상환경 활성화
.\venv\Scripts\activate

가상환경 비활성화
deactivate