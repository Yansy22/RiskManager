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

패키지 설치
pip install -r requirements.txt