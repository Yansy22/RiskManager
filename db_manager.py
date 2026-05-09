import sqlite3
import datetime

class RiskManagerDB:
    def __init__(self, db_path="riskmanager.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 포트폴리오 요약 (week_id를 PRIMARY KEY로)
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_logs (
                week_id TEXT PRIMARY KEY,
                timestamp TEXT,
                total_aum REAL,
                cash_weight REAL,
                portfolio_expected_ret REAL,
                monte_carlo_mdd REAL
            )
            ''')
            
            # 종목별 상세 (week_id와 ticker를 복합 기본키로)
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS asset_logs (
                week_id TEXT,
                timestamp TEXT,
                ticker TEXT,
                target_weight REAL,
                expected_return REAL,
                implied_equilibrium_ret REAL,
                user_view REAL,
                view_confidence REAL,
                cluster_id INTEGER,
                trade_action TEXT,
                trade_qty REAL,
                execution_price REAL,
                PRIMARY KEY (week_id, ticker),
                FOREIGN KEY (week_id) REFERENCES portfolio_logs(week_id)
            )
            ''')
            conn.commit()

    def upsert_portfolio_log(self, week_id, total_aum=None, cash_weight=None, portfolio_expected_ret=None, monte_carlo_mdd=None, timestamp=None):
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # INSERT OR REPLACE 사용
            cursor.execute('''
            INSERT OR REPLACE INTO portfolio_logs (week_id, timestamp, total_aum, cash_weight, portfolio_expected_ret, monte_carlo_mdd)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (week_id, timestamp, total_aum, cash_weight, portfolio_expected_ret, monte_carlo_mdd))
            conn.commit()

    def delete_asset_logs(self, week_id):
        # 주 단위 덮어쓰기를 위해 기존 asset_logs 삭제
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM asset_logs WHERE week_id = ?', (week_id,))
            conn.commit()

    def insert_asset_log(self, week_id, ticker, target_weight=None, expected_return=None, 
                         implied_equilibrium_ret=None, user_view=None, view_confidence=None, 
                         cluster_id=None, trade_action=None, trade_qty=None, execution_price=None, timestamp=None):
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO asset_logs (
                week_id, timestamp, ticker, target_weight, expected_return, 
                implied_equilibrium_ret, user_view, view_confidence, cluster_id, 
                trade_action, trade_qty, execution_price
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (week_id, timestamp, ticker, target_weight, expected_return, 
                  implied_equilibrium_ret, user_view, view_confidence, cluster_id, 
                  trade_action, trade_qty, execution_price))
            conn.commit()
