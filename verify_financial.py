import sys
import os
from pprint import pprint

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from config import load_config
from kucoin import KucoinClient

def test_accounts():
    cfg = load_config()
    client = KucoinClient(cfg)
    
    print("--- Fetching Accounts ---")
    
    print("\nTrade (Spot) Accounts:")
    trade_accts = client.get_trade_accounts()
    pprint([vars(a) for a in trade_accts if float(a.balance) > 0 or a.currency == "USDT"])
    
    print("\nFinancial (Pool-X) Accounts:")
    try:
        fin_accts = client.get_financial_accounts()
        pprint([vars(a) for a in fin_accts if float(a.balance) > 0 or a.currency == "USDT"])
    except Exception as e:
        print(f"Error fetching financial accounts: {e}")

    print("\nMain Accounts Extracts:")
    all_accts = client.get_accounts("main")
    pprint([vars(a) for a in all_accts if float(a.balance) > 0 or a.currency == "USDT"])

if __name__ == "__main__":
    test_accounts()
