import argparse
from train_tune import train_cases

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and tune for the given stokc')
    parser.add_argument('-n','--name',type=str,help='Name of the stock')
    parser.add_argument('-t','--ticker',type=str,help='Ticker symbol')
    args = parser.parse_args()

    names = [args.name]
    tickers = [args.ticker]
    
    train_cases(tickers,names)
