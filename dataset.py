from torch.utils.data import Dataset
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

class PriceData(Dataset):
    def __init__(self, length):
        prices = []
        for directory in os.walk('stock_market_data/'):
            if directory[0].endswith('csv'):
                for csv in tqdm(directory[2]):
                    try:
                        df = pd.read_csv(directory[0] + '/' + csv)
                        df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
                        df = df.set_index('Date')
                        monthly_prices = df.groupby(pd.Grouper(freq='M')).mean()['Close'].values
                        if monthly_prices.mean() < 5:
                            continue
                        rel_monthly_price_changes = monthly_prices[1:]/monthly_prices[:-1] - 1
                        if max(rel_monthly_price_changes) > 15:
                            continue
                        monthly_price_changes = monthly_prices[1:] - monthly_prices[:-1]

                        data_length = len(monthly_prices)-1
                        assert data_length>length
                        start_idx = np.random.randint(0, length)

                        for i in range((data_length-length-start_idx)//length):
                            interval_prices = monthly_price_changes[start_idx:][i*length:(i+1)*length]
                            scale_factor = np.max(np.abs(interval_prices))
                            assert scale_factor>0
                            interval_prices = interval_prices/scale_factor
                            prices.append(interval_prices)
                    except:
                        continue

        self.x = np.array(prices)
        self.x = torch.from_numpy(np.array(prices)).float()
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx]