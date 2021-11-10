# Google trends data for automated stock trading using Reinforcement learning
This project is part of my internship at ULiege on Deep RL in stock market trading with Professor [Damien Ernst](https://scholar.google.com/citations?user=91ZxYSsAAAAJ&hl=en) . Here I am validating the effectiveness of google trends data for an automated stock trading agent using the [FinRL library](https://github.com/AI4Finance-Foundation/FinRL).

If you want to change the ticker symbol and name for trends data, you can do it from the train_tune.py file by uncommenting the required timeframe and ticker symbol. Also, here is the link to my [Medium article](https://athekunal.medium.com/google-trends-data-for-automated-stock-trading-using-reinforcement-learning-9c0fd6d00678)

Start with cloning the repository
```
git clone https://github.com/Athe-kunal/Reinforcement-learning-trading-agent-using-Google-trends-data.git
```

The jump to the directory
```
cd Reinforcement-learning-trading-agent-using-Google-trends-data/
```

Install all the dependences
```
pip install -r requirements.txt
```
Then if you want to download the pytrends data apart from what is present inside the Pytrends folder
First do it to get help of CLI in downloading trends data
```
python pytrends_daily.py -h
```
Example for Apple to download data for month of October in 2021
```
python pytrends_daily.py -n 'Apple' -say 2021 -sam 10 -soy 2021 -som 10 -c 0
```

Now you can run different cases and save results in your Account value folder. Also uncomment the dates that you want for testing in ```train_tune.py```

```
python main.py -n 'Amazon' -t 'AMZN'
```
Note: The default environment is for a single stock trading only
