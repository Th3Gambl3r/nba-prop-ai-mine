import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams
import os
from datetime import datetime
import joblib
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import numpy as np

print(f"[{datetime.now()}] Starting daily NBA data + model update...")

# Download all games from 2015 to today
def refresh_boxscores():
    all_games = []
    for team in teams.get_teams():
        print(f"Fetching {team['abbreviation']}")
        gf = leaguegamefinder.LeagueGameFinder(team_id_nullable=team['id'])
        games = gf.get_data_frames()[0]
        all_games.append(games)
    df = pd.concat(all_games).drop_duplicates(subset=['GAME_ID'])
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df.to_parquet("data/nba_boxscores.parquet")
    print("Boxscores updated!")

refresh_boxscores()

# Rebuild and retrain models (fast after first time)
from nba_api.stats.endpoints import boxscoretraditionalv2

boxscores = pd.read_parquet("data/nba_boxscores.parquet")

def build_dataset(stat):
    rows = []
    for game_id in boxscores['GAME_ID'].unique()[-5000:]:  # Last ~3 seasons fast
        try:
            box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id).get_data_frames()[0]
            game_row = boxscores[boxscores['GAME_ID'] == game_id].iloc[0]
            date = game_row['GAME_DATE']
            opp = game_row['MATCHUP'].split()[-1]

            for _, row in box.iterrows():
                if row['MIN'] is None or float(row['MIN'].split(':')[0]) < 10: continue
                player = row['PLAYER_NAME']
                target = row[stat]
                line = np.percentile(box[stat], 60)  # Approximate Vegas line

                recent = boxscores[(boxscores['PLAYER_NAME'] == player) & 
                                 (boxscores['GAME_DATE'] < date)].tail(15)
                if len(recent) < 5: continue

                rows.append({
                    'rolling_avg': recent[stat].mean(),
                    'rolling_std': recent[stat].std() if len(recent) > 1 else 0,
                    'vs_team_avg': recent[recent['MATCHUP'].str.contains(opp)][stat].mean(),
                    'home': 1 if row['TEAM_ABBREVIATION'] in game_row['MATCHUP'].split()[:2] else 0,
                    'days_rest': 2,
                    'line': line,
                    'over': int(target > line)
                })
        except: continue
    return pd.DataFrame(rows)

for stat, name in [("PTS","pts"), ("REB","reb"), ("FG3M","3pm")]:
    print(f"Training {name.upper()} model...")
    df = build_dataset(stat)
    if len(df) == 0: continue
    X = df[['rolling_avg','rolling_std','vs_team_avg','home','days_rest','line']]
    y = df['over']
    model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, n_jobs=-1)
    calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=3)
    calibrated.fit(X, y)
    joblib.dump(calibrated, f"models/nba_{name}_model.pkl")
    print(f"{name.upper()} model saved!")

print("Daily update complete!")
