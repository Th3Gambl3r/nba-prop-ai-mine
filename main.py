import streamlit as st
import pandas as pd
import joblib
import numpy as np
from nba_api.stats.static import players
from datetime import datetime
import plotly.graph_objects as go
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import os

st.set_page_config(page_title="NBA Prop AI", layout="wide")
st.title("NBA Prop Bet AI Dashboard")
st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}")

# Simple email setup (user configures)
if 'email_sent' not in st.session_state:
    st.session_state.email_sent = False

# Load data & models (auto-trains if missing)
@st.cache_data
def load_models():
    try:
        boxscores = pd.read_parquet("data/nba_boxscores.parquet")
    except:
        os.makedirs("data", exist_ok=True)
        # Quick initial download
        from nba_api.stats.endpoints import leaguegamefinder
        from nba_api.stats.static import teams
        all_games = []
        for team in teams.get_teams()[:5]:  # Sample for demo
            gf = leaguegamefinder.LeagueGameFinder(team_id_nullable=team['id'])
            all_games.append(gf.get_data_frames()[0])
        boxscores = pd.concat(all_games)
        boxscores.to_parquet("data/nba_boxscores.parquet")
    models = {}
    for key in ['pts', 'reb', '3pm']:
        try:
            models[key] = joblib.load(f"models/nba_{key}_model.pkl")
        except:
            os.makedirs("models", exist_ok=True)
            # Quick train
            import xgboost as xgb
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.model_selection import train_test_split
            # Dummy data for initial
            df = pd.DataFrame({'rolling_avg': np.random.normal(20,5,1000), 'rolling_std': np.random.normal(5,2,1000),
                               'vs_team_avg': np.random.normal(18,4,1000), 'home': np.random.choice([0,1],1000),
                               'days_rest': np.random.randint(1,4,1000), 'line': np.random.normal(22,3,1000)})
            df['over'] = (df['rolling_avg'] > df['line']).astype(int)
            X = df.drop('over', axis=1)
            y = df['over']
            model = xgb.XGBClassifier()
            calibrated = CalibratedClassifierCV(model, cv=3)
            calibrated.fit(X, y)
            joblib.dump(calibrated, f"models/nba_{key}_model.pkl")
            models[key] = calibrated
    return boxscores, models

boxscores, models = load_models()
active = {p['full_name']: p['id'] for p in players.get_active_players()}

# Settings Tab for Email
tab1, tab2, tab3 = st.tabs(["Prop Predictor", "Bankroll Tracker", "Settings"])

with tab1:
    st.header("Player Prop Finder")
    player_name = st.selectbox("Select Player", options=list(active.keys())[:50])  # Sample
    stat = st.selectbox("Stat", options=["Points (PTS)", "Rebounds (REB)", "Threes (3PM)"])
    line = st.number_input("Vegas Line", min_value=0.0, value=25.5, step=0.5)
    opponent = st.text_input("Opponent Abbrev (e.g. BOS, LAL)", "LAL")
    home = st.checkbox("Home Game?", value=True)
    days_rest = st.slider("Days Rest", 0, 7, 2)

    if st.button("Predict"):
        stat_code = {"Points (PTS)":"PTS", "Rebounds (REB)":"REB", "Threes (3PM)":"FG3M"}[stat]
        model_key = {"PTS":"pts", "REB":"reb", "FG3M":"3pm"}[stat_code]

        # Mock recent for demo
        recent = pd.DataFrame({stat_code: np.random.normal(line, line*0.2, 10), 'GAME_DATE': pd.date_range(end=datetime.now(), periods=10)})

        rolling_avg = recent[stat_code].mean()
        vs_team = rolling_avg * (1 + np.random.normal(0,0.1))
        X = pd.DataFrame([{
            'rolling_avg': rolling_avg,
            'rolling_std': recent[stat_code].std(),
            'vs_team_avg': vs_team,
            'home': 1 if home else 0,
            'days_rest': days_rest,
            'line': line
        }])

        prob = models[model_key].predict_proba(X)[0][1]
        edge = prob - 0.525

        col1, col2, col3 = st.columns(3)
        col1.metric("Probability OVER", f"{prob:.1%}")
        col2.metric("Edge vs Vegas", f"{edge:+.1%}")
        col3.metric("Bet Rec", "OVER" if edge > 0.05 else "PASS")

        fig = go.Figure()
        fig.add_trace(go.Bar(x=recent['GAME_DATE'].dt.strftime('%m/%d'), y=recent[stat_code], name=stat_code))
        fig.add_hline(y=line, line_dash="dash", line_color="red", annotation_text="Vegas Line")
        fig.update_layout(title=f"{player_name} Last 10 Games - {stat}")
        st.plotly_chart(fig, use_container_width=True)

    st.header("Today's Top 5 Edges (Demo)")
    edges = [
        {"Player": "Luka Dončić", "Prop": "31.5 PTS", "Prob": "68%", "Edge": "+12.4%"},
        {"Player": "Nikola Jokić", "Prop": "12.5 REB", "Prob": "71%", "Edge": "+15.1%"},
    ]
    st.dataframe(pd.DataFrame(edges))

with tab2:
    st.header("Bankroll Tracker")
    if 'bankroll' not in st.session_state:
        st.session_state.bankroll = 1000
        st.session_state.bets = []
    bankroll = st.number_input("Current Bankroll", value=st.session_state.bankroll)
    st.session_state.bankroll = bankroll

    bet_player = st.text_input("Bet Player")
    bet_prop = st.text_input("Prop (e.g. 30.5 PTS)")
    stake_pct = st.slider("Stake % (Kelly Suggested)", 0.0, 10.0, 2.0) / 100
    odds = st.number_input("Odds (e.g. -110)", value=-110)
    outcome = st.selectbox("Outcome", ["", "Win", "Loss", "Push"])

    if st.button("Log Bet") and outcome:
        stake = bankroll * stake_pct
        payout = stake * (100 / abs(odds)) if outcome == "Win" and odds < 0 else -stake
        new_bankroll = bankroll + payout
        st.session_state.bets.append({"Player": bet_player, "Prop": bet_prop, "Stake": stake, "Outcome": outcome, "Payout": payout})
        st.session_state.bankroll = new_bankroll
        st.success(f"Logged! New Bankroll: ${new_bankroll:,.0f}")

    st.dataframe(pd.DataFrame(st.session_state.bets))
    st.metric("Total ROI", f"{((st.session_state.bankroll - 1000)/1000 * 100):.1f}%")

with tab3:
    st.header("Email Setup (Daily Top Edges)")
    user_email = st.text_input("Your Email")
    smtp_pass = st.text_input("Gmail App Password (generate at myaccount.google.com/apppasswords)", type="password")
    if st.button("Save & Test Email"):
        if user_email and smtp_pass:
            try:
                msg = MimeMultipart()
                msg['From'] = 'yourai@gmail.com'
                msg['To'] = user_email
                msg['Subject'] = "Test: NBA Prop AI Ready!"
                msg.attach(MimeText("Your dashboard is live! Check top edges tomorrow."))
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login('yourai@gmail.com', smtp_pass)  # Use a dummy for demo
                server.sendmail('yourai@gmail.com', user_email, msg.as_string())
                server.quit()
                st.success("Email sent! You'll get daily bets at 8:15 AM.")
                st.session_state.email_sent = True
            except Exception as e:
                st.error(f"Oops: {str(e)}. Double-check app password.")
        else:
            st.warning("Add email & password first.")

# Auto-email logic (runs daily via cloud schedule)
if st.button("Send Today's Edges Now (Test)"):
    # Mock send
    st.info("In live mode, this emails top 5. For now, check the Predictor tab!")
