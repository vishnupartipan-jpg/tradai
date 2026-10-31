import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "paper_trading.db"
STARTING_CASH = 100_000.0

st.set_page_config(page_title="NSE AI Paper Trader", page_icon="📈", layout="wide")

def connect():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with connect() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL CHECK(side IN ('BUY','SELL')),
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                value REAL NOT NULL
            )
        """)
init_db()

def normalize_symbol(raw):
    symbol = raw.strip().upper().replace(" ", "")
    if not symbol:
        return ""
    if not symbol.endswith(".NS"):
        symbol += ".NS"
    return symbol

@st.cache_data(ttl=300, show_spinner=False)
def load_prices(symbol, period):
    data = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False)
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.rename(columns=str.title)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in data.columns for c in required):
        return pd.DataFrame()
    data = data[required].dropna().copy()
    close = data["Close"]
    data["EMA20"] = close.ewm(span=20, adjust=False).mean()
    data["EMA50"] = close.ewm(span=50, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    data["RSI14"] = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["MACDSignal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["VolumeAvg20"] = data["Volume"].rolling(20).mean()
    return data

def signal_for(data):
    row = data.iloc[-1]
    score = 0
    reasons = []
    if row["EMA20"] > row["EMA50"]:
        score += 1
        reasons.append("EMA 20 is above EMA 50 (positive trend filter)")
    else:
        score -= 1
        reasons.append("EMA 20 is below EMA 50 (negative trend filter)")
    if pd.notna(row["RSI14"]):
        if row["RSI14"] >= 55:
            score += 1
            reasons.append("RSI is above 55 (positive momentum)")
        elif row["RSI14"] <= 45:
            score -= 1
            reasons.append("RSI is below 45 (weak momentum)")
        else:
            reasons.append("RSI is in the middle zone")
    if row["MACD"] > row["MACDSignal"]:
        score += 1
        reasons.append("MACD is above its signal line")
    else:
        score -= 1
        reasons.append("MACD is below its signal line")
    if pd.notna(row["VolumeAvg20"]) and row["VolumeAvg20"] > 0:
        if row["Volume"] > row["VolumeAvg20"]:
            reasons.append("Today's volume is above its 20-day average")
        else:
            reasons.append("Today's volume is not above its 20-day average")
    label = "BUY" if score >= 2 else ("SELL" if score <= -2 else "HOLD")
    return label, score, reasons

def ledger():
    with connect() as con:
        return pd.read_sql_query("SELECT * FROM trades ORDER BY id DESC", con)

def portfolio_state(trades):
    cash = STARTING_CASH
    positions = {}
    realized = 0.0
    for _, t in trades.sort_values("id").iterrows():
        sym, side = t["symbol"], t["side"]
        qty, price = int(t["quantity"]), float(t["price"])
        if sym not in positions:
            positions[sym] = {"qty": 0, "cost": 0.0}
        p = positions[sym]
        if side == "BUY":
            cash -= qty * price
            p["cost"] += qty * price
            p["qty"] += qty
        else:
            avg = p["cost"] / p["qty"] if p["qty"] else 0
            realized += (price - avg) * qty
            p["cost"] -= avg * qty
            p["qty"] -= qty
            cash += qty * price
    positions = {s: p for s, p in positions.items() if p["qty"] > 0}
    return cash, positions, realized

def record_trade(symbol, side, qty, price):
    with connect() as con:
        con.execute(
            "INSERT INTO trades(timestamp,symbol,side,quantity,price,value) VALUES(?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), symbol, side, qty, price, qty * price)
        )

st.title("📈 NSE AI Paper Trader")
st.caption("A learning project: technical indicators + simulated trades. No real orders are placed.")

with st.sidebar:
    st.header("Market data")
    raw_symbol = st.text_input("NSE symbol", value="RELIANCE", help="Examples: RELIANCE, TCS, INFY, HDFCBANK")
    symbol = normalize_symbol(raw_symbol)
    period = st.selectbox("History", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    st.caption("Yahoo Finance data can be delayed, incomplete or unavailable. It is not an exchange feed.")
    if st.button("Refresh market data"):
        load_prices.clear()

if not symbol:
    st.warning("Enter an NSE ticker symbol in the sidebar.")
    st.stop()

with st.spinner(f"Loading {symbol} market data..."):
    prices = load_prices(symbol, period)

if prices.empty or len(prices) < 55:
    st.error("Could not load enough price history for this symbol. Check the ticker and try again later.")
    st.info("Use the NSE ticker without the suffix, for example RELIANCE or TCS. The app adds .NS automatically.")
    st.stop()

latest = prices.iloc[-1]
last_price = float(latest["Close"])
signal, score, reasons = signal_for(prices)
last_date = prices.index[-1].strftime("%d %b %Y")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest close", f"₹{last_price:,.2f}", help=f"Last available daily bar: {last_date}")
m2.metric("RSI (14)", f"{latest['RSI14']:.1f}" if pd.notna(latest["RSI14"]) else "—")
m3.metric("EMA 20", f"₹{latest['EMA20']:,.2f}")
m4.metric("Rule-based signal", signal, f"Score {score:+d}/3")

st.subheader(f"{symbol} · Daily price chart")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=prices.index, open=prices["Open"], high=prices["High"],
    low=prices["Low"], close=prices["Close"], name="Price"
))
fig.add_trace(go.Scatter(x=prices.index, y=prices["EMA20"], name="EMA 20", line=dict(width=1.5)))
fig.add_trace(go.Scatter(x=prices.index, y=prices["EMA50"], name="EMA 50", line=dict(width=1.5)))
fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=25,b=10))
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns([1, 1])
with left:
    st.subheader("Signal explanation")
    st.write(f"**Current educational signal: {signal}**")
    for reason in reasons:
        st.write(f"- {reason}")
    st.caption("This is a simple indicator score, not a trained AI model and not a prediction of future returns.")
with right:
    st.subheader("Momentum indicators")
    ind = go.Figure()
    ind.add_trace(go.Scatter(x=prices.index, y=prices["RSI14"], name="RSI 14"))
    ind.add_hline(y=70, line_dash="dash")
    ind.add_hline(y=30, line_dash="dash")
    ind.update_layout(height=260, yaxis_range=[0,100], margin=dict(l=10,r=10,t=15,b=10))
    st.plotly_chart(ind, use_container_width=True)

st.divider()
st.header("💼 Paper trading")
st.caption(f"Starting virtual cash: ₹{STARTING_CASH:,.2f}. Trades are saved locally in paper_trading.db.")

trades = ledger()
cash, positions, realized = portfolio_state(trades)
market_value = 0.0
unrealized = 0.0
position_rows = []
for sym, p in positions.items():
    hist = load_prices(sym, "5d")
    mark = float(hist.iloc[-1]["Close"]) if not hist.empty else p["cost"] / p["qty"]
    value = p["qty"] * mark
    avg = p["cost"] / p["qty"]
    market_value += value
    unrealized += value - p["cost"]
    position_rows.append({"Symbol": sym, "Qty": p["qty"], "Avg cost": avg, "Latest price": mark, "Market value": value, "Unrealized P&L": value-p["cost"]})

c1,c2,c3,c4 = st.columns(4)
c1.metric("Available virtual cash", f"₹{cash:,.2f}")
c2.metric("Holdings value", f"₹{market_value:,.2f}")
c3.metric("Unrealized P&L", f"₹{unrealized:,.2f}")
c4.metric("Realized P&L", f"₹{realized:,.2f}")
st.metric("Total paper portfolio value", f"₹{cash+market_value:,.2f}", f"{cash+market_value-STARTING_CASH:,.2f} vs starting cash")

st.subheader(f"Place a simulated trade · {symbol}")
buy_tab, sell_tab = st.tabs(["Buy", "Sell"])
with buy_tab:
    with st.form("buy_form"):
        buy_qty = st.number_input("Quantity to buy", min_value=1, max_value=100000, value=1, step=1, key="buy_qty")
        st.write(f"Estimated simulated cost: **₹{buy_qty*last_price:,.2f}**")
        buy_ok = st.form_submit_button("Simulate BUY")
    if buy_ok:
        cost = int(buy_qty) * last_price
        if cost > cash:
            st.error("Not enough virtual cash for this simulated purchase.")
        else:
            record_trade(symbol, "BUY", int(buy_qty), last_price)
            st.success(f"Simulated BUY recorded: {buy_qty} × {symbol} at ₹{last_price:,.2f}.")
            st.rerun()
with sell_tab:
    held = positions.get(symbol, {"qty": 0})["qty"]
    st.write(f"Available to sell in paper portfolio: **{held} shares**")
    with st.form("sell_form"):
        sell_qty = st.number_input("Quantity to sell", min_value=1, max_value=max(1, held), value=1, step=1, key="sell_qty", disabled=(held == 0))
        st.write(f"Estimated simulated proceeds: **₹{sell_qty*last_price:,.2f}**")
        sell_ok = st.form_submit_button("Simulate SELL", disabled=(held == 0))
    if sell_ok:
        if int(sell_qty) > held:
            st.error("You do not hold that many shares in this paper portfolio.")
        else:
            record_trade(symbol, "SELL", int(sell_qty), last_price)
            st.success(f"Simulated SELL recorded: {sell_qty} × {symbol} at ₹{last_price:,.2f}.")
            st.rerun()

if position_rows:
    st.subheader("Open paper positions")
    st.dataframe(pd.DataFrame(position_rows), use_container_width=True, hide_index=True)
else:
    st.info("No open paper positions yet. Use the Buy tab to record a simulated trade.")

st.subheader("Trade history")
if trades.empty:
    st.write("No simulated trades recorded yet.")
else:
    display = trades.copy()
    display["timestamp"] = pd.to_datetime(display["timestamp"]).dt.strftime("%d %b %Y %H:%M:%S")
    display = display.rename(columns={"timestamp":"Time","symbol":"Symbol","side":"Side","quantity":"Qty","price":"Price","value":"Value"})
    st.dataframe(display[["Time","Symbol","Side","Qty","Price","Value"]], use_container_width=True, hide_index=True)

with st.expander("Reset paper account"):
    st.warning("This permanently deletes the local simulated trade history and restores the virtual starting balance.")
    confirm = st.checkbox("I understand: delete all paper trades")
    if st.button("Reset paper account", disabled=not confirm):
        with connect() as con:
            con.execute("DELETE FROM trades")
        st.success("Paper account reset.")
        st.rerun()

st.divider()
st.caption("Educational software only. Not investment advice. No broker connection, live execution, stop-loss orders, taxes, fees, slippage or circuit limits are modeled. Historical data may be delayed or inaccurate. Do not use this app to make real-money decisions.")
