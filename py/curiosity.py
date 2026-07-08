import yfinance as yf
df = yf.Ticker("TTS.V").history(start="2026-07-03", end="2026-07-04", interval="5m")
print(df[["Open", "High", "Low", "Close", "Volume"]])