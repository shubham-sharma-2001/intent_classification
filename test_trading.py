"""Full test of trading classifier including the originally failing edge cases."""
from predict import IntentPredictor

p = IntentPredictor()

tests = [
    # ── Originally FAILING cases (user-reported) ─────────────────────────
    ("sell", "get rid of sharma stock"),          # unknown stock name
    ("sell", "get rid of all stockes"),           # typo + no stock name
    # ── SELL paraphrases ──────────────────────────────────────────────────
    ("sell", "dump all my stocks right now"),
    ("sell", "unload all my shares"),
    ("sell", "liquidate my entire portfolio today"),
    ("sell", "clear out all my stocks"),
    ("sell", "wipe out my holdings completely"),
    ("sell", "exit GOOG position 30 units"),
    ("sell", "offload all my HDFC bank shares"),
    ("sell", "i want to sell this stock right away"),
    # ── BUY ───────────────────────────────────────────────────────────────
    ("buy", "I want to grab 10 units of XYZ stock"),
    ("buy", "purchase 50 shares of AAPL now"),
    ("buy", "go long on TSLA 200 units"),
    ("buy", "get me into this stock with 200 units"),
    # ── HOLD ──────────────────────────────────────────────────────────────
    ("hold", "keep my RELIANCE position for now"),
    ("hold", "do not sell AAPL just hold"),
    ("hold", "just hold and do nothing with my stock"),
    # ── CANCEL ORDER ──────────────────────────────────────────────────────
    ("cancel_order", "cancel my pending buy order for TSLA"),
    ("cancel_order", "i changed my mind please cancel the order"),
    ("cancel_order", "abort the trade request"),
    # ── CHECK PRICE ───────────────────────────────────────────────────────
    ("check_price", "what is the current price of TSLA"),
    ("check_price", "give me the latest share price"),
    ("check_price", "tell me the live price of the stock"),
    # ── PORTFOLIO STATUS ──────────────────────────────────────────────────
    ("portfolio_status", "show me all my open positions"),
    ("portfolio_status", "what is my total portfolio value today"),
    ("portfolio_status", "am I in profit or loss overall"),
    # ── SET ALERT ─────────────────────────────────────────────────────────
    ("set_alert", "alert me when TSLA reaches 350"),
    ("set_alert", "set a stop loss notification for my position"),
    ("set_alert", "notify me when price reaches my level"),
]

print(f"\n{'Input Text':<50} {'Expected':<18} {'Predicted':<18} {'Conf':>5}  Status")
print("─" * 105)

correct = 0
for expected, text in tests:
    r = p.predict(text)
    ok = "✅" if r["intent"] == expected else "❌"
    if r["intent"] == expected:
        correct += 1
    print(f"{text:<50} {expected:<18} {r['intent']:<18} {r['confidence']:>4.3f}  {ok}")

print(f"\n{'─'*105}")
print(f"Result: {correct}/{len(tests)} correct ({100*correct/len(tests):.0f}%)")
