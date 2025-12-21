# strategy_logic.py

def get_strategy_text(prev_signal, curr_signal):
    # 1. No previous record
    if not prev_signal:
        return f"Initialize position based on current {curr_signal} signal."

    # 2. [BUY -> SELL] : downward reversal
    if prev_signal == 'BUY' and curr_signal == 'SELL':
        return "Trend Reversal (BUY → SELL): Close all Long positions (TQQQ/QLD). Open Short position (SQQQ) with 10% of total budget."

    # 3. [BUY -> BUY] : rise continues
    if prev_signal == 'BUY' and curr_signal == 'BUY':
        return "Trend Continuation (BUY): Add 10% to Long position (TQQQ). If cash is insufficient or fully invested, maintain current position."

    # 4. [BUY -> HOLD] : weakening upward trend
    if prev_signal == 'BUY' and curr_signal == 'HOLD':
        return "Trend Weakening (BUY → HOLD): Hold current Long positions. Do not open new positions. Wait for clearer signals."

    # 5. [SELL -> BUY] : upward reversal
    if prev_signal == 'SELL' and curr_signal == 'BUY':
        return "Trend Reversal (SELL → BUY): Close all Short positions (SQQQ). Open Long position (TQQQ) with 10% of total budget."

    # 6. [SELL -> SELL] : decline continues
    if prev_signal == 'SELL' and curr_signal == 'SELL':
        return "Trend Continuation (SELL): Add 10% to Short position (SQQQ). If cash is insufficient, maintain current position."

    # 7. [HOLD -> BUY]
    if prev_signal == 'HOLD' and curr_signal == 'BUY':
        return "New Entry Opportunity (HOLD → BUY): Enter Long position (TQQQ) with 10% of total budget."
    if prev_signal == 'HOLD' and curr_signal == 'SELL':
        return "New Entry Opportunity (HOLD → SELL): Enter Short position (SQQQ) with 10% of total budget."

    # ETC
    return "Market Neutral (HOLD): No Action. Stay in cash or hold existing positions."
