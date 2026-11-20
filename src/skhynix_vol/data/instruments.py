"""Exchange instrument facts used to validate derivative quote files."""
from __future__ import annotations

KRX_SINGLE_STOCK_OPTIONS = {
    "000660": {
        "underlying": "SK hynix",
        "underlying_ticker": "000660.KS",
        "exchange": "KRX",
        "product": "Single Stock Option",
        "contract_multiplier": 10,
        "exercise_style": "european",
        "settlement_type": "cash",
        "listing_cycle": "monthly; two non-quarterly and four quarterly months",
        "last_trading_day": "second Thursday of the contract month",
        "source_url": "https://global.krx.co.kr/contents/GLB/02/0201/0201040402/GLB0201040402.jsp",
    }
}


def instrument_spec(underlying: str = "000660") -> dict[str, object]:
    """Return verified KRX facts for SK hynix single-stock options."""
    key = str(underlying).upper().replace(".KS", "")
    if key not in KRX_SINGLE_STOCK_OPTIONS:
        raise ValueError(f"No verified KRX single-stock option specification for {underlying!r}")
    return dict(KRX_SINGLE_STOCK_OPTIONS[key])
