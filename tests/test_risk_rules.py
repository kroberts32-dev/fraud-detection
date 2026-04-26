import pandas as pd
import pytest

from features import build_model_frame
from risk_rules import label_risk, score_transaction

# All signals at the lowest tier — baseline score must be exactly 0.
BASE_TX = {
    "device_risk_score": 10,
    "is_international": 0,
    "amount_usd": 100,
    "velocity_24h": 1,
    "failed_logins_24h": 0,
    "prior_chargebacks": 0,
}


# ---------------------------------------------------------------------------
# label_risk — boundary conditions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score,expected", [
    (0,   "low"),
    (29,  "low"),
    (30,  "medium"),
    (59,  "medium"),
    (60,  "high"),
    (100, "high"),
])
def test_label_risk_boundaries(score, expected):
    assert label_risk(score) == expected


# ---------------------------------------------------------------------------
# score_transaction — baseline and cap
# ---------------------------------------------------------------------------

def test_baseline_scores_zero():
    assert score_transaction(BASE_TX) == 0


def test_score_capped_at_100():
    # All signals at max tier sum to 25+15+25+20+20+20 = 125; must clamp to 100.
    tx = {
        "device_risk_score": 85,
        "is_international": 1,
        "amount_usd": 1400,
        "velocity_24h": 10,
        "failed_logins_24h": 8,
        "prior_chargebacks": 3,
    }
    assert score_transaction(tx) == 100


# ---------------------------------------------------------------------------
# score_transaction — device risk (exact point values per tier)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device_score,expected_points", [
    (39, 0),   # below low tier
    (40, 10),  # low tier boundary
    (69, 10),  # top of low tier
    (70, 25),  # high tier boundary
    (85, 25),  # deep in high tier
])
def test_device_risk_tiers(device_score, expected_points):
    tx = {**BASE_TX, "device_risk_score": device_score}
    assert score_transaction(tx) == expected_points


# ---------------------------------------------------------------------------
# score_transaction — international flag
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("is_intl,expected_points", [
    (0, 0),
    (1, 15),
])
def test_international_flag(is_intl, expected_points):
    tx = {**BASE_TX, "is_international": is_intl}
    assert score_transaction(tx) == expected_points


# ---------------------------------------------------------------------------
# score_transaction — transaction amount tiers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("amount,expected_points", [
    (499,  0),
    (500,  10),
    (999,  10),
    (1000, 25),
    (2400, 25),
])
def test_amount_tiers(amount, expected_points):
    tx = {**BASE_TX, "amount_usd": amount}
    assert score_transaction(tx) == expected_points


# ---------------------------------------------------------------------------
# score_transaction — velocity tiers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("velocity,expected_points", [
    (2,  0),
    (3,  5),
    (5,  5),
    (6,  20),
    (10, 20),
])
def test_velocity_tiers(velocity, expected_points):
    tx = {**BASE_TX, "velocity_24h": velocity}
    assert score_transaction(tx) == expected_points


# ---------------------------------------------------------------------------
# score_transaction — failed logins tiers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("logins,expected_points", [
    (1, 0),
    (2, 10),
    (4, 10),
    (5, 20),
    (8, 20),
])
def test_failed_logins_tiers(logins, expected_points):
    tx = {**BASE_TX, "failed_logins_24h": logins}
    assert score_transaction(tx) == expected_points


# ---------------------------------------------------------------------------
# score_transaction — prior chargebacks tiers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("chargebacks,expected_points", [
    (0, 0),
    (1, 5),
    (2, 20),
    (3, 20),
])
def test_prior_chargebacks_tiers(chargebacks, expected_points):
    tx = {**BASE_TX, "prior_chargebacks": chargebacks}
    assert score_transaction(tx) == expected_points


# ---------------------------------------------------------------------------
# build_model_frame — feature engineering
# ---------------------------------------------------------------------------

def _make_frames(amount=100, failed_logins=0):
    transactions = pd.DataFrame([{
        "account_id": 1,
        "transaction_id": 1,
        "amount_usd": amount,
        "failed_logins_24h": failed_logins,
    }])
    accounts = pd.DataFrame([{"account_id": 1, "customer_name": "Test User"}])
    return transactions, accounts


@pytest.mark.parametrize("amount,expected_flag", [
    (999,  0),
    (1000, 1),
    (1500, 1),
])
def test_is_large_amount_flag(amount, expected_flag):
    txns, accts = _make_frames(amount=amount)
    result = build_model_frame(txns, accts)
    assert result["is_large_amount"].iloc[0] == expected_flag


@pytest.mark.parametrize("logins,expected_label", [
    (0, "none"),
    (1, "low"),
    (2, "low"),
    (3, "high"),
    (8, "high"),
])
def test_login_pressure_categories(logins, expected_label):
    txns, accts = _make_frames(failed_logins=logins)
    result = build_model_frame(txns, accts)
    assert result["login_pressure"].iloc[0] == expected_label


def test_build_model_frame_merges_account_fields():
    txns, accts = _make_frames()
    result = build_model_frame(txns, accts)
    assert "customer_name" in result.columns
    assert result["customer_name"].iloc[0] == "Test User"


def test_build_model_frame_unknown_account_produces_nan():
    transactions = pd.DataFrame([{
        "account_id": 999,
        "transaction_id": 1,
        "amount_usd": 100,
        "failed_logins_24h": 0,
    }])
    accounts = pd.DataFrame([{"account_id": 1, "customer_name": "Jane Doe"}])
    result = build_model_frame(transactions, accounts)
    assert pd.isna(result["customer_name"].iloc[0])
