from risk_rules import label_risk, score_transaction

BASE_TX = {
    "device_risk_score": 10,
    "is_international": 0,
    "amount_usd": 100,
    "velocity_24h": 1,
    "failed_logins_24h": 0,
    "prior_chargebacks": 0,
}


def test_label_risk_thresholds():
    assert label_risk(10) == "low"
    assert label_risk(35) == "medium"
    assert label_risk(75) == "high"


def test_large_amount_adds_risk():
    tx = {**BASE_TX, "amount_usd": 1200}
    assert score_transaction(tx) >= 25


def test_high_device_risk_increases_score():
    low_device = {**BASE_TX, "device_risk_score": 10}
    high_device = {**BASE_TX, "device_risk_score": 75}
    assert score_transaction(high_device) > score_transaction(low_device)


def test_international_increases_score():
    domestic = {**BASE_TX, "is_international": 0}
    international = {**BASE_TX, "is_international": 1}
    assert score_transaction(international) > score_transaction(domestic)


def test_high_velocity_increases_score():
    low_vel = {**BASE_TX, "velocity_24h": 1}
    high_vel = {**BASE_TX, "velocity_24h": 8}
    assert score_transaction(high_vel) > score_transaction(low_vel)


def test_prior_chargebacks_increase_score():
    clean = {**BASE_TX, "prior_chargebacks": 0}
    one_cb = {**BASE_TX, "prior_chargebacks": 1}
    two_cb = {**BASE_TX, "prior_chargebacks": 2}
    assert score_transaction(one_cb) > score_transaction(clean)
    assert score_transaction(two_cb) > score_transaction(one_cb)
