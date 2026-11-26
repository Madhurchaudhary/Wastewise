"""End-to-end API tests: auth, validation, prediction persistence, history, RLS scoping, analytics, deletion."""
from __future__ import annotations

import uuid

import numpy as np
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from conftest import CATEGORIES, make_token, jpeg_bytes, png_bytes


def _upload(client, headers, data=None, ctype="image/jpeg", name="item.jpg"):
    return client.post("/api/v1/predict", headers=headers, files={"file": (name, jpeg_bytes() if data is None else data, ctype)})


# ------------------------------------------------------------------ system
def test_health_ok(app_client):
    r = app_client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] and body["supabase_configured"]
    assert body["model_version"] == "test-0.0.1-tiny"
    assert r.headers["x-content-type-options"] == "nosniff"


def test_model_info(app_client):
    r = app_client.get("/api/v1/model-info")
    assert r.status_code == 200
    assert r.json()["categories"] == CATEGORIES
    assert r.json()["confidence_threshold"] == 0.5


def test_livez_and_cors_preflight(app_client):
    assert app_client.get("/livez").json() == {"status": "alive"}
    r = app_client.options("/api/v1/predict", headers={"Origin": "http://localhost:3000",
                                                       "Access-Control-Request-Method": "POST",
                                                       "Access-Control-Request-Headers": "authorization"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"
    r = app_client.options("/api/v1/predict", headers={"Origin": "https://evil.example",
                                                       "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in r.headers


# ------------------------------------------------------------------ auth
def test_requires_token(app_client):
    r = app_client.get("/api/v1/history")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "not_authenticated"


@pytest.mark.parametrize("kind", ["expired", "wrong_aud", "wrong_iss", "wrong_key", "anon_role", "hs256", "garbage"])
def test_rejects_bad_tokens(app_client, kind):
    uid = str(uuid.uuid4())
    tok = {
        "expired": lambda: make_token(uid, exp_in=-120),
        "wrong_aud": lambda: make_token(uid, aud="anon"),
        "wrong_iss": lambda: make_token(uid, iss="https://other.supabase.co/auth/v1"),
        "wrong_key": lambda: make_token(uid, key=ec.generate_private_key(ec.SECP256R1())),
        "anon_role": lambda: make_token(uid, role="anon"),
        "hs256": lambda: make_token(uid, key="x" * 32, alg="HS256"),
        "garbage": lambda: "not.a.jwt",
    }[kind]()
    r = app_client.get("/api/v1/history", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == ("token_expired" if kind == "expired" else "invalid_token")


# ------------------------------------------------------------------ predict
def test_predict_happy_path_persists_everything(app_client, fake, user_a):
    uid, h = user_a
    r = _upload(app_client, h)
    assert r.status_code == 201, r.text
    body = r.json()
    # response matches running the same model directly
    from PIL import Image
    import io
    from app.ml import inference
    from app.ml.model_loader import registry
    direct = inference.predict(registry.model, Image.open(io.BytesIO(jpeg_bytes())).convert("RGB"))
    assert body["predicted_category"] == direct.category
    assert body["confidence"] == pytest.approx(direct.confidence, abs=1e-5)
    assert abs(sum(p["probability"] for p in body["top_predictions"]) - sum(sorted(direct.probabilities.values())[-3:])) < 1e-5
    assert body["guidance"]["disposal"]["en"] and body["guidance"]["disposal"]["hi"]
    assert body["gradcam_image"].startswith("data:image/jpeg;base64,")
    assert body["image_saved"] is True and body["image_url"].startswith("https://test-project.supabase.co/storage/v1/")
    # DB row + storage object
    rows = fake.tables["predictions"]
    assert len(rows) == 1 and rows[0]["id"] == body["id"] and rows[0]["user_id"] == uid
    path = rows[0]["image_path"]
    assert path.startswith(f"{uid}/") and path.endswith(f"{body['id']}.jpg")
    stored = fake.objects["waste-images"][path]
    assert stored[:3] == b"\xff\xd8\xff"
    assert rows[0]["model_version"] == "test-0.0.1-tiny"
    assert rows[0]["disposal_instructions"] == body["guidance"]["disposal"]


def test_predict_respects_save_images_opt_out(app_client, fake):
    uid = str(uuid.uuid4())
    fake.add_profile(uid, save_images=False)
    r = _upload(app_client, {"Authorization": f"Bearer {make_token(uid)}"})
    assert r.status_code == 201
    assert r.json()["image_saved"] is False and r.json()["image_url"] is None
    assert fake.objects["waste-images"] == {}
    assert fake.tables["predictions"][0]["image_path"] is None


def test_predict_png_with_alpha(app_client, user_a):
    r = _upload(app_client, user_a[1], png_bytes(), "image/png", "x.png")
    assert r.status_code == 201


@pytest.mark.parametrize("data,ctype,status,code", [
    (b"", "image/jpeg", 400, "empty_file"),
    (b"%PDF-1.4 not an image", "image/jpeg", 415, "unsupported_type"),
    (None, "text/plain", 415, "unsupported_type"),
    (b"\xff\xd8\xff\xe0" + b"\x00" * 200, "image/jpeg", 400, "corrupt_image"),
])
def test_predict_rejects_invalid_files(app_client, fake, user_a, data, ctype, status, code):
    r = _upload(app_client, user_a[1], data if data is not None else jpeg_bytes(), ctype)
    assert r.status_code == status
    assert r.json()["error"]["code"] == code
    assert fake.tables["predictions"] == []


def test_predict_rejects_oversize(app_client, user_a):
    big = b"\xff\xd8\xff" + b"0" * (4 * 1024 * 1024 + 10)
    r = _upload(app_client, user_a[1], big)
    assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"


def test_predict_rejects_tiny_image(app_client, user_a):
    r = _upload(app_client, user_a[1], jpeg_bytes(size=(40, 40)))
    assert r.status_code == 400 and r.json()["error"]["code"] == "image_too_small"


def test_insert_failure_cleans_up_image(app_client, fake, user_a):
    fake.fail_insert = True
    r = _upload(app_client, user_a[1])
    assert r.status_code == 502
    assert fake.objects["waste-images"] == {}          # compensating delete ran


def test_predict_rate_limit(app_client, user_a):
    from app.config import get_settings
    from app.main import app
    s = get_settings().model_copy(update={"predict_rate_per_minute": 2})
    app.dependency_overrides[get_settings] = lambda: s
    assert _upload(app_client, user_a[1]).status_code == 201
    assert _upload(app_client, user_a[1]).status_code == 201
    r = _upload(app_client, user_a[1])
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert r.json()["error"]["code"] == "rate_limited"


def test_model_unavailable_returns_503(app_client, user_a, monkeypatch):
    from app.ml.model_loader import registry
    monkeypatch.setattr(registry, "_model", None)
    r = _upload(app_client, user_a[1])
    assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"


# ------------------------------------------------------------------ history / RLS
def test_history_list_filter_paginate_and_isolation(app_client, fake, user_a, user_b):
    for c in [(200, 60, 40), (40, 200, 60), (60, 40, 200), (220, 220, 40)]:
        assert _upload(app_client, user_a[1], jpeg_bytes(color=c)).status_code == 201
    assert _upload(app_client, user_b[1]).status_code == 201

    r = app_client.get("/api/v1/history?page=1&page_size=3", headers=user_a[1])
    assert r.status_code == 200
    page = r.json()
    assert page["total"] == 4 and len(page["items"]) == 3
    assert all(i["image_url"] for i in page["items"])
    r2 = app_client.get("/api/v1/history?page=2&page_size=3", headers=user_a[1]).json()
    assert len(r2["items"]) == 1
    ids = {i["id"] for i in page["items"]} | {i["id"] for i in r2["items"]}
    assert len(ids) == 4

    cat = page["items"][0]["predicted_category"]
    filt = app_client.get(f"/api/v1/history?category={cat}", headers=user_a[1]).json()
    assert filt["total"] >= 1 and all(i["predicted_category"] == cat for i in filt["items"])

    # user B sees only their own scan and cannot read/delete A's
    assert app_client.get("/api/v1/history", headers=user_b[1]).json()["total"] == 1
    a_id = page["items"][0]["id"]
    assert app_client.get(f"/api/v1/history/{a_id}", headers=user_b[1]).status_code == 404
    assert app_client.delete(f"/api/v1/history/{a_id}", headers=user_b[1]).status_code == 404
    assert any(r["id"] == a_id for r in fake.tables["predictions"])


def test_history_detail_feedback_and_delete(app_client, fake, user_a):
    pid = _upload(app_client, user_a[1]).json()["id"]
    d = app_client.get(f"/api/v1/history/{pid}", headers=user_a[1])
    assert d.status_code == 200 and d.json()["feedback"] is None
    assert d.json()["guidance"]["bin_label"]["hi"]

    bad = app_client.put(f"/api/v1/history/{pid}/feedback", headers=user_a[1],
                         json={"user_correctness_feedback": True, "corrected_category": "glass"})
    assert bad.status_code == 400
    fb = app_client.put(f"/api/v1/history/{pid}/feedback", headers=user_a[1],
                        json={"user_correctness_feedback": False, "corrected_category": "glass", "comments": "jar"})
    assert fb.status_code == 200 and fb.json()["corrected_category"] == "glass"
    fb2 = app_client.put(f"/api/v1/history/{pid}/feedback", headers=user_a[1],
                         json={"user_correctness_feedback": True})
    assert fb2.status_code == 200 and len(fake.tables["feedback"]) == 1      # upsert, not duplicate
    assert "verified_label" not in fake.tables["feedback"][0] or fake.tables["feedback"][0]["verified_label"] is False

    path = fake.tables["predictions"][0]["image_path"]
    assert app_client.delete(f"/api/v1/history/{pid}", headers=user_a[1]).status_code == 204
    assert fake.tables["predictions"] == [] and path not in fake.objects["waste-images"]
    assert fake.tables["feedback"] == []
    assert app_client.get(f"/api/v1/history/{pid}", headers=user_a[1]).status_code == 404


def test_history_validation_errors(app_client, user_a):
    r = app_client.get("/api/v1/history?page=0", headers=user_a[1])
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    assert app_client.get("/api/v1/history/not-a-uuid", headers=user_a[1]).status_code == 422
    r = app_client.get("/api/v1/history?date_from=2026-05-02&date_to=2026-05-01", headers=user_a[1])
    assert r.status_code == 400


# ------------------------------------------------------------------ analytics
def test_analytics_endpoint(app_client, user_a):
    for _ in range(3):
        _upload(app_client, user_a[1])
    a = app_client.get("/api/v1/analytics", headers=user_a[1]).json()
    assert a["total_scans"] == 3
    assert sum(c["count"] for c in a["category_distribution"]) == 3
    assert len(a["daily_counts"]) == 30 and a["daily_counts"][-1]["count"] == 3
    assert a["confident_scans"] + a["uncertain_scans"] == 3
    dist = app_client.get("/api/v1/analytics/category-distribution", headers=user_a[1]).json()
    assert len(dist) == 7


# ------------------------------------------------------------------ account deletion
def test_delete_account_removes_images_rows_and_user(app_client, fake, user_a, user_b):
    for _ in range(2):
        _upload(app_client, user_a[1])
    _upload(app_client, user_b[1])
    r = app_client.delete("/api/v1/account", headers=user_a[1])
    assert r.status_code == 204
    assert fake.deleted_users == [user_a[0]]
    assert all(not p.startswith(user_a[0]) for p in fake.objects["waste-images"])
    assert len(fake.objects["waste-images"]) == 1                       # B untouched
    assert all(r["user_id"] != user_a[0] for r in fake.tables["predictions"])


def test_error_envelope_never_leaks_internals(app_client, user_a, monkeypatch):
    from app.services import analytics_service

    async def boom(*a, **k):
        raise RuntimeError("secret internal detail")
    monkeypatch.setattr(analytics_service, "user_analytics", boom)
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/api/v1/analytics", headers=user_a[1])
    assert r.status_code == 500
    assert "secret" not in r.text and r.json()["error"]["code"] == "internal_error"


def test_storage_upload_is_never_shared_cached():
    """Private photos must not be kept by the CDN, so a deleted image stops being served at once."""
    import asyncio

    import httpx

    from app.database.supabase_client import PRIVATE_IMAGE_CACHE_CONTROL, SupabaseGateway

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, json={"Key": "uploads/x.jpg"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            gw = SupabaseGateway("https://sb.example", "anon", "service", http)
            await gw.upload("tok", "waste-images", "u/x.jpg", b"jpeg", "image/jpeg")

    asyncio.run(run())
    cc = seen["cache-control"]
    assert cc == PRIVATE_IMAGE_CACHE_CONTROL and "no-store" in cc and "private" in cc
