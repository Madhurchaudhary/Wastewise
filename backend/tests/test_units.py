"""Unit tests: preprocessing parity, Grad-CAM exactness, knowledge base, analytics, quality, validators, limiter."""
from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image

from conftest import CATEGORIES, MODEL_DIR, jpeg_bytes


def _img(seed=0, size=(333, 251)):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


# ------------------------------------------------------------------ preprocessing == torchvision eval transform
@pytest.mark.parametrize("size", [(333, 251), (251, 333), (256, 256), (1024, 300), (225, 900)])
def test_preprocess_matches_torchvision(size):
    tv = pytest.importorskip("torchvision.transforms")
    from app.ml.preprocessing import preprocess
    img = _img(1, size)
    ref = tv.Compose([tv.Resize(256, interpolation=tv.InterpolationMode.BILINEAR), tv.CenterCrop(224), tv.ToTensor(),
                      tv.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(img).numpy()[None]
    ours = preprocess(img)
    assert ours.shape == (1, 3, 224, 224) and ours.dtype == np.float32
    assert np.abs(ours - ref).max() < 1e-5


# ------------------------------------------------------------------ Grad-CAM weights == true gradients
def _head_forward(head, feats):
    pooled = feats.reshape(feats.shape[0], -1).mean(axis=1)
    z = head["w1"] @ pooled + head["b1"]
    h = z * np.clip(z + 3, 0, 6) / 6
    return head["w2"] @ h + head["b2"]


def test_gradcam_alpha_equals_numeric_gradient():
    from app.ml import inference
    from app.ml.model_loader import ModelRegistry
    m = ModelRegistry().load(str(MODEL_DIR))
    res = inference.predict(m, _img(3))
    feats = res.features.astype(np.float64)
    head = {k: v.astype(np.float64) for k, v in m.head.items()}
    # the numpy head reproduces the ONNX logits
    logits = _head_forward(head, feats)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    np.testing.assert_allclose(p, [res.probabilities[c] for c in CATEGORIES], atol=1e-4)
    c = CATEGORIES.index(res.category)
    eps = 1e-4
    C, H, W = feats.shape
    num = np.zeros(C)
    for k in range(C):   # d logit_c / d A_k(0,0)
        f1, f2 = feats.copy(), feats.copy()
        f1[k, 0, 0] += eps
        f2[k, 0, 0] -= eps
        num[k] = (_head_forward(head, f1)[c] - _head_forward(head, f2)[c]) / (2 * eps)
    # analytic alpha from gradcam(): recompute it the same way
    pooled = feats.reshape(C, -1).mean(axis=1)
    z = head["w1"] @ pooled + head["b1"]
    alpha = (head["w1"].T @ (head["w2"][c] * inference._hardswish_grad(z))) / (H * W)
    np.testing.assert_allclose(alpha, num, rtol=1e-4, atol=1e-7)
    cam = inference.gradcam(m, res.features, res.category)
    assert cam.shape == (H, W) and cam.min() >= 0 and cam.max() <= 1 + 1e-6


def test_low_confidence_marks_uncertain():
    from app.ml import inference
    from app.ml.model_loader import ModelRegistry
    m = ModelRegistry().load(str(MODEL_DIR))
    r = inference.predict(m, _img(4), threshold_override=0.999)
    assert r.is_confident is False and "low_confidence" in r.reasons and r.item is None
    r2 = inference.predict(m, _img(4), threshold_override=0.0)
    assert ("ambiguous_top2" in r2.reasons) == (not r2.is_confident)


# ------------------------------------------------------------------ knowledge base
def test_knowledge_base_complete_for_every_category():
    from app.services import knowledge
    for c in CATEGORIES:
        g = knowledge.guidance_for(c)
        for f in ("segregation", "disposal", "recycling", "safety", "common_mistakes", "environment"):
            assert g[f]["en"] and len(g[f]["en"]) == len(g[f]["hi"]), (c, f)
            for lang in ("en", "hi"):
                # Whole sentences, never single characters (regression: "Why it matters" rendered
                # one letter per bullet because a plain string went through list()).
                assert all(isinstance(x, str) and len(x) > 15 for x in g[f][lang]), (c, f, lang)
        assert g["name"]["hi"] and g["bin_label"]["en"]


def test_guidance_list_normalises_plain_strings():
    from app.services.knowledge import _bl
    assert _bl({"en": "One sentence.", "hi": "एक वाक्य।"}) == {"en": ["One sentence."], "hi": ["एक वाक्य।"]}
    assert _bl({"en": ["a b", ""], "hi": None}) == {"en": ["a b"], "hi": []}
    assert _bl(None) == {"en": [], "hi": []}


def test_hazardous_notice_rules():
    from app.services.knowledge import notices_for
    codes = lambda *a: [n["code"] for n in notices_for(*a)]  # noqa: E731
    assert "hazardous_scope" in codes("hazardous", True, ["hazardous", "metal"], "aerosol_can")
    assert "hazardous_scope" in codes("metal", True, ["metal", "hazardous"], None)       # close alternative
    assert "hazardous_scope" not in codes("plastic", True, ["plastic", "glass", "hazardous"], None)
    assert codes("e_waste", True, ["e_waste", "metal"], "battery") == ["ewaste_scope", "battery_safety"]
    assert codes("glass", False, ["glass", "plastic"], None) == ["uncertain"]


def test_recyclable_flag():
    from app.services.knowledge import is_recyclable
    assert is_recyclable("plastic") and is_recyclable("e_waste")
    assert not is_recyclable("organic") and not is_recyclable("hazardous")


# ------------------------------------------------------------------ analytics
def test_summarise():
    from app.services.analytics_service import summarise
    now = datetime(2026, 11, 25, 12, tzinfo=timezone.utc)
    rows = [
        {"predicted_category": "plastic", "confidence_score": 0.9, "is_confident": True, "recyclable": True,
         "created_at": (now - timedelta(hours=1)).isoformat()},
        {"predicted_category": "plastic", "confidence_score": 0.7, "is_confident": True, "recyclable": True,
         "created_at": (now - timedelta(days=3)).isoformat()},
        {"predicted_category": "hazardous", "confidence_score": 0.4, "is_confident": False, "recyclable": False,
         "created_at": (now - timedelta(days=20)).isoformat()},
        {"predicted_category": "organic", "confidence_score": 0.8, "is_confident": True, "recyclable": False,
         "created_at": (now - timedelta(days=45)).isoformat().replace("+00:00", "Z")},
    ]
    s = summarise(rows, now=now)
    assert s["total_scans"] == 4 and s["confident_scans"] == 3 and s["uncertain_scans"] == 1
    assert s["recyclable_scans"] == 2 and s["hazardous_or_ewaste_scans"] == 1
    assert s["average_confidence"] == pytest.approx(0.7)
    assert s["most_common_category"] == "plastic"
    assert s["scans_last_7_days"] == 2 and s["scans_last_30_days"] == 3
    assert {d["category"]: d["percentage"] for d in s["category_distribution"]}["plastic"] == 50.0
    assert len(s["daily_counts"]) == 30 and s["daily_counts"][-1] == {"date": "2026-11-25", "count": 1}
    empty = summarise([], now=now)
    assert empty["average_confidence"] is None and empty["most_common_category"] is None


# ------------------------------------------------------------------ image quality + storage
def test_quality_warnings():
    from app.services.image_service import quality_warnings
    dark = Image.open(io.BytesIO(jpeg_bytes(color=(5, 5, 5), noise=False)))
    codes = [w["code"] for w in quality_warnings(dark)]
    assert "too_dark" in codes and "blurry" in codes
    sharp = Image.open(io.BytesIO(jpeg_bytes()))
    assert quality_warnings(sharp) == []
    small = Image.open(io.BytesIO(jpeg_bytes(size=(120, 100))))
    assert "low_resolution" in [w["code"] for w in quality_warnings(small)]


def test_encode_for_storage_downscales_and_strips_exif():
    from app.services.image_service import encode_for_storage, storage_path
    img = _img(5, (3000, 2000))
    exif = Image.Exif()
    exif[0x010F] = "SecretCam"
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    src = Image.open(io.BytesIO(buf.getvalue()))
    out = Image.open(io.BytesIO(encode_for_storage(src, 1280)))
    assert max(out.size) == 1280 and out.format == "JPEG"
    assert 0x010F not in out.getexif()
    p = storage_path("11111111-2222-3333-4444-555555555555", "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                     datetime(2026, 3, 9, tzinfo=timezone.utc))
    assert p == "11111111-2222-3333-4444-555555555555/2026/03/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jpg"
    with pytest.raises(ValueError):
        storage_path("u", "../../etc/passwd")


# ------------------------------------------------------------------ validators
def test_owned_storage_path():
    from app.utils.validators import is_owned_storage_path
    u = "11111111-2222-3333-4444-555555555555"
    good = f"{u}/2026/11/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jpg"
    assert is_owned_storage_path(good, u)
    assert not is_owned_storage_path(good, "99999999-2222-3333-4444-555555555555")
    assert not is_owned_storage_path(f"{u}/../other/x.jpg", u)
    assert not is_owned_storage_path(None, u)


def test_exif_rotation_applied():
    from app.utils.validators import validate_image_bytes
    img = _img(6, (300, 200))
    exif = Image.Exif()
    exif[0x0112] = 6   # rotate 90 CW on display
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif)
    out = validate_image_bytes(buf.getvalue(), "image/jpeg", 4 << 20, 40_000_000)
    assert out.size == (200, 300) and out.mode == "RGB"


# ------------------------------------------------------------------ limiter
def test_sliding_window_limiter():
    from app.utils.rate_limit import SlidingWindowLimiter
    lim = SlidingWindowLimiter()
    lim.check("k", [(2, 60)], now=0)
    lim.check("k", [(2, 60)], now=1)
    with pytest.raises(HTTPException) as e:
        lim.check("k", [(2, 60)], now=2)
    assert e.value.status_code == 429 and int(e.value.headers["Retry-After"]) == 59
    lim.check("k", [(2, 60)], now=61)          # window slid
    lim.check("other", [(2, 60)], now=2)       # keys are independent


# --------------------------------------------------------------------------- remote model artifacts
def _patch_httpx(monkeypatch, payloads: dict[str, bytes]):
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, content=payloads[name]) if name in payloads else httpx.Response(404)

    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: real(transport=httpx.MockTransport(handler), **{x: y for x, y in k.items() if x != "transport"}))


def test_materialise_downloads_and_verifies(tmp_path, monkeypatch):
    import hashlib
    from app.ml.model_loader import materialise_artifacts

    onnx, npz = b"onnx-bytes" * 100, b"npz-bytes" * 50
    card = {"onnx_sha256": hashlib.sha256(onnx).hexdigest(), "gradcam_head_sha256": hashlib.sha256(npz).hexdigest()}
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    _patch_httpx(monkeypatch, {"wastewise.onnx": onnx, "gradcam_head.npz": npz})
    d = materialise_artifacts(tmp_path / "card", "https://example.test/artifacts/", card)
    assert (d / "wastewise.onnx").read_bytes() == onnx
    assert (d / "gradcam_head.npz").read_bytes() == npz


def test_materialise_rejects_tampered_download(tmp_path, monkeypatch):
    import hashlib
    from app.ml.model_loader import materialise_artifacts

    card = {"onnx_sha256": hashlib.sha256(b"genuine").hexdigest(), "gradcam_head_sha256": hashlib.sha256(b"x").hexdigest()}
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    _patch_httpx(monkeypatch, {"wastewise.onnx": b"tampered", "gradcam_head.npz": b"x"})
    with pytest.raises(RuntimeError, match="checksum"):
        materialise_artifacts(tmp_path / "card", "https://example.test/a", card)
    assert not list((tmp_path / "tmp").rglob("wastewise.onnx"))


def test_materialise_requires_checksums_and_url(tmp_path):
    from app.ml.model_loader import materialise_artifacts

    with pytest.raises(FileNotFoundError):
        materialise_artifacts(tmp_path, "", {})
    with pytest.raises(RuntimeError, match="checksums"):
        materialise_artifacts(tmp_path, "https://example.test", {"onnx_sha256": "abc"})
