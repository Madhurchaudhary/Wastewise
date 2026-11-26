# WasteWise API reference (v1)

**Base URL:** `<API_ORIGIN>/api/v1`

**Interactive docs:** `/docs`, available when `ENVIRONMENT` is not `production`.

**Authentication:** every endpoint except `/health` and `/model-info` needs `Authorization: Bearer <Supabase access token>`.
- Tokens are verified against the project's JWKS: ES256, audience `authenticated`, issuer `<SUPABASE_URL>/auth/v1`.
- The API forwards the same token to Supabase, so Row Level Security limits every query to the caller's own rows and storage objects.

## Error format

Every error uses the same envelope:

```json
{ "error": { "code": "file_too_large", "message": "Image exceeds the 4 MB limit.", "request_id": "3f9c..." } }
```

Validation errors (422) also include `fields: [{field, message}]`. Each response carries an `X-Request-ID` header.

| Status | Codes |
|---|---|
| 400 | `empty_file`, `corrupt_image`, `image_too_small`, `invalid_date_range`, `invalid_feedback` |
| 401 | `not_authenticated`, `invalid_token`, `token_expired` |
| 403 | `forbidden` |
| 404 | `not_found` |
| 413 | `file_too_large`, `image_too_large` |
| 415 | `unsupported_type` |
| 422 | `validation_error` |
| 429 | `rate_limited` (sets the `Retry-After` header) |
| 500 | `internal_error`; internal details are never exposed |
| 502 | `upstream_error` (Supabase failed) |
| 503 | `model_unavailable`, `database_unavailable` |

## Endpoints

### `GET /health`
Readiness check. Returns 200 when the model is loaded and Supabase is configured, otherwise 503 with `status: "degraded"`.

The body reports `model_loaded`, `model_error`, `model_version` and `supabase_configured`. `GET /livez` (no prefix) is a pure liveness probe.

### `GET /model-info`
Returns the public part of `model_card.json`:
- version and architecture
- categories and items
- calibrated `temperature` and `confidence_threshold`
- held-out test metrics and latency
- training datasets with their licences
- known limitations

### `POST /predict`
Accepts `multipart/form-data` with a `file` field: JPG, PNG or WEBP, at most 4 MB and at least 64×64 px.

Rate limit: 10 requests per minute and 300 per day per user. Both are configurable.

Processing steps:
1. **Validate.** Checks the declared MIME type, the magic bytes and a full decode, and enforces the pixel limit. EXIF rotation is applied and transparency is flattened onto white.
2. **Infer.** ONNX Runtime runs the exact training eval transform (resize 256 → centre-crop 224 → ImageNet normalisation). Probabilities are temperature-scaled.
3. **Reliability check.** `is_confident` is true only if the top probability is at least `confidence_threshold` *and* the gap between the top two is at least 0.15. Otherwise `uncertainty_reasons` lists `low_confidence` and/or `ambiguous_top2`, and the UI shows the result as uncertain.
4. **Fine-grained item.** `predicted_item` is set only when the result is confident and the item probability within the predicted category is at least 0.6.
5. **Grad-CAM.** Computed exactly, using the analytic gradient of the classifier head. The overlay comes back as a `data:` URL and is never stored.
6. **Image quality hints.** Flags `blurry`, `too_dark`, `too_bright` and `low_resolution`. These are advisory only and do not change the prediction.
7. **Storage.** Skipped if the user turned off `save_images`. The image is re-encoded as JPEG (max side 1280 px, which removes EXIF/GPS) and uploaded to the private bucket `waste-images/<user_id>/<yyyy>/<mm>/<prediction_id>.jpg`.
8. **Persist.** The row is inserted into `predictions`. If the insert fails, the uploaded image is deleted.

Response `201`:
```json
{
  "id": "uuid", "created_at": "2026-11-25T17:40:00Z",
  "predicted_category": "plastic", "confidence": 0.93, "is_confident": true, "uncertainty_reasons": [],
  "top_predictions": [{"category": "plastic", "probability": 0.93}, {"category": "glass", "probability": 0.04}, {"category": "metal", "probability": 0.01}],
  "predicted_item": "plastic_bottle", "predicted_item_label": {"en": "Plastic bottle", "hi": "प्लास्टिक की बोतल"},
  "item_confidence": 0.81, "item_candidates": [{"item": "plastic_bottle", "probability": 0.81}],
  "guidance": {"name": {"en": "...", "hi": "..."}, "bin": "blue", "bin_label": {...}, "recyclable": "conditional",
               "segregation": {"en": [...], "hi": [...]}, "disposal": {...}, "recycling": {...}, "safety": {...},
               "common_mistakes": {...}, "environment": {...}, "color": "#3B8FD9", "icon": "cup-soda"},
  "notices": [{"code": "hazardous_scope", "severity": "danger", "message": {"en": "...", "hi": "..."}}],
  "quality_warnings": [{"code": "blurry", "message": {"en": "...", "hi": "..."}}],
  "image_url": "https://<project>.supabase.co/storage/v1/object/sign/...", "image_saved": true,
  "gradcam_image": "data:image/jpeg;base64,...",
  "model_name": "wastewise-classifier", "model_version": "1.1.0-mobilenet_v3_large", "inference_time_ms": 14.2
}
```
The values above only illustrate the response shape; they are not measured results.

Notice codes:

| Code | When it appears |
|---|---|
| `uncertain` | The prediction is not confident. |
| `hazardous_scope` | Hazardous is the prediction or the runner-up. The model only knows aerosol cans, so the UI must show this limitation. |
| `ewaste_scope` | The prediction is e-waste. |
| `battery_safety` | The item is a battery. |

### `GET /history`
Query parameters:

| Parameter | Values |
|---|---|
| `page` | ≥ 1 |
| `page_size` | 1–50, default 12 |
| `category` | one of the 7 categories |
| `confident` | `true` / `false` |
| `date_from`, `date_to` | `YYYY-MM-DD`, inclusive |
| `sort` | `newest`, `oldest`, `confidence_desc`, `confidence_asc` |

Returns `{items: [...], page, page_size, total}`. Each item includes a signed `image_url` that is valid for 1 hour.

### `GET /history/{id}`
Returns the full record:
- top 3 predictions
- stored bilingual disposal and safety text
- quality warnings
- current guidance and notices
- the user's feedback, if any

Returns 404 if the record doesn't exist *or* belongs to someone else.

### `DELETE /history/{id}`
Deletes the row and its stored image. Feedback is removed by cascade. Returns 204, or 404 if not found.

### `PUT /history/{id}/feedback`
Body: `{"user_correctness_feedback": bool, "corrected_category": "glass" | ... | "other" | null, "comments": "..."}`

- Creates or updates one feedback entry per prediction.
- `corrected_category` is allowed only when the answer was wrong.
- Feedback is stored for review only. `verified_label` can only be set by a reviewer with service-role access, so feedback never enters training automatically.

### `GET /analytics`
Statistics over the caller's scans:
- total, confident, uncertain and recyclable counts
- hazardous + e-waste count
- average confidence and most common category
- scans in the last 7 and 30 days
- category distribution
- daily counts for the last 30 days, with zero days filled in

### `GET /analytics/category-distribution`
Returns only the category distribution: `[{category, count, percentage}]`.

### `DELETE /account`
Deletes all of the caller's stored images and avatar, then deletes the auth user through the server-only service-role key. Profile, prediction and feedback rows are removed by cascade. Limited to 3 attempts per hour. Returns 204.

## Configuration
All settings are environment variables; see `backend/.env.example`. `SUPABASE_SERVICE_ROLE_KEY` is only needed for account deletion. Never expose it to the browser or commit it.
