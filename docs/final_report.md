# WasteWise: Final Project Report

**AI waste classification with Indian disposal guidance**
Samsung Innovation Campus capstone, November 2026

- Live web app: https://wastewise-three-lilac.vercel.app
- Live API: https://wastewise-api-three.vercel.app/api/v1/health
- Source: https://github.com/Madhurchaudhary/Wastewise

Every number in this report comes from `ml/reports/` in the repository. These files are produced by the training and evaluation code in `ml/src/`. No metric was typed in by hand, and no prediction is mocked.

---

## 1. Problem and goal

Household waste in India is still mostly mixed at source, even though the Solid Waste Management Rules 2016 require segregation into wet, dry and domestic-hazardous streams. Many people are unsure which bin an item belongs in. They also often don't know that e-waste and aerosol cans need special handling.

WasteWise lets a user photograph one waste item and get back:

1. A category out of 7: organic, paper & cardboard, plastic, metal, glass, e-waste, hazardous. The confidence score is calibrated.
2. A Grad-CAM heat-map showing which part of the image drove the decision.
3. Disposal guidance for India in English and Hindi: bin colour, steps, safety precautions, common mistakes and the relevant rules.
4. A private history and personal analytics, stored in Supabase.

**Design rule:** if the model is not confident, the app says so and asks for a better photo. It never forces a label.

## 2. Architecture

| Layer | Technology | Notes |
|---|---|---|
| ML | PyTorch 2.x and torchvision; ONNX Runtime for inference | Three architectures compared; the winner is exported to ONNX |
| API | FastAPI (Python 3.12) on Vercel | Lazy-bootstrap wrapper; rate limits of 10/min and 300/day per user; 4 MB upload cap |
| Data | Supabase: Auth, Postgres with RLS, private Storage | Each request uses the user's own JWT, so RLS applies. The service key is used only to delete an account |
| Web | Next.js 15, TypeScript, Tailwind, shadcn/ui on Vercel | `/api/v1/*` is proxied server-side to the API; EN/HI; dark theme; respects reduced-motion |

**Pages:** landing, sign-up/login/forgot-password, dashboard, analyze, result, history, analytics, waste guide, settings.

**API endpoints:**
- `GET /health`, `GET /model-info`
- `POST /predict`
- `GET /history`, `GET /history/{id}`, `DELETE /history/{id}`, `PUT /history/{id}/feedback`
- `GET /analytics`, `GET /analytics/category-distribution`
- `DELETE /account`

## 3. Dataset

- **Sources:** 24,284 raw images from public datasets, mapped onto a 7-category, 27-item Indian taxonomy (`ml/src/taxonomy.py`).
- **Cleaning:** exact and near-duplicates were removed. The split was then grouped so that near-duplicates never land on both sides of the train/test boundary.
- **Final split:**

  | Split | Images |
  |---|---|
  | Train | 4,804 |
  | Validation | 1,037 |
  | Test | 1,044 |
  | Out-of-distribution probes (non-waste images, used to test rejection) | 600 |

- **Hazardous class:** 199 / 43 / 43 images (train / validation / test). It covers aerosol cans only.
- **Licensing:** some sources (household30, EWasteNet) are non-commercial / academic only. WasteWise is therefore an academic project. The terms are documented in the README, the model card and `docs/ml_documentation.md`.

## 4. Experiments

All training ran on CPU only (2 cores, about 3.8 GB RAM) with a budget of about 50 minutes per model.

| Experiment | Params | Epochs | Train min | Val acc | Val macro-F1 |
|---|---|---|---|---|---|
| exp01 Custom CNN (from scratch) | 1.96M | 12 | 46.6 | 0.526 | 0.495 |
| exp02 MobileNetV3-Large (ImageNet) | 3.48M | 15 | 47.6 | 0.915 | 0.905 |
| exp03 EfficientNet-B0 (ImageNet) | 4.68M | 10 (early stop, best epoch 6) | 52.3 | 0.914 | 0.908 |
| exp04 MobileNetV3-Large, dataset v1.1 (warm start from exp02) | 3.48M | 6 (best epoch 4) | 30.5 (on top of exp02) | 0.935 | 0.925 |

**Selection rule (fixed before looking at the test set):** pick the highest validation macro-F1; if two models are within 0.5 percentage points, pick the one with lower CPU latency. In the first round exp02 and exp03 were within 0.3 pp, so **MobileNetV3-Large** was selected (18.1 ms vs 30.1 ms per image in PyTorch on CPU).

**Accuracy round (dataset v1.1, 2026-11-26).** The training split was expanded from 4,804 to 10,174 images (`ml/src/expand_train.py`). Validation, test and OOD splits were kept byte-identical (MD5-checked), so results are directly comparable. New images that were near-duplicates of any validation, test or OOD image were dropped (259). Hazardous could not grow (199 training images; the source is exhausted). MobileNetV3-Large was then fine-tuned from the exp02 weights (exp04). Under the same rule exp04 wins clearly: validation macro-F1 0.925 vs 0.905.

## 5. Results on the held-out test set (1,044 images, evaluated once)

| Model | Test acc | Test macro-F1 | ROC-AUC | Hazardous recall | Threshold | Coverage | Acc on accepted | OOD rejected |
|---|---|---|---|---|---|---|---|---|
| exp01 Custom CNN | 0.553 | 0.518 | 0.856 | 0.395 | 0.87 | 3.9% | 92.7% | 99.2% |
| **exp02 MobileNetV3-Large (deployed)** | **0.923** | **0.919** | **0.994** | **0.977** | 0.59 | 94.6% | 94.6% | 35.7% |
| exp03 EfficientNet-B0 | 0.907 | 0.900 | 0.990 | 0.977 | 0.59 | 93.8% | 93.4% | 33.7% |
| **exp04 MobileNetV3-Large v1.1 (deployed, v1.1.0)** | **0.926** | **0.922** | **0.994** | 0.953 | 0.65 | 94.4% | **95.6%** | **37.8%** |

exp02 is the previous deployment (v1.0.0). The test-set gain of exp04 is small (+0.3 pp accuracy, +0.2 pp macro-F1; about 3 images). The clearer improvements are in accuracy on accepted images (95.6% vs 94.6% at the same ~94.5% coverage), metal F1 (0.895 vs 0.880), e-waste F1 (0.985 vs 0.973), plastic recall (0.917 vs 0.875) and calibration (test ECE 0.020 vs 0.025). Hazardous recall fell from 42/43 to 41/43.

**Threshold policy.** The validation rule (smallest threshold reaching 95% selective accuracy) gave exp04 a threshold of 0.52. That would have accepted more images but lowered OOD rejection to 24.3%. Model upgrades must not loosen rejection, so the deployed threshold is 0.65, the documented default. At 0.65 exp04 has the same validation coverage as exp02 (0.932 vs 0.934), higher validation selective accuracy (0.965 vs 0.951) and higher OOD rejection. The OOD probe set informed this decision, so its 37.8% figure is not an unbiased estimate. Both evaluations are kept (`ml/reports/exp04_mobilenet_v3_large_v11/evaluation.json` and `evaluation_rule_threshold_0.52.json`).

**Test-time augmentation was evaluated but not adopted.** Averaging the logits of the image and its horizontal flip raised validation macro-F1 from 0.925 to 0.933. It barely changed out-of-distribution separability (AUROC 0.821 vs 0.820) and would double inference cost, so it is left as future work (`ml/src/tta_check.py`).

**Deployed model details:**

- **Calibration:** temperature scaling with T = 0.661 reduced test ECE from 0.085 to 0.020.
- **Per-class results (test):**

  | Class | F1 | Recall |
  |---|---|---|
  | Organic | 0.941 | 0.915 |
  | Paper | 0.945 | 0.964 |
  | Plastic | 0.895 | 0.917 |
  | Metal | 0.895 | 0.879 |
  | Glass | 0.909 | 0.887 |
  | E-waste | 0.985 | 0.988 |
  | Hazardous | 0.882 | 0.953 |

- **Size and speed:** 13.9 MB ONNX file; 4.7 ms median ONNX Runtime latency on CPU.
- **Rejection rule:** a prediction is accepted only if the calibrated confidence is at least 0.65 and the gap between the top two classes is at least 0.15. See the threshold policy above.

**Real-photo check (v1.0.0 ONNX model, run locally):** a real photo containing a PET bottle, crumpled paper and a soda can was classified as *plastic* (item: plastic bottle) with 0.992 confidence, in 5.65 ms.

## 6. Limitations (stated honestly)

1. **Out-of-distribution images.** The confidence and margin gate rejects only 37.8% of non-waste probe images (35.7% for v1.0.0). A photo of something that is not waste can still receive a label. The custom CNN's 99% rejection rate is not better behaviour: that model is unsure about almost everything, so it accepts only 3.9% of real test images.
2. **One label per photo.** In the multi-object photo above, only the bottle was labelled; the paper and the can were not. The UI asks for one item per photo.
3. **Narrow hazardous class.** Hazardous detection covers aerosol cans only. The result page shows a notice about this. Other hazardous items (paint, pesticide, medicines, CFLs) are covered in text in the Waste Guide.
4. **Dataset bias.** Most images are web or studio photos with clean backgrounds. Indian street conditions (wet, dirty, partly hidden items) are under-represented. Clear PET bottles versus glass is the most common confusion.
5. **Operations.** The built-in Supabase mailer is rate-limited, so custom SMTP is needed before a public launch. The free Vercel tier has cold starts: the first request after idle loads the model, which took about 3.6 s in our test.

## 7. Future work

- Collect and label photos taken in Indian homes. Fine-tune with the user-feedback table, after manual review.
- Add a dedicated OOD detector (for example energy-based or Mahalanobis) or a "not waste" class.
- Add multi-object detection (YOLO-style) for mixed bins.
- Broaden the hazardous class: batteries, paint, medicines, CFL/tube lights.
- Make the app an installable PWA with on-device ONNX inference (the model is 14 MB).

## 8. Testing and deployment

- **Backend:** 50 automated tests (pytest), all passing. They cover units, the API contract and the real ONNX model.
- **Production smoke checks (25–26 Nov 2026):**
  - `/__bootstrap` reported `ready`.
  - `/api/v1/health` reported `status: ok` with the model loaded.
  - `/api/v1/model-info` returned the model card.
  - Requests through the web proxy reached the API.
  - The login and Waste Guide pages rendered.
- **Results recorded in:**
  - Supabase table `model_metrics` (3 rows)
  - Google Sheets experiment tracker
  - Drive `WasteWise/Model_Evaluation` (confusion matrix, classification report, curves, Grad-CAM sample)
  - Drive `Model_Experiments` (checkpoints)
  - Notion knowledge base
  - Linear project WasteWise (M1–M8)

## 9. Reproducing

See the `README.md` quick start and `docs/deployment.md`. Rebuild the datasets with `ml/scripts/fetch_datasets.sh`. Then run `python -m ml.src.train`, `python -m ml.src.evaluate` and `python -m ml.src.select_model --version 1.1.0` (dataset v1.1: `python -m ml.src.expand_train`). These regenerate every file in `ml/reports/` and the ONNX model in `backend/model_artifacts/`.
