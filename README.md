# WasteWise ♻️

**AI-powered household waste classification with Indian disposal guidance in English and Hindi.**

Samsung Innovation Campus capstone project.

Upload a photo of a single waste item and WasteWise will:

1. **Classify it** into one of 7 categories with a fine-tuned CNN served through ONNX Runtime: organic, paper & cardboard, plastic, metal, glass, e-waste and hazardous.
2. **Report calibrated confidence.** Confidence is temperature-scaled. Uncertain or out-of-scope photos are **rejected**, not force-labelled.
3. **Show a Grad-CAM heat-map** of the pixels that drove the decision.
4. **Explain disposal for India.** This covers bin colour, the SWM Rules 2016, E-Waste Rules 2022 and BMW guidance, and dos and don'ts. Guidance is in **English and हिन्दी**.
5. **Save the result** to a private, per-user history with analytics. This uses Supabase Auth, Postgres with Row Level Security and private Storage.

> No metric in this repository is hand-typed. Every number comes from code in `ml/src` and is stored under `ml/reports/`. The API never returns mock predictions: if the model is not loaded, `/predict` returns `503`.

---

## Architecture

```
Browser ──► Next.js 15 (Vercel: wastewise) ──/api/v1/* proxy──► FastAPI + ONNX Runtime (Vercel/Docker: wastewise-api)
   │                                                                   │
   └──── supabase-js (auth session) ────► Supabase (Auth, Postgres+RLS, Storage) ◄── user JWT ─┘
```

| Layer | Tech | Where |
|---|---|---|
| ML | PyTorch 2, torchvision, scikit-learn, ONNX | `ml/` |
| API | FastAPI, Pydantic v2, onnxruntime, httpx | `backend/` |
| Data | Supabase Postgres (RLS on every table), Storage (private buckets), Auth (ES256 JWT) | `supabase/migrations/` |
| Web | Next.js 15 App Router, React 19, TypeScript, Tailwind CSS v4, shadcn/ui-style components, Recharts, next-themes | `frontend/` |
| Shared | Taxonomy and bilingual disposal knowledge base, generated into the frontend and backend | `shared/`, `scripts/sync_shared.py` |

Detailed documentation:

- [`docs/final_report.md`](docs/final_report.md): final project report (results, limitations, future work)
- [`docs/implementation_plan.md`](docs/implementation_plan.md)
- [`docs/ml_documentation.md`](docs/ml_documentation.md)
- [`docs/api_documentation.md`](docs/api_documentation.md)
- [`docs/deployment.md`](docs/deployment.md)
- [`docs/project_tracking.md`](docs/project_tracking.md)

## Repository layout

```
ml/          data prep, training, evaluation, model selection, ONNX export, notebooks, reports
  src/       taxonomy · prepare_data · train · evaluate · select_model · inference · metrics
  configs/   one JSON per experiment (custom_cnn, mobilenet_v3, efficientnet_b0)
  datasets/  manifests only (raw images are NOT committed)
  models/    best_model.pth + experiments/<id>/training_history.json
  reports/   dataset report, per-model evaluation, comparison, curves
backend/     FastAPI app (app/), tests, Dockerfile, Vercel entry (index.py), model_artifacts/
frontend/    Next.js app (src/app, src/components, src/lib), vitest tests
supabase/    SQL migrations (schema, RLS policies, storage buckets, analytics views)
shared/      waste_knowledge.json (EN/HI disposal guidance), taxonomy
docs/        architecture, ML, API, deployment and tracking documents
```

## Model results

<!-- README-RESULTS:START -->
| Experiment | Params | Epochs | Train min | Val macro-F1 | Test acc | Test macro-F1 | ROC-AUC | Hazardous recall | Coverage @ thr | Acc on accepted | OOD rejected | CPU ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exp01_custom_cnn | 1.96 M | 12 | 47 | 0.4951 | 0.5527 | 0.5184 | 0.8555 | 0.395 | 3.9 % | 92.7 % | 99.2 % | 20 |
| exp02_mobilenet_v3_large | 3.48 M | 15 | 48 | 0.9049 | 0.9234 | 0.9192 | 0.9935 | 0.977 | 94.6 % | 94.6 % | 35.7 % | 18 |
| exp03_efficientnet_b0 | 4.68 M | 10 | 52 | 0.9078 | 0.9071 | 0.9000 | 0.9904 | 0.977 | 93.8 % | 93.4 % | 33.7 % | 30 |
| **exp04_mobilenet_v3_large_v11** (selected) | 3.48 M | 6 | 30 | 0.9252 | 0.9262 | 0.9216 | 0.9943 | 0.954 | 94.4 % | 95.6 % | 37.8 % | 15 |

_exp04_mobilenet_v3_large_v11: warm-started from the previously deployed model and fine-tuned on dataset v1.1 (same validation/test/OOD splits, larger training split). Its 30 training minutes come on top of the starting model's own training._

_exp01_custom_cnn: its high OOD-rejection figure is not a strength. The model is unconfident on almost everything (only 3.9 % of in-scope test images pass its threshold of 0.87)._

**Selected model: `exp04_mobilenet_v3_large_v11`** (mobilenet_v3_large, model version `1.1.0-mobilenet_v3_large`).

- **Test set** (1044 images): accuracy 0.9262, macro-F1 0.9216, weighted F1 0.9263, macro ROC-AUC 0.9943.
- **Calibration:** temperature 0.6606; test ECE 0.0849 → 0.0202.
- **Rejection:** threshold 0.65 accepts 94.4 % of test images, with 95.6 % accuracy on those. 37.8 % of 600 out-of-scope probe images are rejected.
- **Threshold policy:** the validation rule gave 0.52; the deployed threshold is 0.65. Reason: Non-regression policy for model upgrades: the validation rule gave 0.52, which would lower OOD rejection versus the deployed exp02 (0.357). 0.65 (the documented default) gives exp04 validation coverage 0.932 vs exp02 0.934 at 0.59, with higher validation selective accuracy (0.965 vs 0.951) and higher OOD rejection (0.378 vs 0.357). The OOD set informed this choice, so its OOD figure is not an unbiased estimate.
- **Top confusions:** plastic→glass (8), organic→paper_cardboard (7), metal→plastic (7), glass→plastic (7), glass→metal (7).
- **Accuracy by source dataset:** ewastenet 0.984 (n=128), garbage12 0.942 (n=171), household30:default 0.920 (n=375), household30:real_world 0.902 (n=297), trashnet 0.918 (n=73).
- **Latency (2 vCPU, no GPU):** ONNX Runtime 5 ms, PyTorch 15 ms median per image.

| Category | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| organic | 0.968 | 0.915 | 0.941 | 165 |
| paper_cardboard | 0.926 | 0.964 | 0.945 | 168 |
| plastic | 0.875 | 0.917 | 0.895 | 168 |
| metal | 0.912 | 0.879 | 0.895 | 165 |
| glass | 0.931 | 0.887 | 0.909 | 168 |
| e_waste | 0.982 | 0.988 | 0.985 | 167 |
| hazardous | 0.820 | 0.953 | 0.882 | 43 |

Selection uses validation macro-F1 only; test numbers are reported for information. All figures are produced by `ml/src/evaluate.py` and `ml/src/select_model.py`. See `ml/reports/` for confusion matrices, selective-prediction curves and misclassified examples.
<!-- README-RESULTS:END -->

How the deployed model is chosen:

- **Selection rule:** highest validation macro-F1. If two models are within 0.5 pp, the lower CPU latency wins.
- **Confidence threshold:** the smallest value in [0.5, 0.9] that gives ≥ 95 % selective accuracy on validation.
- **Margin rule:** the top-2 probability margin must also be ≥ 0.15.

## Quick start

### 1. API

```bash
cd backend
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # SUPABASE_URL, SUPABASE_ANON_KEY (+ SUPABASE_SERVICE_ROLE_KEY for account deletion)
uvicorn app.main:app --reload --port 8000     # docs at http://localhost:8000/docs
pytest                                        # unit + API + real-model tests
```

### 2. Frontend

```bash
cd frontend
npm ci
cp .env.example .env.local      # NEXT_PUBLIC_SUPABASE_*, API_URL=http://localhost:8000
npm run dev                     # http://localhost:3000
npm run typecheck && npm run lint && npm test
```

### 3. Docker (API, any container host)

```bash
docker build -t wastewise-api backend
docker run -p 8000:8000 --env-file backend/.env wastewise-api
```

## Reproducing the ML pipeline

The steps below run on CPU. A full three-model run takes about 2.5 h on 2 cores.

```bash
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchvision==0.29.2
pip install -r ml/requirements.txt

bash ml/scripts/fetch_datasets.sh          # downloads the 4 public datasets (needs Kaggle/GitHub access)
bash ml/scripts/fetch_pretrained.sh        # ImageNet weights, SHA-256 verified
python -m ml.src.prepare_data              # dedupe, group-aware 70/15/15 split -> dataset v1
python -m ml.src.train --config ml/configs/mobilenet_v3.json      # also custom_cnn, efficientnet_b0
python -m ml.src.expand_train --raw-dir data/raw --out-dir data/processed_v11   # dataset v1.1: larger train split, same val/test/OOD
python -m ml.src.evaluate --experiment exp04_mobilenet_v3_large_v11   # one per experiment
python -m ml.src.select_model --version 1.1.0   # comparison, best_model.pth, ONNX + Grad-CAM head + model card
python -m pytest ml/tests
```

The notebooks in `ml/notebooks/` (01 dataset exploration, 02 training, 03 evaluation, 04 inference & explainability) read the same reports and are executed with outputs saved.

## Main features

- **Pages:** landing, sign-up / login / password reset, dashboard, analyze (drag-and-drop or camera), result, history (filter, sort, paginate, delete), analytics, waste guide (all categories, including hazardous items the model does not detect), profile and settings.
- **Accessibility and preferences:** EN / हिन्दी toggle (persisted), dark theme, `prefers-reduced-motion` support, keyboard-accessible components.
- **Security:**
  - The API acts with the **user's JWT**, so Postgres RLS enforces data ownership.
  - The Supabase secret key is used only to delete an auth user on account deletion.
  - Images are kept in private buckets and served through short-lived signed URLs.
  - Uploads are checked for size, MIME type and magic bytes, EXIF is stripped, and requests are rate-limited.
  - CSP and security headers are set.
- **Honest uncertainty:** low-confidence predictions are shown as "not sure" together with photo tips. The result page carries a hazardous-class limitation notice.

## Known limitations

- The model expects **one item per photo**. With several items in one photo it reports the most dominant one.
- **Hazardous detection covers aerosol cans only.** The training data contains no other hazardous items, such as paint, chemicals, medicines or sharps. The Waste Guide covers these in text.
- Clear PET bottles and crumpled paper are the hardest cases.
- Rate limiting is in memory, so it is per instance on serverless.
- The built-in Supabase mailer is rate-limited. Configure custom SMTP before public launch.

## Datasets and licence notices ⚠️

WasteWise is trained on four public datasets. **Two of them restrict use to non-commercial, academic or research purposes.** The trained weights (`ml/models/best_model.pth`, `backend/model_artifacts/wastewise.onnx`) are therefore released for **non-commercial, educational and research use only**.

| Dataset | Source | Licence / terms |
|---|---|---|
| TrashNet | Thung & Yang, Stanford CS229 (github.com/garythung/trashnet) | MIT |
| Garbage Classification (12 classes) | Kaggle, mostafaabla | ODbL 1.0 (database); images © their owners |
| Recyclable and Household Waste Classification | Kaggle, Alistair King | **Educational / research / non-commercial use only**; citation required |
| EWasteNet | Nifulislam (GitHub) | **No explicit licence**; used for academic purposes with citation only |

- Raw images are **not** redistributed in this repository. Only file manifests are committed.
- Commercial use of WasteWise would require retraining on data licensed for that purpose.
- See [`ml/datasets/README.md`](ml/datasets/README.md) and the model card (`backend/model_artifacts/model_card.json`).

**Code licence:** the source code is provided for the Samsung Innovation Campus capstone. The dataset terms above apply to the model weights regardless of the code licence.

## Disposal-guidance disclaimer

Guidance is general and based on the Solid Waste Management Rules 2016, the E-Waste (Management) Rules 2022 and CPCB guidance. Local municipal rules take precedence; check with your ULB or ward office.
