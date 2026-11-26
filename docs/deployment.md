# Deployment guide

WasteWise runs as three parts:

| Part | Where | Notes |
|---|---|---|
| Frontend (Next.js 15) | Vercel project `wastewise` (root directory `frontend/`) | Proxies `/api/v1/*` to the API on the server side, so the browser only ever talks to one origin (no CORS, no API URL in client bundles) |
| API (FastAPI + ONNX Runtime) | Vercel project `wastewise-api` (root directory `backend/`, Vercel **FastAPI** framework preset, entry point `index.py`) | Same code runs under `uvicorn`/Docker (`backend/Dockerfile`), so it can move to Render, Railway, Fly.io or Cloud Run without changes |
| Database, auth, storage | Supabase project `tfcwyccbhspkxfttfefc` (ap-south-1) | Migration `supabase/migrations/20261125000001_initial_schema.sql` (RLS on every table, private buckets) |

## Environment variables

### `wastewise-api`

| Key | Value | Type |
|---|---|---|
| `ENVIRONMENT` | `production` (turns off `/docs`) | plain |
| `SUPABASE_URL` | `https://tfcwyccbhspkxfttfefc.supabase.co` | plain |
| `SUPABASE_ANON_KEY` | publishable key (`sb_publishable_…`) | plain |
| `SUPABASE_SERVICE_ROLE_KEY` | the project's service-role key (legacy JWT; new `sb_secret_…` keys are only shown once in the dashboard). Used **only** by `DELETE /api/v1/account` | **sensitive** |
| `CORS_ORIGINS` | the frontend production URL (only needed for direct browser calls; the proxy doesn't need it) | plain |

### `wastewise`

| Key | Value |
|---|---|
| `NEXT_PUBLIC_SUPABASE_URL` | `https://tfcwyccbhspkxfttfefc.supabase.co` |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | publishable key |
| `API_URL` | production URL of `wastewise-api`: `https://wastewise-api-three.vercel.app`. Read at **build** time by the `next.config.ts` rewrite, so redeploy the frontend after changing it |

## Supabase auth settings

* `site_url` = frontend production URL.
* `uri_allow_list` includes `<frontend>/auth/callback` (email confirmation and password reset both land there).
* Email confirmation is on. The built-in Supabase mailer sends at most **2 emails per hour for the whole project** and is meant for testing only. The production E2E run on 2026-11-26 hit this limit: sign-up failed with "email rate limit exceeded" after one real user's sign-up had used the hour's quota. The app now shows a friendly EN/HI message for this case, but the fix is custom SMTP (below). Tracked as Linear MAD-12.

### Custom SMTP (owner action, about 5 minutes)

SMTP needs a password that only the account owner can create, and the Supabase connector cannot set SMTP fields, so this step is done in the dashboard.

**Option A: Gmail (free, about 500 emails/day, good for a capstone demo)**

1. Turn on 2-Step Verification for the Google account: https://myaccount.google.com/security
2. Create an App Password at https://myaccount.google.com/apppasswords (name it "WasteWise Supabase"). Copy the 16-character password. It is shown only once; don't share it in chat or commit it.
3. Open Supabase → project **WasteWise** → **Authentication → Emails → SMTP Settings** (https://supabase.com/dashboard/project/tfcwyccbhspkxfttfefc/auth/smtp), turn on **Enable custom SMTP** and enter:

   | Field | Value |
   |---|---|
   | Sender email | your Gmail address (e.g. `madhurchaudhary65@gmail.com`) |
   | Sender name | `WasteWise` |
   | Host | `smtp.gmail.com` |
   | Port | `465` |
   | Username | the same Gmail address |
   | Password | the 16-character App Password (no spaces) |
   | Minimum interval between emails | `60` seconds (default) |

4. Save. Then open **Authentication → Rate Limits** (https://supabase.com/dashboard/project/tfcwyccbhspkxfttfefc/auth/rate-limits) and set **Rate limit for sending emails** to e.g. `30` per hour. This setting can only be raised once custom SMTP is on.
5. Test: sign up at https://wastewise-three-lilac.vercel.app/signup with a new address. The email should now come from "WasteWise <your Gmail>".

**Option B: Brevo (free, 300 emails/day, no personal Gmail involved)**

Create a Brevo account, verify a sender address, then under **SMTP & API → SMTP** generate an SMTP key. In Supabase use host `smtp-relay.brevo.com`, port `587`, username = your Brevo SMTP login, password = the SMTP key, and sender = the verified address.

(Resend also works, but it can only send to arbitrary recipients after you verify a domain you own.)

## Serverless notes (Vercel Python)

* The model (about 14 MB ONNX) loads lazily on the first request to each instance. Cold starts are slower, warm requests are fast.
* Request bodies are limited to 4.5 MB by Vercel, so the API limit is 4 MB. The frontend downscales large phone photos (to 2048 px max) before upload.
* The per-user rate limiter (`app/utils/rate_limit.py`) keeps state in memory. On serverless, each instance counts separately, so the limit is per instance. For strict limits, move it to Redis/Upstash or a Postgres counter.
* The ASGI lifespan may not run on serverless. `app/api/deps.py` therefore creates the model session and the Supabase HTTP client lazily if needed.

## Running locally

```bash
# API
cd backend && python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # fill SUPABASE_* values
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend && npm ci
cp .env.example .env.local      # API_URL=http://localhost:8000
npm run dev                     # http://localhost:3000
```

## Docker (any container host)

```bash
docker build -t wastewise-api backend/
docker run -p 8000:8000 --env-file backend/.env wastewise-api
```

The image is non-root and respects `$PORT` and `$WEB_CONCURRENCY`. Its health check calls `/livez`.

## Re-training and re-deploying a model

```bash
bash ml/scripts/fetch_datasets.sh && bash ml/scripts/fetch_pretrained.sh
python -m ml.src.prepare_data --raw-dir ml/datasets/raw --out-dir ml/datasets/processed
python -m ml.src.train --config ml/configs/mobilenet_v3.json      # and the other configs
python -m ml.src.evaluate --experiment exp02_mobilenet_v3_large    # for every experiment
python -m ml.src.select_model --version 1.0.1                      # writes backend/model_artifacts/*
cd backend && pytest                                               # includes real-model parity tests
```

Then commit `backend/model_artifacts/` and `ml/models/best_model.pth`, and redeploy the API.

## Live deployment (v1)

| Part | URL |
|---|---|
| Web app | https://wastewise-three-lilac.vercel.app (alias: `wastewise-madhurchaudhary65-7604s-projects.vercel.app`) |
| API | https://wastewise-api-three.vercel.app (`/api/v1/health`, `/api/v1/model-info`; `/__bootstrap` shows cold-start status) |

Supabase auth is set up as follows:

* `site_url` = `https://wastewise-three-lilac.vercel.app`.
* `uri_allow_list` = `<web>/auth/callback**` for both production aliases plus `http://localhost:3000/auth/callback**`.

### How v1 is deployed: file upload with pinned sources

The Vercel projects are **not linked to GitHub** yet. Linking needs a GitHub *Login Connection* on the Vercel account, and Vercel's anonymous GitHub fetches were rate-limited. Instead, `scripts/make_vercel_deploy.py` generates a few small bootstrap files in `deploy/vercel/{web,api}/`. These files are committed and reviewable, and they are uploaded with the Vercel deployments API:

* **web.** `fetch_source.sh` downloads the `frontend/` sources of one **pinned commit** from `codeload.github.com`. After that, `npm ci` and `next build` run normally, using the committed `package-lock.json`.
* **api.** `index.py` (FastAPI preset) is a tiny ASGI wrapper. On the first request to an instance it downloads the **GitHub commit archive** of the release commit (`https://github.com/<repo>/archive/<sha>.tar.gz`, byte-stable), checks its **SHA-256**, extracts only `backend/` (including `model_artifacts/`) to `/tmp`, and then hands every request to `app.main:app`. When the model files are loaded, they are checked again against the checksums in `model_card.json`. The bootstrap runs lazily in a worker thread (about 3.5 s on Vercel), so it never hits the function init limit; `GET /__bootstrap` reports `cold | bootstrapping | ready | failed` plus the error message, which makes cold-start problems diagnosable without log access.
* Deploy the API with project settings `{"framework": "fastapi"}`. Do **not** add a catch-all rewrite to `/api/index`: with backend framework presets Vercel passes the rewritten path to the app, so every route returns 404. Note that the Hobby plan ran the function in `iad1` even though `vercel.json` asks for `bom1`.

Redeploying:

```bash
# web: after pushing the commit you want live
python scripts/make_vercel_deploy.py web --sha $(git rev-parse HEAD)
# api: pin the commit archive (after pushing), then generate the entry point
SHA=$(git rev-parse HEAD)
curl -fsSL -o /tmp/ww.tar.gz https://github.com/Madhurchaudhary/Wastewise/archive/$SHA.tar.gz
python scripts/make_vercel_deploy.py api --bundle-url https://github.com/Madhurchaudhary/Wastewise/archive/$SHA.tar.gz --bundle-sha256 $(sha256sum /tmp/ww.tar.gz | cut -d' ' -f1)
# upload deploy/vercel/<target>/* as a production deployment (Vercel dashboard, CLI `vercel deploy --prod`, or API)
```

Once GitHub is connected in Vercel (Account Settings → Login Connections, then Project → Settings → Git), switch to normal Git deployments. Use root directory `frontend/` for `wastewise` and `backend/` for `wastewise-api`; `backend/vercel.json` (framework `fastapi`) and `backend/index.py` are already set up for that. After the switch, `deploy/vercel/` is no longer needed.
