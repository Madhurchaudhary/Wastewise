# Production end-to-end check

`prod-e2e.mjs` drives a real Chromium browser against the **live** deployment
(https://wastewise-three-lilac.vercel.app). It uses the real API, the real model,
real Supabase Auth/Postgres/Storage and a real confirmation email. Nothing is mocked.

| # | Step | What it proves |
|---|---|---|
| 1 | Health through the web proxy | Web → API rewrite, model loaded |
| 2 | Sign up in the UI | Supabase sign-up, confirmation email sent |
| 3 | Password login before confirming | Unconfirmed accounts are blocked |
| 4–5 | Open the real email link | Email delivery, PKCE `/auth/callback`, redirect to dashboard |
| 6 | Log out, then log in with the form | Middleware protects pages; login works |
| 7 | Upload a photo | Real ONNX prediction, saved with its image (201) |
| 8–9 | Grad-CAM + stored image | Heatmap renders; private image served via a signed URL and not marked shared-cacheable (the caching headers are recorded) |
| 10–11 | History + Analytics | The scan is listed and counted (UI + API with the user's JWT) |
| 12 | Switch to हिंदी | Nav, result and disposal guidance render in Hindi |
| 13 | Delete the scan (in Hindi) | Row gone (404), image no longer served |
| 14–16 | Delete the account | Account removed; login now fails |

## How a run works

1. Pushing a change to `e2e/` or the workflow (or running **Production E2E**
   manually) starts `.github/workflows/e2e-production.yml`.
2. The test signs up `madhurchaudhary65+wastewise-e2e@gmail.com` with a random,
   never-logged password and waits up to 25 minutes for the confirmation link.
3. An operator copies the link from the email and posts it as a commit comment
   on the commit under test: `E2E-LINK-B64 <base64 of the url>`, e.g. `echo "E2E-LINK-B64 $(printf %s "$URL" | base64 -w0)"`. Delete the comment afterwards.
   Do not post the plain URL. In run 45777888257 the email was confirmed about 5 s after the plain link was posted, before the runner opened it. The source is unidentified: no GitHub notification email was found, so it was probably another automated link fetcher. The runner then got `otp_expired` because the one-time token was already used.
4. The results (`results.json` plus one screenshot per step) are committed to
   `e2e/results/latest/`.

The built-in Supabase mailer sends about 2 emails per hour, so wait between runs
unless custom SMTP is configured (see `docs/deployment.md` → Custom SMTP).

### Run 45779096142

Steps 1–9 passed, including the confirmation link relayed as `E2E-LINK-B64`. The signed-URL step then failed: the stored object has `cacheControl = private, no-store, max-age=0` in `storage.objects`, but Supabase's storage edge sent the image with no `cache-control` header. The step now fails only when the image is explicitly shared-cacheable. The real check is the delete step: the signed URL must stop serving the image right after the scan is deleted.
