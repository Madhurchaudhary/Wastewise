// Production end-to-end check for WasteWise (MAD-14).
//
// Drives a real Chromium browser against the LIVE deployment:
//   sign up -> real confirmation email link -> dashboard -> log out / log in
//   -> upload a real photo -> real model prediction saved -> history -> analytics
//   -> switch to Hindi -> delete the scan (+ image) -> delete the account.
//
// The confirmation link is relayed by an operator as a commit comment
// ("E2E-LINK <url>") on the commit under test, because the link arrives in a
// real inbox. Nothing is mocked. Every step is recorded in results.json and
// screenshots are saved to OUT_DIR.
//
// Env: BASE_URL, SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, E2E_EMAIL, OUT_DIR,
//      GITHUB_REPOSITORY, GITHUB_SHA, GITHUB_TOKEN, LINK_WAIT_SECONDS (default 1500)
import { chromium } from "playwright";
import { randomBytes } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

const BASE = process.env.BASE_URL.replace(/\/$/, "");
const SB_URL = process.env.SUPABASE_URL;
const SB_KEY = process.env.SUPABASE_PUBLISHABLE_KEY;
const EMAIL = process.env.E2E_EMAIL;
const OUT = process.env.OUT_DIR;
const PHOTO = path.resolve("ml/reports/samples/user_photo_input.jpg");
// Random per-run password, never printed or stored.
const PASSWORD = `Ww1-${randomBytes(18).toString("base64url")}`;
const LINK_WAIT = Number(process.env.LINK_WAIT_SECONDS ?? 1500);

mkdirSync(OUT, { recursive: true });
const results = { started_at: new Date().toISOString(), base_url: BASE, email: EMAIL, steps: [], facts: {} };
const save = () => writeFileSync(path.join(OUT, "results.json"), JSON.stringify(results, null, 2));

let page;
let n = 0;
async function step(name, fn, { soft = false } = {}) {
  const t0 = Date.now();
  const rec = { n: ++n, name, ok: false, ...(soft ? { soft: true } : {}) };
  results.steps.push(rec);
  try {
    rec.detail = (await fn()) ?? null;
    rec.ok = true;
  } catch (e) {
    rec.error = String(e?.message ?? e).slice(0, 600);
  }
  rec.ms = Date.now() - t0;
  const shot = `${String(rec.n).padStart(2, "0")}-${name.replace(/[^a-z0-9]+/gi, "-").toLowerCase()}.png`;
  try { await page.screenshot({ path: path.join(OUT, shot), fullPage: false }); rec.screenshot = shot; } catch { /* ignore */ }
  console.log(`${rec.ok ? "PASS" : "FAIL"} ${rec.n}. ${name}${rec.ok ? "" : ` -> ${rec.error}`}`);
  save();
  if (!rec.ok && !soft) throw new Error(`step failed: ${name}`);
  return rec.detail;
}

function assert(cond, msg) { if (!cond) throw new Error(msg); }

async function passwordGrant() {
  const r = await fetch(`${SB_URL}/auth/v1/token?grant_type=password`, {
    method: "POST",
    headers: { apikey: SB_KEY, "Content-Type": "application/json" },
    body: JSON.stringify({ email: EMAIL, password: PASSWORD }),
  });
  return { status: r.status, body: await r.json().catch(() => null) };
}

async function waitForLink() {
  const api = `https://api.github.com/repos/${process.env.GITHUB_REPOSITORY}/commits/${process.env.GITHUB_SHA}/comments`;
  const deadline = Date.now() + LINK_WAIT * 1000;
  while (Date.now() < deadline) {
    const r = await fetch(api, { headers: { Authorization: `Bearer ${process.env.GITHUB_TOKEN}`, Accept: "application/vnd.github+json" } });
    if (r.ok) {
      const comments = await r.json();
      const c = comments.find((x) => x.body?.trim().startsWith("E2E-LINK "));
      if (c) return c.body.trim().slice("E2E-LINK ".length).trim();
    }
    await new Promise((res) => setTimeout(res, 10_000));
  }
  throw new Error(`no confirmation link relayed within ${LINK_WAIT}s`);
}

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1280, height: 900 }, locale: "en-IN" });
page = await context.newPage();
let bearer = null;
page.on("request", (req) => {
  const a = req.headers()["authorization"];
  if (a && req.url().includes("/api/v1/")) bearer = a;
});
const apiGet = (p) => context.request.get(`${BASE}/api/v1${p}`, { headers: { Authorization: bearer } });

let exitCode = 0;
try {
  await step("health through web proxy", async () => {
    const r = await context.request.get(`${BASE}/api/v1/health`);
    const b = await r.json();
    assert(r.ok() && b.status === "ok" && b.model_loaded, `health ${r.status()} ${JSON.stringify(b)}`);
    return b;
  });

  await step("sign up through the UI", async () => {
    await page.goto(`${BASE}/signup`);
    await page.fill("#name", "WasteWise E2E");
    await page.fill("#email", EMAIL);
    await page.fill("#password", PASSWORD);
    await page.fill("#confirm", PASSWORD);
    await page.getByRole("button", { name: "Create account" }).click();
    const sent = page.getByRole("status").filter({ hasText: "Check your email" });
    const limited = page.getByText("hourly email limit reached");
    await sent.or(limited).first().waitFor({ timeout: 20_000 });
    if (await limited.isVisible()) {
      throw new Error("Supabase email rate limit reached (built-in mailer: 2 emails/hour). Friendly message shown correctly; configure custom SMTP or retry later.");
    }
  });

  await step("login blocked before email confirmation", async () => {
    const g = await passwordGrant();
    assert(g.status === 400, `expected 400 before confirmation, got ${g.status}`);
    return { status: g.status, error: g.body?.error_code ?? g.body?.msg ?? null };
  });

  let link = null;
  await step("receive real confirmation email link", async () => {
    link = await waitForLink();
    assert(link.startsWith(`${SB_URL}/auth/v1/verify`), "unexpected link host");
    return { host: new URL(link).host, type: new URL(link).searchParams.get("type") };
  });

  await step("open confirmation link -> auth callback -> dashboard", async () => {
    // Diagnostics: names (never values) of the Supabase cookies present before the callback.
    const cookieNames = (await context.cookies()).map((c) => c.name).filter((x) => x.startsWith("sb-"));
    results.facts.cookies_before_callback = cookieNames;
    await page.goto(link);
    await page.waitForURL(/\/(dashboard|login)/, { timeout: 30_000 });
    const landed = new URL(page.url());
    results.facts.callback_landing = landed.pathname + landed.search;
    if (landed.pathname.startsWith("/login")) {
      await page.getByRole("status").or(page.locator("form")).first().waitFor();
      throw new Error(`callback did not create a session: ${landed.search} (sb cookies before: ${cookieNames.join(", ") || "none"})`);
    }
    await page.getByText("Hello").first().waitFor({ timeout: 20_000 });
    return { url: landed.pathname, cookies_before_callback: cookieNames };
  }, { soft: true });

  await step("log out and log back in with the login form", async () => {
    await context.clearCookies();
    await page.evaluate(() => window.localStorage.clear());
    await page.goto(`${BASE}/dashboard`);
    await page.waitForURL(/\/login/, { timeout: 20_000 });
    await page.fill("#email", EMAIL);
    await page.fill("#password", PASSWORD);
    await page.getByRole("button", { name: "Log in" }).click();
    await page.waitForURL(/\/dashboard/, { timeout: 20_000 });
    return { protected_redirect: "ok" };
  });

  const pred = await step("upload photo -> real prediction saved", async () => {
    await page.goto(`${BASE}/analyze`);
    await page.locator('input[type="file"]').first().setInputFiles(PHOTO);
    const respP = page.waitForResponse((r) => r.url().includes("/api/v1/predict") && r.request().method() === "POST", { timeout: 90_000 });
    await page.getByRole("button", { name: "Analyze", exact: true }).click();
    const resp = await respP;
    const body = await resp.json();
    assert(resp.status() === 201, `predict status ${resp.status()} ${JSON.stringify(body).slice(0, 300)}`);
    await page.waitForURL(/\/result\//, { timeout: 30_000 });
    await page.getByRole("heading", { level: 2 }).first().waitFor();
    await page.getByText("How to dispose").waitFor();
    const facts = {
      id: body.id, category: body.predicted_category, confidence: body.confidence, is_confident: body.is_confident,
      item: body.predicted_item, item_confidence: body.item_confidence, model_version: body.model_version,
      inference_ms: body.inference_time_ms, image_saved: body.image_saved, has_gradcam: Boolean(body.gradcam_image),
      top: (body.top_predictions ?? []).slice(0, 3),
    };
    results.facts.prediction = facts;
    results.facts.image_url = body.image_url;
    assert(body.image_saved && body.image_url, "image was not saved");
    return facts;
  });

  await step("Grad-CAM heatmap rendered", async () => {
    const img = page.getByAltText("Grad-CAM heatmap over the uploaded image");
    await img.waitFor();
    const w = await img.evaluate((el) => el.naturalWidth);
    assert(w > 0, "heatmap image did not load");
    return { natural_width: w };
  });

  await step("stored image reachable via signed URL", async () => {
    const r = await context.request.get(results.facts.image_url);
    assert(r.status() === 200, `signed URL status ${r.status()}`);
    return { status: r.status(), content_type: r.headers()["content-type"] };
  });

  await step("scan appears in History (UI + API)", async () => {
    await page.goto(`${BASE}/history`);
    await page.locator(`a[href="/result/${pred.id}"]`).waitFor({ timeout: 20_000 });
    const r = await apiGet("/history?page=1&page_size=10");
    const b = await r.json();
    assert(r.ok() && b.items.some((i) => i.id === pred.id), "prediction missing from /history");
    return { total: b.total };
  });

  await step("Analytics reflect the scan", async () => {
    await page.goto(`${BASE}/analytics`);
    await page.getByText("Category distribution").waitFor({ timeout: 20_000 });
    const r = await apiGet("/analytics");
    const b = await r.json();
    assert(r.ok() && b.total_scans >= 1, `analytics ${JSON.stringify(b).slice(0, 200)}`);
    return { total_scans: b.total_scans, most_common_category: b.most_common_category, distribution: b.category_distribution };
  });

  await step("switch to Hindi: nav, result and guidance in Hindi", async () => {
    await page.locator("header button", { hasText: "हिं" }).first().click();
    await page.waitForFunction(() => document.documentElement.lang === "hi");
    await page.goto(`${BASE}/result/${pred.id}`);
    await page.getByText("कैसे निपटाएँ").waitFor({ timeout: 20_000 });
    await page.locator("header").getByText("इतिहास").first().waitFor();
    const heading = await page.getByRole("heading", { level: 2 }).first().innerText();
    return { html_lang: await page.evaluate(() => document.documentElement.lang), heading };
  });

  await step("delete the scan (in Hindi) -> row and image removed", async () => {
    await page.getByRole("button", { name: "हटाएँ" }).first().click();
    await page.getByRole("dialog").getByRole("button", { name: "हटाएँ" }).click();
    await page.waitForURL(/\/history$/, { timeout: 20_000 });
    const d = await apiGet(`/history/${pred.id}`);
    assert(d.status() === 404, `deleted scan still returns ${d.status()}`);
    const img = await context.request.get(results.facts.image_url);
    assert(img.status() !== 200, `deleted image still served (${img.status()})`);
    return { history_detail_status: d.status(), signed_url_status_after_delete: img.status() };
  });

  await step("switch back to English", async () => {
    await page.locator("header button", { hasText: "EN" }).first().click();
    await page.waitForFunction(() => document.documentElement.lang === "en");
  });

  await step("delete the account from Settings", async () => {
    await page.goto(`${BASE}/settings`);
    await page.getByRole("button", { name: "Delete account" }).first().click();
    await page.fill("#del-confirm", "DELETE");
    await page.getByRole("dialog").getByRole("button", { name: "Delete account" }).click();
    await page.waitForURL((u) => u.pathname === "/", { timeout: 30_000 });
  });

  await step("deleted account can no longer log in", async () => {
    const g = await passwordGrant();
    assert(g.status === 400, `expected 400 after deletion, got ${g.status}`);
    return { status: g.status };
  });
} catch (e) {
  exitCode = 1;
  results.error = String(e?.message ?? e);
}
results.finished_at = new Date().toISOString();
results.passed = results.steps.filter((s) => s.ok).length;
results.total = results.steps.length;
results.ok = exitCode === 0 && results.steps.every((s) => s.ok);
if (!results.ok) exitCode = 1;
delete results.facts.image_url; // signed URL; expired/irrelevant after the run
save();
await browser.close();
process.exit(exitCode);
