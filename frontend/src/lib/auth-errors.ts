import type { Dictionary } from "@/lib/i18n/dictionary";

/** Minimal shape of a Supabase AuthError (code is set by GoTrue for known failures). */
export interface AuthErrorLike { code?: string; status?: number; message: string }

/** Turn Supabase Auth errors into friendly, translated messages (falls back to the raw message). */
export function authErrorMessage(error: AuthErrorLike, t: Dictionary): string {
  const code = error.code ?? "";
  const msg = error.message.toLowerCase();
  if (code === "over_email_send_rate_limit" || msg.includes("email rate limit")) return t.auth.emailRateLimited;
  if (code === "over_request_rate_limit" || error.status === 429) return t.common.rateLimited;
  if (code === "email_not_confirmed" || msg.includes("email not confirmed")) return t.auth.notConfirmed;
  if (code === "invalid_credentials" || msg.includes("invalid login credentials")) return t.auth.invalidCredentials;
  return error.message;
}
