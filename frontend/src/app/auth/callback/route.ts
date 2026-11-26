import { NextResponse, type NextRequest } from "next/server";
import { getServerSupabase } from "@/lib/supabase/server";

/**
 * Handles Supabase email-confirmation and password-recovery links (PKCE code exchange).
 *
 * Supabase only redirects here with `?code=` after it has verified the emailed token, so a
 * sign-up email is already confirmed at that point. The exchange itself also needs the PKCE
 * verifier cookie that was set in the browser where the user signed up. When the link is opened
 * somewhere else (another browser, the Gmail app, another device), that cookie is missing.
 * We then tell the user their email is confirmed and ask them to log in, rather than
 * reporting a misleading "invalid link".
 */
export async function GET(request: NextRequest) {
  const { searchParams, origin } = request.nextUrl;
  const code = searchParams.get("code");
  const nextParam = searchParams.get("next") ?? "/dashboard";
  const next = nextParam.startsWith("/") && !nextParam.startsWith("//") ? nextParam : "/dashboard";
  const isRecovery = next.startsWith("/reset-password");
  const login = (params: Record<string, string>) =>
    NextResponse.redirect(`${origin}/login?${new URLSearchParams(params).toString()}`);

  if (code) {
    const supabase = await getServerSupabase();
    const { error } = await supabase.auth.exchangeCodeForSession(code);
    if (!error) return NextResponse.redirect(`${origin}${next}`);
    const reason = error.code ?? "exchange_failed";
    console.error(`auth callback: code exchange failed (${reason}): ${error.message}`);
    return login(isRecovery ? { notice: "recovery_other_browser", reason } : { notice: "confirmed", reason });
  }

  // Supabase reports verification failures as query parameters. `otp_expired` covers links that
  // expired and links that were already used. A link scanner or a second click can use a
  // link before the user does, and in that case the email is usually already confirmed.
  // The login page shows a translated notice for these cases; Supabase's raw English text is never shown.
  const errorCode = searchParams.get("error_code") ?? searchParams.get("error") ?? "missing_code";
  return login({ notice: isRecovery ? "recovery_link_used" : "link_used", reason: errorCode });
}
