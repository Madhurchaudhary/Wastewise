import { describe, expect, it } from "vitest";
import { authErrorMessage } from "./auth-errors";
import { dictionary } from "./i18n/dictionary";

const t = dictionary.en;

describe("authErrorMessage", () => {
  it("maps the Supabase email rate limit (seen in production) to a friendly message", () => {
    expect(authErrorMessage({ code: "over_email_send_rate_limit", status: 429, message: "email rate limit exceeded" }, t)).toBe(t.auth.emailRateLimited);
    expect(authErrorMessage({ message: "email rate limit exceeded" }, t)).toBe(t.auth.emailRateLimited);
  });
  it("maps unconfirmed and invalid-credential logins", () => {
    expect(authErrorMessage({ code: "email_not_confirmed", status: 400, message: "Email not confirmed" }, t)).toBe(t.auth.notConfirmed);
    expect(authErrorMessage({ code: "invalid_credentials", status: 400, message: "Invalid login credentials" }, t)).toBe(t.auth.invalidCredentials);
  });
  it("maps generic 429s and keeps unknown messages", () => {
    expect(authErrorMessage({ status: 429, message: "Too Many Requests" }, t)).toBe(t.common.rateLimited);
    expect(authErrorMessage({ message: "Signups not allowed" }, t)).toBe("Signups not allowed");
  });
  it("has Hindi strings for every new key", () => {
    for (const k of ["emailRateLimited", "notConfirmed", "invalidCredentials"] as const) {
      expect(dictionary.hi.auth[k].length).toBeGreaterThan(5);
    }
  });
});
