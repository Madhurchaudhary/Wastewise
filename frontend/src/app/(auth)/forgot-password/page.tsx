"use client";

import Link from "next/link";
import { useState } from "react";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { AuthCard, FormError } from "../auth-card";
import { getSupabase } from "@/lib/supabase/client";
import { useI18n } from "@/lib/i18n/provider";
import { authErrorMessage } from "@/lib/auth-errors";

export default function ForgotPasswordPage() {
  const { t } = useI18n();
  const [email, setEmail] = useState("");
  const [busy, setBusy] = useState(false);
  const [sent, setSent] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const { error } = await getSupabase().auth.resetPasswordForEmail(email.trim(), {
      redirectTo: `${window.location.origin}/auth/callback?next=/reset-password`,
    });
    setBusy(false);
    // Do not reveal whether the address exists; only surface transport/rate-limit errors.
    if (error && error.status !== 400) setError(authErrorMessage(error, t));
    else setSent(true);
  }

  return (
    <AuthCard title={t.auth.resetTitle} sub={t.auth.resetSub}
      footer={<Link className="font-medium text-primary hover:underline" href="/login">{t.auth.backToLogin}</Link>}>
      {sent ? (
        <p role="status" className="rounded-md bg-accent p-3 text-sm">{t.auth.resetSent}</p>
      ) : (
        <form className="space-y-4" onSubmit={onSubmit}>
          <FormError message={error} />
          <div className="space-y-1.5">
            <Label htmlFor="email">{t.auth.email}</Label>
            <Input id="email" type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
          </div>
          <Button type="submit" className="w-full" disabled={busy || !email}>{busy && <Loader2 className="animate-spin" />}{t.auth.sendLink}</Button>
        </form>
      )}
    </AuthCard>
  );
}
