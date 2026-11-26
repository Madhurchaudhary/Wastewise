"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Loader2, MailCheck } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { AuthCard, FormError } from "../auth-card";
import { getSupabase } from "@/lib/supabase/client";
import { useI18n } from "@/lib/i18n/provider";
import { passwordStrongEnough } from "@/lib/validation";
import { authErrorMessage } from "@/lib/auth-errors";

export default function SignupPage() {
  const { t } = useI18n();
  const router = useRouter();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [sent, setSent] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (!passwordStrongEnough(password)) return setError(t.auth.weak);
    if (password !== confirm) return setError(t.auth.mismatch);
    setBusy(true);
    const { data, error } = await getSupabase().auth.signUp({
      email: email.trim(),
      password,
      options: { data: { full_name: name.trim() }, emailRedirectTo: `${window.location.origin}/auth/callback?next=/dashboard` },
    });
    setBusy(false);
    if (error) return setError(authErrorMessage(error, t));
    if (data.session) { router.replace("/dashboard"); router.refresh(); return; }
    setSent(true);
  }

  if (sent) {
    return (
      <AuthCard title={t.auth.signupTitle}>
        <div className="flex flex-col items-center gap-3 text-center">
          <MailCheck className="h-10 w-10 text-primary" aria-hidden />
          <p role="status">{t.auth.checkEmail}</p>
          <Button asChild variant="outline"><Link href="/login">{t.auth.backToLogin}</Link></Button>
        </div>
      </AuthCard>
    );
  }

  return (
    <AuthCard title={t.auth.signupTitle} sub={t.auth.signupSub}
      footer={<>{t.auth.haveAccount} <Link className="font-medium text-primary hover:underline" href="/login">{t.auth.login}</Link></>}>
      <form className="space-y-4" onSubmit={onSubmit} noValidate>
        <FormError message={error} />
        <div className="space-y-1.5">
          <Label htmlFor="name">{t.auth.fullName}</Label>
          <Input id="name" autoComplete="name" maxLength={120} value={name} onChange={(e) => setName(e.target.value)} />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="email">{t.auth.email}</Label>
          <Input id="email" type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="password">{t.auth.password}</Label>
          <Input id="password" type="password" autoComplete="new-password" required minLength={8} value={password} onChange={(e) => setPassword(e.target.value)} aria-describedby="pw-help" />
          <p id="pw-help" className="text-xs text-muted-foreground">{t.auth.weak}</p>
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="confirm">{t.auth.confirmPassword}</Label>
          <Input id="confirm" type="password" autoComplete="new-password" required value={confirm} onChange={(e) => setConfirm(e.target.value)} />
        </div>
        <Button type="submit" className="w-full" disabled={busy || !email || !password}>
          {busy && <Loader2 className="animate-spin" />}{t.auth.signup}
        </Button>
      </form>
    </AuthCard>
  );
}
