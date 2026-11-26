"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { AuthCard, FormError } from "../auth-card";
import { getSupabase } from "@/lib/supabase/client";
import { useI18n } from "@/lib/i18n/provider";
import { authErrorMessage } from "@/lib/auth-errors";

function safeNext(next: string | null): string {
  return next && next.startsWith("/") && !next.startsWith("//") ? next : "/dashboard";
}

function LoginForm() {
  const { t } = useI18n();
  const router = useRouter();
  const params = useSearchParams();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(params.get("error"));
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const { error } = await getSupabase().auth.signInWithPassword({ email: email.trim(), password });
    setBusy(false);
    if (error) { setError(authErrorMessage(error, t)); return; }
    router.replace(safeNext(params.get("next")));
    router.refresh();
  }

  return (
    <AuthCard title={t.auth.loginTitle} sub={t.auth.loginSub}
      footer={<>{t.auth.noAccount} <Link className="font-medium text-primary hover:underline" href="/signup">{t.auth.signup}</Link></>}>
      <form className="space-y-4" onSubmit={onSubmit} noValidate>
        <FormError message={error} />
        <div className="space-y-1.5">
          <Label htmlFor="email">{t.auth.email}</Label>
          <Input id="email" type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
        </div>
        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <Label htmlFor="password">{t.auth.password}</Label>
            <Link href="/forgot-password" className="text-xs text-primary hover:underline">{t.auth.forgot}</Link>
          </div>
          <Input id="password" type="password" autoComplete="current-password" required value={password} onChange={(e) => setPassword(e.target.value)} />
        </div>
        <Button type="submit" className="w-full" disabled={busy || !email || !password}>
          {busy && <Loader2 className="animate-spin" />}{t.auth.login}
        </Button>
      </form>
    </AuthCard>
  );
}

export default function LoginPage() {
  return <Suspense><LoginForm /></Suspense>;
}
