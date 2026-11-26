"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { AuthCard, FormError } from "../auth-card";
import { getSupabase } from "@/lib/supabase/client";
import { useI18n } from "@/lib/i18n/provider";
import { passwordStrongEnough } from "@/lib/validation";
import { authErrorMessage } from "@/lib/auth-errors";

export default function ResetPasswordPage() {
  const { t } = useI18n();
  const router = useRouter();
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (!passwordStrongEnough(password)) return setError(t.auth.weak);
    if (password !== confirm) return setError(t.auth.mismatch);
    setBusy(true);
    const { error } = await getSupabase().auth.updateUser({ password });
    setBusy(false);
    if (error) return setError(authErrorMessage(error, t));
    toast.success(t.auth.passwordUpdated);
    router.replace("/dashboard");
  }

  return (
    <AuthCard title={t.auth.newPasswordTitle}>
      <form className="space-y-4" onSubmit={onSubmit}>
        <FormError message={error} />
        <div className="space-y-1.5">
          <Label htmlFor="password">{t.auth.password}</Label>
          <Input id="password" type="password" autoComplete="new-password" required value={password} onChange={(e) => setPassword(e.target.value)} />
        </div>
        <div className="space-y-1.5">
          <Label htmlFor="confirm">{t.auth.confirmPassword}</Label>
          <Input id="confirm" type="password" autoComplete="new-password" required value={confirm} onChange={(e) => setConfirm(e.target.value)} />
        </div>
        <Button type="submit" className="w-full" disabled={busy}>{busy && <Loader2 className="animate-spin" />}{t.auth.updatePassword}</Button>
      </form>
    </AuthCard>
  );
}
