"use client";

import { AlertTriangle, CheckCircle2, Globe2, Recycle, ShieldCheck, Trash2, XCircle } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useI18n } from "@/lib/i18n/provider";
import type { BilingualList, Guidance } from "@/lib/types";

function Section({ title, icon: Icon, items, tone }: { title: string; icon: LucideIcon; items: string[]; tone?: string }) {
  if (!items?.length) return null;
  return (
    <section>
      <h4 className="mb-2 flex items-center gap-2 text-sm font-semibold"><Icon className={`h-4 w-4 ${tone ?? "text-primary"}`} aria-hidden />{title}</h4>
      <ul className="space-y-1.5 text-sm leading-relaxed text-muted-foreground">
        {items.map((s, i) => <li key={i} className="flex gap-2"><span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-current opacity-60" aria-hidden />{s}</li>)}
      </ul>
    </section>
  );
}

export function GuidancePanel({ g }: { g: Guidance }) {
  const { t, pick } = useI18n();
  // Guard against a plain string (it would otherwise be iterated character by character).
  const L = (v: BilingualList) => {
    const items = pick(v) as unknown;
    if (typeof items === "string") return items.trim() ? [items] : [];
    return Array.isArray(items) ? items.filter((x): x is string => typeof x === "string" && x.trim() !== "") : [];
  };
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><Trash2 className="h-5 w-5 text-primary" aria-hidden />{pick(g.name)}</CardTitle>
        <p className="text-sm text-muted-foreground">{pick(g.short)}</p>
        <div className="mt-2 grid gap-2 sm:grid-cols-2">
          <div className="rounded-md border p-3">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">{t.result.bin}</div>
            <div className="mt-1 flex items-center gap-2 font-medium">
              <span className="h-3 w-3 rounded-full border" style={{ backgroundColor: g.bin === "green" ? "#2E9E4F" : g.bin === "blue" ? "#2F6FD6" : g.bin === "e-waste" ? "#7B61FF" : "#E5484D" }} aria-hidden />
              {pick(g.bin_label)}
            </div>
          </div>
          <div className="rounded-md border p-3">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">{t.result.recyclable}</div>
            <div className="mt-1 font-medium">{pick(g.recyclable_label)}</div>
          </div>
        </div>
      </CardHeader>
      <CardContent className="grid gap-6 md:grid-cols-2">
        <Section title={t.result.segregation} icon={CheckCircle2} items={L(g.segregation)} />
        <Section title={t.result.disposal} icon={Trash2} items={L(g.disposal)} />
        <Section title={t.result.recycling} icon={Recycle} items={L(g.recycling)} />
        <Section title={t.result.safety} icon={ShieldCheck} items={L(g.safety)} tone="text-warning" />
        <Section title={t.result.mistakes} icon={XCircle} items={L(g.common_mistakes)} tone="text-destructive" />
        <Section title={t.result.environment} icon={Globe2} items={L(g.environment)} />
      </CardContent>
    </Card>
  );
}

export function UncertainBanner({ reasons }: { reasons: string[] }) {
  const { t } = useI18n();
  return (
    <div role="alert" className="flex gap-3 rounded-lg border border-warning/40 bg-warning/10 p-4">
      <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-warning" aria-hidden />
      <div>
        <p className="font-semibold">{t.result.uncertainTitle}</p>
        <p className="mt-1 text-sm text-muted-foreground">{t.result.uncertainBody}</p>
        {reasons.length > 0 && (
          <p className="mt-2 flex flex-wrap gap-2 text-xs">
            {reasons.map((r) => <span key={r} className="rounded-full bg-warning/15 px-2 py-0.5 text-warning">{t.result.reasons[r as keyof typeof t.result.reasons] ?? r}</span>)}
          </p>
        )}
      </div>
    </div>
  );
}
