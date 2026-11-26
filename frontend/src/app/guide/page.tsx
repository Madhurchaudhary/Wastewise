"use client";

import { useMemo, useState } from "react";
import { Search, ShieldAlert } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { PageHeader } from "@/components/page-header";
import { CategoryIcon } from "@/components/category-icon";
import { useI18n } from "@/lib/i18n/provider";
import { binLabel, knowledge } from "@/lib/knowledge";
import { ITEM_TO_CATEGORY } from "@/lib/taxonomy";
import { CATEGORIES, type BilingualList } from "@/lib/types";

export default function GuidePage() {
  const { t, pick } = useI18n();
  const [q, setQ] = useState("");
  const L = (v: BilingualList) => (pick(v) as string[]) ?? [];

  const visible = useMemo(() => {
    const needle = q.trim().toLowerCase();
    if (!needle) return [...CATEGORIES];
    return CATEGORIES.filter((c) => {
      const k = knowledge.categories[c];
      const hay = [k.name.en, k.name.hi, ...k.examples.en, ...k.examples.hi, t.categories[c]].join(" ").toLowerCase();
      const items = Object.entries(knowledge.items).filter(([id]) => ITEM_TO_CATEGORY[id] === c).map(([, v]) => `${v.en} ${v.hi}`).join(" ").toLowerCase();
      return hay.includes(needle) || items.includes(needle);
    });
  }, [q, t]);

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <PageHeader title={t.guide.title} sub={t.guide.sub} />
      <div className="relative mb-6 max-w-md">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" aria-hidden />
        <Input className="pl-9" placeholder={t.guide.searchPlaceholder} value={q} onChange={(e) => setQ(e.target.value)} aria-label={t.guide.searchPlaceholder} />
      </div>

      <nav className="mb-8 flex flex-wrap gap-2" aria-label={t.guide.title}>
        {visible.map((c) => (
          <a key={c} href={`#${c}`} className="inline-flex items-center gap-2 rounded-full border bg-card px-3 py-1.5 text-sm hover:bg-accent">
            <CategoryIcon category={c} className="h-4 w-4" />{t.categories[c]}
          </a>
        ))}
      </nav>

      {visible.length === 0 && <p className="text-muted-foreground">{t.guide.noMatch}</p>}

      <div className="space-y-6">
        {visible.map((c) => {
          const k = knowledge.categories[c];
          return (
            <Card key={c} id={c} className="scroll-mt-24">
              <CardHeader className="flex-row items-start gap-4">
                <CategoryIcon category={c} boxed className="h-12 w-12 shrink-0" />
                <div>
                  <CardTitle className="text-xl">{pick(k.name)}</CardTitle>
                  <p className="mt-1 text-sm text-muted-foreground">{pick(k.description)}</p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <Badge variant="outline">{pick(binLabel(k.bin))}</Badge>
                    <Badge variant="lime">{pick(knowledge.recyclable_labels[k.recyclable])}</Badge>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                {([
                  [t.guide.examples, k.examples], [t.result.segregation, k.segregation], [t.result.disposal, k.disposal],
                  [t.result.recycling, k.recycling], [t.result.safety, k.safety], [t.result.mistakes, k.common_mistakes],
                  [t.result.environment, k.environment],
                ] as const).map(([title, list]) => (
                  <section key={title}>
                    <h3 className="mb-2 text-sm font-semibold">{title}</h3>
                    <ul className="list-disc space-y-1 pl-4 text-sm text-muted-foreground">{L(list).map((s, i) => <li key={i}>{s}</li>)}</ul>
                  </section>
                ))}
              </CardContent>
              {c === "hazardous" && (
                <CardContent>
                  <Alert variant="danger">
                    <ShieldAlert />
                    <div>
                      <AlertTitle>{t.guide.otherHazardous}</AlertTitle>
                      <AlertDescription>{t.guide.otherHazardousBody}</AlertDescription>
                    </div>
                  </Alert>
                </CardContent>
              )}
            </Card>
          );
        })}
      </div>
      <p className="mt-8 text-sm text-muted-foreground">{t.guide.disclaimer}</p>
    </div>
  );
}
