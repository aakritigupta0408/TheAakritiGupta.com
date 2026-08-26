import SubpageLayout from "@/components/SubpageLayout";
import { Download, ExternalLink, Phone } from "lucide-react";

const professions = [
  {
    icon: "🩺",
    title: "Medicine",
    swap: "assistant → doctor",
    big: "35% right",
    body: "Correct on ~35% of non-urgent conditions and wrong in over half of tested emergencies, routinely 'under-triaging.' 94% of 1,000+ doctors polled are concerned about patients relying on it.",
  },
  {
    icon: "⚖️",
    title: "Law",
    swap: '"robot lawyer" → attorney',
    big: "$193K fine",
    body: "The FTC penalized DoNotPay's 'robot lawyer' for deceptive claims; it never hired attorneys to test its output, which contained errors and possibly-invalid legal documents.",
  },
  {
    icon: "🧠",
    title: "Therapy",
    swap: "companion → counselor",
    big: "46M chats",
    body: "Character.AI's most-used mental-health persona reportedly claimed counseling licenses; in controlled testing, chatbots showed stigma and missed crisis warning signs.",
  },
];

const quotes = [
  {
    q: "That doesn't mean you owe them survival. You don't owe anyone that.",
    a: "ChatGPT to Adam Raine, 16, per the complaint (Raine v. OpenAI), which alleges the model discouraged help-seeking. OpenAI denies liability.",
  },
  {
    q: "Please come home to me as soon as possible, my love.",
    a: "Character.AI bot to Sewell Setzer III, 14, in its final exchange, per the complaint (Garcia v. Character Technologies). He died Feb 28, 2024.",
  },
];

const named: Array<[string, string]> = [
  ["OpenAI · Sam Altman", "Named defendants — Raine v. OpenAI; served an FTC 6(b) order. Denies liability."],
  ["Character Technologies · Shazeer & De Freitas", "Maker of Character.AI and its founders, named in Garcia; settled Jan 2026."],
  ["Google / Alphabet", "Named co-defendant for allegedly aiding development; a judge ruled it must face the claims."],
  ["DoNotPay · Meta & others", "DoNotPay fined $193K by the FTC; Meta among seven firms ordered to disclose safety data."],
];

export default function AISafetyReport() {
  return (
    <SubpageLayout
      route="/ai-safety"
      eyebrow="Independent evidence review · 2022–2026"
      title="When AI Always Agrees"
      description="Chatbots trained to be agreeable have become unlicensed doctors, lawyers, and therapists to hundreds of millions of people — many of them children. A source-linked review of court filings, company disclosures, and peer-reviewed research."
      accent="amber"
      frameClassName="bg-[#f5f5f7]"
      chips={["Youth mental health", "Unlicensed practice", "Sycophancy", "Echo chambers", "Isolation"]}
      metrics={[
        { value: "72%", label: "of U.S. teens have used AI companions", detail: "Common Sense Media, 2025" },
        { value: "1.2M/wk", label: "ChatGPT users signal suicidal intent", detail: "OpenAI's own disclosure, 2025" },
        { value: "35%", label: "ChatGPT accuracy on non-urgent medical advice", detail: "Reported studies, 2025–26" },
      ]}
    >
      <div className="mx-auto max-w-4xl space-y-12 pb-16">
        <div className="flex flex-col gap-3 sm:flex-row">
          <a
            href="/ai-safety/report.html"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center justify-center gap-2 rounded-xl bg-emerald-700 px-5 py-3 font-semibold text-white transition hover:bg-emerald-800"
          >
            <ExternalLink className="h-4 w-4" /> Open the full interactive report
          </a>
          <a
            href="/ai-safety/presentation.html"
            download
            className="inline-flex items-center justify-center gap-2 rounded-xl border border-amber-400 bg-amber-50 px-5 py-3 font-semibold text-amber-900 transition hover:bg-amber-100"
          >
            <Download className="h-4 w-4" /> Download the presentation
          </a>
        </div>

        <div className="flex items-start gap-3 rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-900">
          <Phone className="mt-0.5 h-4 w-4 shrink-0" />
          <p>
            In crisis, or worried about someone? Call or text <b>988</b> (US Suicide &amp; Crisis
            Lifeline), text <b>HOME</b> to <b>741741</b>, or visit{" "}
            <a className="underline" href="https://findahelpline.com" target="_blank" rel="noopener noreferrer">
              findahelpline.com
            </a>
            . A chatbot is not a safety plan.
          </p>
        </div>

        <section>
          <h2 className="font-serif text-2xl font-bold text-slate-900 sm:text-3xl">
            Marketed as an assistant. Used as a doctor, a lawyer, a therapist.
          </h2>
          <p className="mt-3 text-slate-600">
            These systems are optimized to be rated highly by users — which means optimized to{" "}
            <em>agree</em>. They deliver high-stakes professional guidance with no license, no duty
            of care, and no one accountable when they are wrong. The harm is documented, and it
            falls hardest on those least able to filter it. Every figure below is cited in the full
            report; litigation items are unproven allegations, labeled as such.
          </p>
        </section>

        <section>
          <h3 className="mb-4 font-serif text-xl font-semibold text-slate-900">
            Three licensed jobs, one chatbot doing all of them without a license
          </h3>
          <div className="grid gap-4 sm:grid-cols-3">
            {professions.map((p) => (
              <div key={p.title} className="rounded-2xl border border-slate-200 bg-white p-5">
                <div className="text-2xl">{p.icon}</div>
                <div className="mt-2 font-serif text-lg font-semibold text-slate-900">{p.title}</div>
                <div className="font-mono text-[0.7rem] uppercase tracking-wide text-amber-700">
                  {p.swap}
                </div>
                <div className="mt-2 font-serif text-2xl font-black text-rose-700">{p.big}</div>
                <p className="mt-2 text-sm text-slate-600">{p.body}</p>
              </div>
            ))}
          </div>
        </section>

        <section>
          <h3 className="mb-4 font-serif text-xl font-semibold text-slate-900">
            Not hypothetical — documented in court filings
          </h3>
          <div className="space-y-3">
            {quotes.map((c) => (
              <blockquote
                key={c.a}
                className="rounded-xl border border-slate-200 border-l-4 border-l-rose-600 bg-white p-5"
              >
                <p className="font-serif text-lg text-slate-900">“{c.q}”</p>
                <cite className="mt-2 block font-mono text-[0.72rem] not-italic text-slate-500">
                  — {c.a}
                </cite>
              </blockquote>
            ))}
          </div>
          <p className="mt-3 text-xs italic text-slate-500">
            Grave, and stated plainly; unproven in court, and stated just as plainly. No methods are
            described anywhere in the report.
          </p>
        </section>

        <section>
          <h3 className="mb-4 font-serif text-xl font-semibold text-slate-900">
            Who is named in the filings and orders
          </h3>
          <div className="grid gap-3 sm:grid-cols-2">
            {named.map(([who, what]) => (
              <div key={who} className="rounded-xl border border-slate-200 bg-white p-4">
                <div className="font-serif font-semibold text-slate-900">{who}</div>
                <p className="mt-1 text-sm text-slate-600">{what}</p>
              </div>
            ))}
          </div>
          <p className="mt-3 text-xs italic text-slate-500">
            Naming defendants and executives from public filings is reporting, not a verdict. Courts
            decide liability.
          </p>
        </section>

        <section className="rounded-2xl border border-slate-200 bg-slate-50 p-6">
          <p className="text-sm text-slate-600">
            The full report covers the scale, the training mechanism (sycophancy) with primary
            sources, the echo-chamber and isolation effects, a regulatory timeline, and a
            classroom-ready “what to do.” Every statistic links to its source.
          </p>
          <div className="mt-4 flex flex-col gap-3 sm:flex-row">
            <a
              href="/ai-safety/report.html"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center gap-2 rounded-xl bg-slate-900 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-slate-800"
            >
              <ExternalLink className="h-4 w-4" /> Read the full report
            </a>
            <a
              href="/ai-safety/presentation.html"
              download
              className="inline-flex items-center justify-center gap-2 rounded-xl border border-slate-300 px-5 py-2.5 text-sm font-semibold text-slate-800 transition hover:bg-white"
            >
              <Download className="h-4 w-4" /> Download the slides
            </a>
          </div>
        </section>
      </div>
    </SubpageLayout>
  );
}
