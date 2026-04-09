import { motion } from "framer-motion";
import { ArrowLeft, ExternalLink, Sparkles, Star, Moon, Sun } from "lucide-react";
import { useNavigate } from "react-router-dom";

import Navigation from "@/components/Navigation";

// RENDER_URL is injected at build time via Vite env.
// Falls back to the HF Space while the Render service is being set up.
const DEMO_URL =
  import.meta.env.VITE_VEDIC_ASTRO_URL ?? "https://radha006-vedic-astro-ai.hf.space";

const FEATURES = [
  { icon: Star, label: "Multi-agent pipeline", desc: "Natal · Dasha · Transit · Divisional agents run in parallel" },
  { icon: Moon, label: "BPHS rules engine", desc: "Classical Sanskrit texts encoded as structured rule dictionaries" },
  { icon: Sun, label: "Self-correcting output", desc: "Critic + Reviser agents catch and fix low-quality readings" },
  { icon: Sparkles, label: "Personalised calibration", desc: "Convergence loop aligns model weights to your life history" },
];

export default function VedicAstroDemo() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-black text-white">
      <Navigation />

      {/* Header */}
      <section className="relative z-20 pt-32 pb-10 px-8">
        <div className="max-w-7xl mx-auto">
          <motion.button
            onClick={() => navigate("/ai-playground")}
            whileHover={{ x: -4 }}
            className="flex items-center gap-2 text-sm text-slate-400 hover:text-white mb-10 transition-colors"
          >
            <ArrowLeft size={16} />
            Back to AI Playground
          </motion.button>

          <div className="inline-flex rounded-full border border-violet-300/30 bg-violet-400/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.2em] text-violet-200 mb-5">
            Live interactive demo
          </div>

          <h1 className="text-5xl md:text-6xl font-black mb-6 leading-tight">
            Vedic Astrology
            <br />
            <span className="bg-gradient-to-r from-violet-400 via-amber-300 to-rose-400 bg-clip-text text-transparent">
              AI System
            </span>
          </h1>

          <p className="text-slate-300 text-lg leading-8 max-w-3xl mb-10">
            A production-grade multi-agent system grounded in classical Vedic
            texts (BPHS, Brihat Jataka). Enter a birth chart and receive a
            structured reading — natal strengths, active Dasha timing, current
            transits, and divisional chart overlays — synthesised and
            self-corrected by a critic–reviser loop.
          </p>

          {/* Feature chips */}
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-12">
            {FEATURES.map(({ icon: Icon, label, desc }) => (
              <div
                key={label}
                className="rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur-sm"
              >
                <div className="flex items-center gap-2 mb-2">
                  <Icon size={16} className="text-violet-300" />
                  <span className="text-xs font-bold uppercase tracking-widest text-violet-200">
                    {label}
                  </span>
                </div>
                <p className="text-sm text-slate-400 leading-6">{desc}</p>
              </div>
            ))}
          </div>

          <a
            href={DEMO_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 text-xs text-slate-500 hover:text-slate-300 transition-colors mb-2"
          >
            <ExternalLink size={12} />
            Open in full screen
          </a>
        </div>
      </section>

      {/* Embedded Space */}
      <section className="relative z-20 pb-20 px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="rounded-[2rem] overflow-hidden border border-violet-300/20 shadow-[0_0_80px_rgba(139,92,246,0.12)]"
          >
            <iframe
              src={DEMO_URL}
              title="Vedic Astrology AI — live demo"
              width="100%"
              height="900"
              style={{ border: "none", display: "block", background: "#0a0a0a" }}
              allow="clipboard-write"
              loading="lazy"
            />
          </motion.div>
        </div>
      </section>
    </div>
  );
}
