import React from "react";
import { motion } from "framer-motion";
import SubpageLayout from "@/components/SubpageLayout";

const resumeUrl =
  "https://drive.google.com/file/d/1Mnmk6nP9l_Av0LvpgJQ5Tkjb7BqhY7nb/view?usp=sharing";

const profileLinks = [
  {
    title: "LinkedIn",
    description: "Professional experience, leadership history, and network.",
    href: "https://www.linkedin.com/in/aakritigupta4894/",
    accent: "from-blue-500/20 to-cyan-500/20",
  },
  {
    title: "GitHub",
    description: "Projects, repositories, and public engineering work.",
    href: "https://github.com/aakritigupta0408?tab=achievements",
    accent: "from-slate-500/20 to-zinc-500/20",
  },
];

const promptTemplates = [
  {
    title: "Tailor for a specific role",
    prompt:
      "Rewrite my resume for this job description. Keep the strongest quantified achievements, remove low-signal items, and return a tighter version focused on business impact, technical ownership, and leadership.",
  },
  {
    title: "Strengthen bullets",
    prompt:
      "Turn these resume bullets into sharper achievement statements. Start each bullet with a strong action, add measurable outcomes where possible, and keep every line concise and executive-readable.",
  },
  {
    title: "Create a summary section",
    prompt:
      "Write a 3-line professional summary for a senior AI/ML engineer with product, research, and leadership experience. Emphasize scope, cross-functional influence, and shipped business impact.",
  },
];

export default function ResumeBuilder() {
  return (
    <SubpageLayout
      route="/resume-builder"
      eyebrow="Career toolkit"
      title="Resume assets, profile links, and reusable prompt templates in one place"
      description="This page now follows the shared level-one shell so the practical resume resources feel like part of the same product system instead of a visually separate microsite."
      accent="blue"
      chips={[
        "Current public resume",
        "LinkedIn and GitHub references",
        "Copy-ready AI prompt templates",
      ]}
      metrics={[
        {
          value: "1",
          label: "Public resume link",
        },
        {
          value: profileLinks.length.toString(),
          label: "Profile references",
        },
        {
          value: promptTemplates.length.toString(),
          label: "Prompt templates",
        },
        {
          value: "4",
          label: "Usage steps",
        },
      ]}
    >
      <div className="container mx-auto px-6 py-10 sm:py-12">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="mx-auto max-w-5xl"
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-12">
            <motion.a
              href={resumeUrl}
              target="_blank"
              rel="noopener noreferrer"
              whileHover={{ scale: 1.02, y: -3 }}
              className="rounded-3xl border border-white/15 bg-white/10 p-8 backdrop-blur-xl shadow-2xl"
            >
              <div className="text-4xl mb-4">📄</div>
              <h2 className="text-2xl font-black mb-3">Open Current Resume</h2>
              <p className="text-slate-200 leading-relaxed">
                Direct access to the latest public resume for downloading,
                reviewing, or using as a reference while tailoring applications.
              </p>
              <div className="mt-5 text-cyan-200 font-bold">Open resume →</div>
            </motion.a>

            <div className="rounded-3xl border border-white/15 bg-slate-950/30 p-8 backdrop-blur-xl shadow-2xl">
              <h2 className="text-2xl font-black mb-4">How To Use It</h2>
              <div className="space-y-3 text-slate-200 leading-relaxed">
                <p>1. Open the resume and copy the most relevant experience.</p>
                <p>2. Match it to the target role before rewriting anything.</p>
                <p>3. Use the prompts below to tighten bullets and summaries.</p>
                <p>4. Keep measurable outcomes and remove generic filler.</p>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
            {profileLinks.map((profile, index) => (
              <motion.a
                key={profile.title}
                href={profile.href}
                target="_blank"
                rel="noopener noreferrer"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 + index * 0.08 }}
                whileHover={{ scale: 1.02, y: -3 }}
                className={`rounded-3xl border border-white/15 bg-gradient-to-r ${profile.accent} p-6 backdrop-blur-xl shadow-2xl`}
              >
                <h3 className="text-xl font-black mb-2">{profile.title}</h3>
                <p className="text-slate-200">{profile.description}</p>
                <div className="mt-4 text-cyan-200 font-bold">
                  Open profile →
                </div>
              </motion.a>
            ))}
          </div>

          <div className="rounded-[2rem] border border-white/15 bg-white/10 p-8 backdrop-blur-xl shadow-2xl">
            <h2 className="text-2xl sm:text-3xl font-black mb-6">
              AI Prompt Templates
            </h2>
            <div className="grid grid-cols-1 gap-5">
              {promptTemplates.map((template, index) => (
                <motion.div
                  key={template.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.2 + index * 0.08 }}
                  className="rounded-3xl border border-white/15 bg-slate-950/40 p-6"
                >
                  <h3 className="text-lg font-black text-cyan-100 mb-3">
                    {template.title}
                  </h3>
                  <p className="rounded-2xl border border-white/10 bg-black/30 p-4 text-sm sm:text-base text-slate-200 leading-relaxed">
                    {template.prompt}
                  </p>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </SubpageLayout>
  );
}
