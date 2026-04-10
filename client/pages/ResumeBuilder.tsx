import React, { useMemo, useState } from "react";
import { motion } from "framer-motion";
import SubpageLayout from "@/components/SubpageLayout";
import RecruiterAgentChat from "@/components/resume-agent/RecruiterAgentChat";
import { getPageRefreshContent } from "@/data/siteRefreshContent";
import { buildResumeAgent } from "@/api/resume-agent";
import { extractResumeTextFromFile } from "@/lib/resume-upload";
import { isResumeAgentPersistenceConfigured } from "@/lib/resume-agent-persistence";
import type { ResumeAgentProfile } from "@shared/resume-agent";

const workflowSteps = [
  "Upload a PDF, TXT, or Markdown resume.",
  "Add simple-English project notes and missing context.",
  "Build a grounded recruiter agent with a share link.",
  "Send recruiters the link and let the page redirect them back when they finish.",
];

const intakeStreams = [
  {
    eyebrow: "Source 01",
    title: "Resume upload",
    body: "PDF, TXT, or Markdown becomes the base evidence set for the recruiter agent.",
    accent:
      "border-cyan-300/20 bg-cyan-400/10 text-cyan-100",
  },
  {
    eyebrow: "Source 02",
    title: "Project notes",
    body: "Plain-English notes add the real story behind projects, responsibilities, and outcomes.",
    accent:
      "border-violet-300/20 bg-violet-400/10 text-violet-100",
  },
  {
    eyebrow: "Source 03",
    title: "LinkedIn import",
    body: "Optional member-authorized import when official LinkedIn OAuth is configured. Public scraping is not used.",
    accent:
      "border-amber-300/20 bg-amber-400/10 text-amber-100",
  },
];

const agentRules = [
  "The chatbot answers only from approved candidate material.",
  "It should never invent tools, dates, metrics, employers, or outcomes.",
  "If a recruiter asks for something missing, it must say that detail was not provided.",
  "The recruiter link should stay stable when hosted persistence is available.",
];

const recruiterJourney = [
  {
    title: "Candidate prepares facts",
    body: "Resume text, project notes, and optional LinkedIn data become one approved fact sheet.",
  },
  {
    title: "Agent builds a grounded profile",
    body: "The model compresses those facts into a recruiter-safe summary, sections, and suggested prompts.",
  },
  {
    title: "Recruiter receives a clean link",
    body: "The recruiter opens a dedicated chat page and asks questions about skills, projects, and experience.",
  },
  {
    title: "Recruiter exits back to the main site",
    body: "When the review is done, the recruiter is redirected to theaakritigupta.com.",
  },
];

const defaultFactSections = [
  {
    id: "resume",
    title: "Resume evidence",
    bullets: [
      "Parsed directly from the uploaded file or pasted resume text.",
      "Used as the primary source for experience and skill claims.",
    ],
  },
  {
    id: "projects",
    title: "Project context",
    bullets: [
      "Candidate-written notes explain what was built and why it mattered.",
      "Only outcomes already provided by the candidate can be used in answers.",
    ],
  },
  {
    id: "policy",
    title: "Agent policy",
    bullets: [
      "No fabricated facts.",
      "No silent rewriting of the candidate's history.",
    ],
  },
];

const defaultPreviewQuestions = [
  "What are this candidate's strongest skills?",
  "Can you summarize the most relevant projects?",
  "Which experience should a recruiter ask about first?",
];

export default function ResumeBuilder() {
  const pageRefresh = getPageRefreshContent("/resume-builder");
  const persistenceConfigured = isResumeAgentPersistenceConfigured();
  const linkedInImportAvailable = false;
  const [candidateName, setCandidateName] = useState("");
  const [resumeText, setResumeText] = useState("");
  const [projectNotes, setProjectNotes] = useState("");
  const [uploadedFileName, setUploadedFileName] = useState("");
  const [isParsingFile, setIsParsingFile] = useState(false);
  const [isBuildingAgent, setIsBuildingAgent] = useState(false);
  const [parseError, setParseError] = useState("");
  const [builderMessage, setBuilderMessage] = useState("");
  const [agentProfile, setAgentProfile] = useState<ResumeAgentProfile | null>(null);
  const [shareToken, setShareToken] = useState("");
  const [shareId, setShareId] = useState("");
  const [usedModel, setUsedModel] = useState(false);
  const [copyState, setCopyState] = useState<"idle" | "copied" | "failed">("idle");

  const shareUrl = useMemo(() => {
    if (typeof window === "undefined") {
      return "";
    }

    const url = shareId
      ? new URL(`/resume-builder/recruiter/${shareId}`, window.location.origin)
      : shareToken
        ? new URL("/resume-builder/recruiter", window.location.origin)
        : null;

    if (!url) {
      return "";
    }

    if (!shareId) {
      url.searchParams.set("agent", shareToken);
    }

    return url.toString();
  }, [shareId, shareToken]);

  const recruiterLinkMode = shareId
    ? "Persistent recruiter route"
    : persistenceConfigured
      ? "Portable fallback while save is unavailable"
      : "Portable token link";
  const approvedFactCount = agentProfile
    ? agentProfile.sections.reduce((total, section) => total + section.bullets.length, 0) +
      2
    : 0;
  const factSections = agentProfile?.sections ?? defaultFactSections;
  const previewQuestions =
    agentProfile?.suggestedQuestions?.slice(0, 4) ?? defaultPreviewQuestions;

  const handleCopyShareUrl = async () => {
    if (!shareUrl) {
      return;
    }

    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopyState("copied");
    } catch (error) {
      console.error("Failed to copy recruiter share link:", error);
      setCopyState("failed");
    }

    window.setTimeout(() => setCopyState("idle"), 1800);
  };

  const handleResumeUpload = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    setIsParsingFile(true);
    setParseError("");
    setBuilderMessage("");

    try {
      const extractedText = await extractResumeTextFromFile(file);

      if (!extractedText) {
        throw new Error("No readable text was found in that file.");
      }

      setUploadedFileName(file.name);
      setResumeText(extractedText);
      setBuilderMessage(
        "Resume text extracted. Review it below, then add project notes before building the recruiter link.",
      );
    } catch (error) {
      console.error("Failed to extract resume text:", error);
      setParseError(
        error instanceof Error
          ? error.message
          : "Could not read that file. Try another resume file.",
      );
    } finally {
      setIsParsingFile(false);
      event.target.value = "";
    }
  };

  const handleBuildAgent = async () => {
    if (!resumeText.trim()) {
      setParseError("Upload a resume or paste resume text before building.");
      return;
    }

    setIsBuildingAgent(true);
    setParseError("");
    setBuilderMessage("");

    try {
      const result = await buildResumeAgent({
        candidateName: candidateName.trim() || undefined,
        resumeText,
        projectNotes,
      });

      setAgentProfile(result.profile);
      setShareToken(result.shareToken);
      setShareId(result.shareId || "");
      setUsedModel(result.usedModel);
      setBuilderMessage(
        result.shareId
          ? result.usedModel
            ? "Recruiter agent built, persisted, and exposed as a stable recruiter link."
            : "Recruiter agent built and persisted in grounded fallback mode. The stable link still uses only your uploaded material."
          : result.usedModel
            ? "Recruiter agent built with the model and grounded to the uploaded material."
            : "Recruiter agent built in grounded fallback mode. The chat still uses only your uploaded material.",
      );
    } catch (error) {
      console.error("Failed to build recruiter agent:", error);
      setParseError("The recruiter agent could not be built. Please try again.");
    } finally {
      setIsBuildingAgent(false);
    }
  };

  return (
    <SubpageLayout
      route="/resume-builder"
      eyebrow="Resume Agent Builder"
      title="Turn a resume into a recruiter-safe AI agent"
      description="Upload the resume, add plain-English project context, optionally import approved LinkedIn data when configured, and publish a recruiter link that answers only from candidate-provided facts."
      accent="blue"
      chips={[
        "Grounded only",
        "Stable recruiter link",
        "LinkedIn-ready when configured",
        "Redirect after review",
      ]}
      refreshSummary="The page is now structured as a full recruiter-agent product flow: evidence intake, fact approval, grounded agent build, recruiter handoff, and share-link publishing."
      updatedAtLabel={pageRefresh.updatedAtLabel}
      metrics={[
        {
          value: "3",
          label: "Evidence channels",
          detail: "Resume, project notes, and optional LinkedIn import",
        },
        {
          value: "100%",
          label: "Grounding policy",
          detail: "The chatbot must refuse unknown facts instead of guessing",
        },
        {
          value: shareId ? "Live" : persistenceConfigured ? "Ready" : "Fallback",
          label: "Link state",
          detail: "Persistent route when hosted storage is available",
        },
        {
          value: shareUrl ? approvedFactCount.toString() : "0",
          label: "Approved facts",
          detail: "Structured bullets inside the recruiter-facing profile",
        },
      ]}
    >
      <div className="container mx-auto px-6 py-10 sm:py-12">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="mx-auto max-w-6xl"
        >
          <div className="mb-8 rounded-3xl border border-white/15 bg-white/10 p-6 backdrop-blur-xl md:flex md:items-center md:justify-between">
            <div>
              <p className="text-sm font-bold uppercase tracking-[0.2em] text-cyan-100">
                Resume agent status
              </p>
              <p className="mt-2 text-lg font-semibold text-white">
                Upload once, add project context, then generate a recruiter-safe
                chat link.
              </p>
            </div>
            <p className="mt-3 max-w-xl text-sm leading-relaxed text-slate-300 md:mt-0">
              The recruiter chatbot is grounded only in the material the user
              provides. It does not invent extra skills, projects, or experience.
            </p>
          </div>

          <div className="mb-8 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {intakeStreams.map((stream) => (
              <div
                key={stream.title}
                className={`rounded-[1.75rem] border p-5 backdrop-blur-xl ${stream.accent}`}
              >
                <p className="text-xs font-bold uppercase tracking-[0.18em]">
                  {stream.eyebrow}
                </p>
                <h3 className="mt-3 text-lg font-black text-white">
                  {stream.title}
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-slate-200">
                  {stream.body}
                </p>
                <p className="mt-3 text-xs">
                  {stream.title === "Resume upload"
                    ? uploadedFileName
                      ? `Loaded: ${uploadedFileName}`
                      : "No file loaded yet"
                    : stream.title === "Project notes"
                      ? projectNotes.trim()
                        ? "Context added"
                        : "No project notes yet"
                      : linkedInImportAvailable
                        ? "LinkedIn import is available"
                        : "LinkedIn OAuth is not configured on this deployment"}
                </p>
              </div>
            ))}

            <div className="rounded-[1.75rem] border border-emerald-300/20 bg-emerald-400/10 p-5 backdrop-blur-xl">
              <p className="text-xs font-bold uppercase tracking-[0.18em] text-emerald-100">
                Output
              </p>
              <h3 className="mt-3 text-lg font-black text-white">
                Recruiter link mode
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-200">
                The builder creates a recruiter chat route grounded only in candidate-approved facts.
              </p>
              <p className="mt-3 text-xs text-emerald-100">
                {recruiterLinkMode}
              </p>
            </div>
          </div>

          <div className="mb-8 grid gap-6 lg:grid-cols-[1.15fr,0.85fr]">
            <div className="rounded-[2rem] border border-white/15 bg-white/10 p-8 shadow-2xl backdrop-blur-xl">
              <div className="inline-flex rounded-full border border-slate-200/20 bg-white/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.2em] text-slate-100">
                Product contract
              </div>
              <h2 className="mt-4 text-3xl font-black text-white md:text-4xl">
                One candidate fact sheet. One recruiter link. No invented answers.
              </h2>
              <p className="mt-4 max-w-3xl text-sm leading-7 text-slate-200">
                This page is now centered on a single product: a recruiter-facing
                chatbot trained only on the candidate's own material. The system
                should build a compact fact sheet, publish a shareable route, and
                keep every answer inside the boundaries of those approved facts.
              </p>

              <div className="mt-6 grid gap-3 md:grid-cols-2">
                {agentRules.map((rule) => (
                  <div
                    key={rule}
                    className="rounded-3xl border border-white/10 bg-black/20 p-4 text-sm leading-relaxed text-slate-200"
                  >
                    {rule}
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-[2rem] border border-white/15 bg-slate-950/35 p-8 shadow-2xl backdrop-blur-xl">
              <div className="inline-flex rounded-full border border-cyan-300/20 bg-cyan-400/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.2em] text-cyan-100">
                Recruiter outcome
              </div>
              <h2 className="mt-4 text-2xl font-black text-white md:text-3xl">
                What the recruiter actually gets
              </h2>
              <div className="mt-6 space-y-4">
                <div className="rounded-3xl border border-white/10 bg-white/5 p-4">
                  <p className="text-xs font-bold uppercase tracking-[0.18em] text-slate-300">
                    Chat behavior
                  </p>
                  <p className="mt-2 text-sm leading-relaxed text-slate-200">
                    Grounded Q&A about resume, skills, and projects using only candidate-provided material.
                  </p>
                </div>
                <div className="rounded-3xl border border-white/10 bg-white/5 p-4">
                  <p className="text-xs font-bold uppercase tracking-[0.18em] text-slate-300">
                    Link behavior
                  </p>
                  <p className="mt-2 text-sm leading-relaxed text-slate-200">
                    Stable route when hosted persistence is active, with token fallback if storage is unavailable.
                  </p>
                </div>
                <div className="rounded-3xl border border-white/10 bg-white/5 p-4">
                  <p className="text-xs font-bold uppercase tracking-[0.18em] text-slate-300">
                    Exit behavior
                  </p>
                  <p className="mt-2 text-sm leading-relaxed text-slate-200">
                    A finish action returns the recruiter to `theaakritigupta.com` after the review is done.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="mb-12 grid gap-6 lg:grid-cols-[1.05fr,0.95fr]">
            <div className="rounded-[2rem] border border-white/15 bg-white/10 p-8 backdrop-blur-xl shadow-2xl">
              <div className="mb-5">
                <div className="mb-3 inline-flex rounded-full border border-cyan-300/30 bg-cyan-400/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.2em] text-cyan-100">
                  Step 1
                </div>
                <h2 className="text-2xl font-black text-white md:text-3xl">
                  Import the candidate evidence
                </h2>
              </div>

              <label className="mb-5 block rounded-3xl border border-dashed border-white/20 bg-slate-950/30 p-6 text-center transition-colors hover:border-cyan-300/40">
                <input
                  type="file"
                  accept=".pdf,.txt,.md,text/plain,application/pdf,text/markdown"
                  className="hidden"
                  onChange={handleResumeUpload}
                  disabled={isParsingFile}
                />
                <div className="text-4xl">📄</div>
                <p className="mt-3 text-lg font-semibold text-white">
                  {isParsingFile ? "Reading resume..." : "Upload resume"}
                </p>
                <p className="mt-2 text-sm text-slate-300">
                  Supports PDF, TXT, and Markdown resumes.
                </p>
                {uploadedFileName && (
                  <p className="mt-3 text-sm font-semibold text-cyan-100">
                    Loaded: {uploadedFileName}
                  </p>
                )}
              </label>

              <div className="space-y-4">
                <div>
                  <label className="mb-2 block text-sm font-bold uppercase tracking-[0.18em] text-slate-300">
                    Candidate name
                  </label>
                  <input
                    type="text"
                    value={candidateName}
                    onChange={(event) => setCandidateName(event.target.value)}
                    placeholder="Optional. Example: Priya Mehta"
                    className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-white placeholder:text-slate-400 focus:border-cyan-300/40 focus:outline-none"
                  />
                </div>

                <div>
                  <label className="mb-2 block text-sm font-bold uppercase tracking-[0.18em] text-slate-300">
                    Resume text
                  </label>
                  <textarea
                    value={resumeText}
                    onChange={(event) => setResumeText(event.target.value)}
                    placeholder="Upload a resume file or paste resume text here."
                    rows={14}
                    className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm leading-6 text-white placeholder:text-slate-400 focus:border-cyan-300/40 focus:outline-none"
                  />
                </div>

                <div className="rounded-2xl border border-amber-300/15 bg-amber-400/5 p-4">
                  <p className="text-xs font-bold uppercase tracking-[0.18em] text-amber-100">
                    LinkedIn import
                  </p>
                  <p className="mt-2 text-sm leading-relaxed text-slate-200">
                    Sign-in import is reserved for official LinkedIn API setup. Until then, paste any relevant LinkedIn facts into the resume text or project notes so the agent stays grounded.
                  </p>
                  <button
                    type="button"
                    disabled
                    className="mt-4 rounded-full border border-amber-300/20 bg-amber-400/10 px-4 py-2 text-xs font-bold uppercase tracking-[0.18em] text-amber-100 opacity-70"
                  >
                    LinkedIn OAuth not configured
                  </button>
                </div>
              </div>
            </div>

            <div className="rounded-[2rem] border border-white/15 bg-slate-950/30 p-8 backdrop-blur-xl shadow-2xl">
              <div className="mb-5">
                <div className="mb-3 inline-flex rounded-full border border-violet-300/30 bg-violet-400/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.2em] text-violet-100">
                  Step 2
                </div>
                <h2 className="text-2xl font-black text-white md:text-3xl">
                  Shape the approved fact sheet
                </h2>
              </div>

              <textarea
                value={projectNotes}
                onChange={(event) => setProjectNotes(event.target.value)}
                placeholder="Explain the candidate's projects in plain English. Include what they built, why it mattered, tools used, and outcomes that are already true."
                rows={18}
                className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm leading-6 text-white placeholder:text-slate-400 focus:border-violet-300/40 focus:outline-none"
              />

              <div className="mt-6 rounded-3xl border border-white/10 bg-white/5 p-5">
                <p className="text-xs font-bold uppercase tracking-[0.18em] text-violet-100">
                  Build rules
                </p>
                <div className="mt-3 space-y-2">
                  {workflowSteps.map((step) => (
                    <p key={step} className="text-sm leading-relaxed text-slate-200">
                      - {step}
                    </p>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="mb-10 rounded-[2rem] border border-white/15 bg-white/10 p-6 backdrop-blur-xl shadow-2xl">
            <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
              <div>
                <p className="text-sm font-bold uppercase tracking-[0.2em] text-emerald-100">
                  Step 3
                </p>
                <p className="mt-2 text-lg font-semibold text-white">
                  Publish the recruiter-facing agent and share link.
                </p>
              </div>
              <motion.button
                type="button"
                onClick={handleBuildAgent}
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                disabled={isBuildingAgent || !resumeText.trim()}
                className="rounded-full bg-[#0071e3] px-6 py-3 text-sm font-bold text-white transition-colors hover:bg-[#0077ed] disabled:cursor-not-allowed disabled:bg-slate-500"
              >
                {isBuildingAgent ? "Building agent..." : "Build recruiter agent"}
              </motion.button>
            </div>

            {builderMessage && (
              <p className="mt-4 rounded-2xl border border-emerald-300/20 bg-emerald-400/10 px-4 py-3 text-sm text-emerald-100">
                {builderMessage}
              </p>
            )}

            {parseError && (
              <p className="mt-4 rounded-2xl border border-rose-300/20 bg-rose-400/10 px-4 py-3 text-sm text-rose-100">
                {parseError}
              </p>
            )}

            {shareUrl && agentProfile && (
              <div className="mt-6 grid gap-4 lg:grid-cols-[1fr,auto]">
                <div className="rounded-3xl border border-white/10 bg-black/20 p-5">
                  <p className="text-xs font-bold uppercase tracking-[0.18em] text-cyan-100">
                    Recruiter link
                  </p>
                  <div className="mt-3 inline-flex rounded-full border border-white/10 bg-white/5 px-3 py-1 text-[11px] font-bold uppercase tracking-[0.18em] text-slate-200">
                    {recruiterLinkMode}
                  </div>
                  <p className="mt-3 break-all text-sm leading-relaxed text-slate-200">
                    {shareUrl}
                  </p>
                  <p className="mt-3 text-xs text-slate-400">
                    {shareId
                      ? "Stable recruiter route created on the hosted persistence store."
                      : "Using the portable token link fallback because persistent storage was not available."}{" "}
                    {usedModel
                      ? "Model-backed build completed."
                      : "Built in grounded fallback mode because the live API was not available."}
                  </p>
                </div>

                <div className="flex flex-col gap-3">
                  <motion.button
                    type="button"
                    onClick={handleCopyShareUrl}
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97 }}
                    className="rounded-full border border-white/15 bg-white/10 px-5 py-3 text-center text-sm font-bold text-white transition-colors hover:bg-white/15"
                  >
                    {copyState === "copied"
                      ? "Link copied"
                      : copyState === "failed"
                        ? "Copy failed"
                        : "Copy recruiter link"}
                  </motion.button>
                  <motion.a
                    href={shareUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97 }}
                    className="rounded-full border border-cyan-300/30 bg-cyan-400/10 px-5 py-3 text-center text-sm font-bold text-white transition-colors hover:bg-cyan-400/15"
                  >
                    Open recruiter link
                  </motion.a>
                </div>
              </div>
            )}
          </div>

          <div className="mb-10 grid gap-6 xl:grid-cols-[1.1fr,0.9fr]">
            <div className="rounded-[2rem] border border-white/15 bg-white/10 p-8 shadow-2xl backdrop-blur-xl">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="inline-flex rounded-full border border-cyan-300/20 bg-cyan-400/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.2em] text-cyan-100">
                    Approved fact sheet
                  </div>
                  <h2 className="mt-4 text-2xl font-black text-white md:text-3xl">
                    What the agent is allowed to know
                  </h2>
                </div>
                <div className="rounded-full border border-white/10 bg-white/10 px-4 py-2 text-sm font-bold text-white">
                  {shareUrl ? `${approvedFactCount} facts` : "Waiting for build"}
                </div>
              </div>

              <div className="mt-6 grid gap-4 md:grid-cols-2">
                {factSections.map((section) => (
                  <div
                    key={section.id}
                    className="rounded-3xl border border-white/10 bg-black/20 p-5"
                  >
                    <p className="text-xs font-bold uppercase tracking-[0.18em] text-slate-300">
                      {section.title}
                    </p>
                    <div className="mt-3 space-y-2">
                      {section.bullets.slice(0, 4).map((bullet) => (
                        <p
                          key={bullet}
                          className="text-sm leading-relaxed text-slate-200"
                        >
                          - {bullet}
                        </p>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-[2rem] border border-white/15 bg-slate-950/35 p-8 shadow-2xl backdrop-blur-xl">
              <div className="inline-flex rounded-full border border-emerald-300/20 bg-emerald-400/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.2em] text-emerald-100">
                Recruiter journey
              </div>
              <h2 className="mt-4 text-2xl font-black text-white md:text-3xl">
                How the handoff works
              </h2>

              <div className="mt-6 space-y-4">
                {recruiterJourney.map((step, index) => (
                  <div
                    key={step.title}
                    className="rounded-3xl border border-white/10 bg-white/5 p-5"
                  >
                    <p className="text-xs font-bold uppercase tracking-[0.18em] text-emerald-100">
                      Stage {index + 1}
                    </p>
                    <h3 className="mt-2 text-lg font-black text-white">
                      {step.title}
                    </h3>
                    <p className="mt-2 text-sm leading-relaxed text-slate-200">
                      {step.body}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {agentProfile && (
            <div className="mb-12">
              <div className="mb-5">
                <div className="inline-flex rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.2em] text-slate-100">
                  Recruiter preview
                </div>
                <h2 className="mt-4 text-3xl font-black text-white md:text-4xl">
                  Preview the exact recruiter experience
                </h2>
                <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-200">
                  This preview mirrors the shared recruiter route. It uses the same grounded profile, the same suggested prompts, and the same finish-and-redirect behavior.
                </p>
              </div>
              <RecruiterAgentChat
                profile={agentProfile}
                mode="preview"
                shareUrl={shareUrl}
              />
            </div>
          )}

          <div className="grid gap-6 lg:grid-cols-[0.9fr,1.1fr]">
            <div className="rounded-[2rem] border border-white/15 bg-white/10 p-8 shadow-2xl backdrop-blur-xl">
              <div className="inline-flex rounded-full border border-sky-300/20 bg-sky-400/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.2em] text-sky-100">
                Suggested recruiter prompts
              </div>
              <div className="mt-6 space-y-3">
                {previewQuestions.map((question) => (
                  <div
                    key={question}
                    className="rounded-2xl border border-white/10 bg-black/20 px-4 py-3 text-sm font-medium text-slate-100"
                  >
                    {question}
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-[2rem] border border-white/15 bg-slate-950/35 p-8 shadow-2xl backdrop-blur-xl">
              <div className="inline-flex rounded-full border border-amber-300/20 bg-amber-400/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.2em] text-amber-100">
                Launch checklist
              </div>
              <div className="mt-6 grid gap-4 md:grid-cols-2">
                {workflowSteps.map((step, index) => (
                  <div
                    key={step}
                    className="rounded-3xl border border-white/10 bg-white/5 p-5"
                  >
                    <p className="text-xs font-bold uppercase tracking-[0.18em] text-amber-100">
                      Step {index + 1}
                    </p>
                    <p className="mt-3 text-sm leading-relaxed text-slate-200">
                      {step}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </SubpageLayout>
  );
}
