import React, { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Link2, ShieldCheck, Upload } from "lucide-react";
import SubpageLayout from "@/components/SubpageLayout";
import RecruiterAgentChat from "@/components/resume-agent/RecruiterAgentChat";
import { buildResumeAgent } from "@/api/resume-agent";
import { useExperiment } from "@/lib/experiments";
import { isResumeAgentPersistenceConfigured } from "@/lib/resume-agent-persistence";
import { extractResumeTextFromFile } from "@/lib/resume-upload";
import { buildStaticSiteUrl } from "@/lib/site-routing";
import type { ResumeAgentProfile } from "@shared/resume-agent";

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

export default function ResumeBuilder() {
  const persistenceConfigured = isResumeAgentPersistenceConfigured();
  const linkedInImportAvailable = false;
  const layoutExperiment = useExperiment("resume-builder-layout");
  const isGuidedLayout = layoutExperiment.variant === "guided";
  const [candidateName, setCandidateName] = useState("");
  const [resumeText, setResumeText] = useState("");
  const [projectNotes, setProjectNotes] = useState("");
  const [uploadedFileName, setUploadedFileName] = useState("");
  const [isParsingFile, setIsParsingFile] = useState(false);
  const [isBuildingAgent, setIsBuildingAgent] = useState(false);
  const [parseError, setParseError] = useState("");
  const [builderMessage, setBuilderMessage] = useState("");
  const [agentProfile, setAgentProfile] = useState<ResumeAgentProfile | null>(
    null,
  );
  const [shareToken, setShareToken] = useState("");
  const [shareId, setShareId] = useState("");
  const [usedModel, setUsedModel] = useState(false);
  const [copyState, setCopyState] = useState<"idle" | "copied" | "failed">(
    "idle",
  );

  const shareUrl = useMemo(() => {
    if (shareId) {
      return buildStaticSiteUrl(`/resume-builder/recruiter/${shareId}`);
    }

    if (shareToken) {
      return buildStaticSiteUrl("/resume-builder/recruiter", {
        agent: shareToken,
      });
    }

    return "";
  }, [shareId, shareToken]);

  const recruiterLinkMode = shareId
    ? "Stable route"
    : persistenceConfigured
      ? "Local link"
      : "Portable token";
  const approvedFactCount = agentProfile
    ? agentProfile.sections.reduce(
        (total, section) => total + section.bullets.length,
        0,
      ) + 2
    : 0;
  const factSections = agentProfile?.sections ?? defaultFactSections;
  const heroCards = isGuidedLayout
    ? [
        {
          icon: Upload,
          iconClassName: "text-cyan-100",
          title: "Add evidence",
          detail: "Resume text plus simple-English project notes.",
        },
        {
          icon: ShieldCheck,
          iconClassName: "text-violet-100",
          title: "Approve facts",
          detail: "The bot can only speak from what the candidate provided.",
        },
        {
          icon: Link2,
          iconClassName: "text-emerald-100",
          title: "Share static link",
          detail: "GitHub Pages-safe recruiter URL with redirect on finish.",
        },
      ]
    : [
        {
          icon: Upload,
          iconClassName: "text-cyan-100",
          title: "Upload resume",
          detail: "PDF, TXT, or Markdown.",
        },
        {
          icon: ShieldCheck,
          iconClassName: "text-violet-100",
          title: "Approve facts",
          detail: "Add project context in simple English.",
        },
        {
          icon: Link2,
          iconClassName: "text-emerald-100",
          title: "Share recruiter link",
          detail: "Redirects back after review.",
        },
      ];
  const groundingRules = isGuidedLayout
    ? [
        "Use only the resume text and project notes shared with this link.",
        "Treat missing facts as missing. Never improvise around them.",
        "Publish a static share link that works on GitHub Pages.",
      ]
    : [
        "Use only the resume text and project notes shared with this link.",
        "Do not invent skills, dates, employers, metrics, or outcomes.",
        "If a detail is missing, say it was not provided.",
      ];

  const handleCopyShareUrl = async () => {
    if (!shareUrl) {
      return;
    }

    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopyState("copied");
      layoutExperiment.trackMetric("share_link_copied", {
        recruiter_link_mode: recruiterLinkMode,
      });
    } catch (error) {
      void error;
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
      layoutExperiment.trackMetric("resume_loaded", {
        source: "file_upload",
      });
      setBuilderMessage(
        "Resume loaded. Add project context, then publish the recruiter link.",
      );
    } catch (error) {
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

    layoutExperiment.trackMetric("build_clicked", {
      has_project_notes: Boolean(projectNotes.trim()),
      has_candidate_name: Boolean(candidateName.trim()),
    });
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
      layoutExperiment.trackMetric("build_completed", {
        recruiter_link_mode: result.shareId ? "stable" : "portable",
        used_model: result.usedModel,
      });
      setBuilderMessage(
        result.shareId
          ? "Recruiter link is live and tied to the approved candidate facts."
          : "Recruiter link is ready in portable mode and still grounded to the approved candidate facts.",
      );
    } catch (error) {
      void error;
      setParseError("The recruiter agent could not be built. Please try again.");
    } finally {
      setIsBuildingAgent(false);
    }
  };

  return (
    <SubpageLayout
      route="/resume-builder"
      eyebrow="Resume Agent Builder"
      title="Build a recruiter-ready resume agent"
      description="Upload a resume, add plain-English project context, and publish a recruiter link that answers only from candidate-provided facts."
      accent="blue"
      metrics={[
        {
          value: "3",
          label: "Inputs",
          detail: "Résumé, project notes",
        },
        {
          value: shareUrl ? recruiterLinkMode : "Draft",
          label: "Recruiter link",
          detail: "Shareable recruiter view",
        },
        {
          value: shareUrl ? approvedFactCount.toString() : "0",
          label: "Approved facts",
          detail: "Used by the recruiter chatbot",
        },
      ]}
    >
      <div className="container mx-auto px-4 py-8 sm:px-6 sm:py-10">
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mx-auto max-w-5xl space-y-6"
        >
          <div className="grid gap-4 lg:grid-cols-[1.15fr,0.85fr]">
            <section className="rounded-[2rem] border border-white/15 bg-white/10 p-5 shadow-2xl backdrop-blur-xl sm:p-6">
              <p className="text-xs font-bold uppercase tracking-[0.2em] text-cyan-100">
                {isGuidedLayout
                  ? "Static-safe recruiter handoff"
                  : "What this page does"}
              </p>
              <h2 className="mt-3 text-2xl font-black text-white sm:text-3xl">
                {isGuidedLayout
                  ? "Turn approved candidate evidence into one recruiter-safe link."
                  : "Create one grounded recruiter link from the candidate's own material."}
              </h2>
              <p className="mt-3 max-w-2xl text-sm leading-7 text-slate-200">
                {isGuidedLayout
                  ? "This variant leads with a tighter handoff flow: collect evidence, lock the fact sheet, and publish a recruiter chat that stays inside those facts."
                  : "The builder takes resume evidence and plain-English project notes, turns them into an approved fact sheet, and publishes a recruiter chat that refuses to guess beyond those facts."}
              </p>
              {layoutExperiment.isOverride && (
                <p className="mt-3 text-xs font-bold uppercase tracking-[0.18em] text-cyan-100/90">
                  Variant override active: {layoutExperiment.variant}
                </p>
              )}

              <div className="mt-5 grid gap-3 sm:grid-cols-3">
                {heroCards.map((card) => {
                  const Icon = card.icon;

                  return (
                    <div
                      key={card.title}
                      className="rounded-3xl border border-white/10 bg-black/20 p-4"
                    >
                      <Icon className={`h-5 w-5 ${card.iconClassName}`} />
                      <p className="mt-3 text-sm font-semibold text-white">
                        {card.title}
                      </p>
                      <p className="mt-2 text-sm leading-6 text-slate-300">
                        {card.detail}
                      </p>
                    </div>
                  );
                })}
              </div>
            </section>

            <section className="rounded-[2rem] border border-white/15 bg-slate-950/35 p-5 shadow-2xl backdrop-blur-xl sm:p-6">
              <p className="text-xs font-bold uppercase tracking-[0.2em] text-emerald-100">
                Grounding rules
              </p>
              <div className="mt-4 space-y-3">
                {groundingRules.map((rule) => (
                  <div
                    key={rule}
                    className="rounded-3xl border border-white/10 bg-white/5 p-4 text-sm leading-6 text-slate-200"
                  >
                    {rule}
                  </div>
                ))}
              </div>
            </section>
          </div>

          <div className="grid gap-6 lg:grid-cols-[1.1fr,0.9fr]">
            <section className="rounded-[2rem] border border-white/15 bg-white/10 p-5 shadow-2xl backdrop-blur-xl sm:p-6">
              <div className="mb-5">
                <div className="inline-flex rounded-full border border-cyan-300/30 bg-cyan-400/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.2em] text-cyan-100">
                  Step 1
                </div>
                <h2 className="mt-3 text-2xl font-black text-white sm:text-3xl">
                  Upload the resume
                </h2>
              </div>

              <label className="block rounded-3xl border border-dashed border-white/20 bg-slate-950/30 p-5 text-center transition-colors hover:border-cyan-300/40 sm:p-6">
                <input
                  type="file"
                  accept=".pdf,.txt,.md,text/plain,application/pdf,text/markdown"
                  className="hidden"
                  onChange={handleResumeUpload}
                  disabled={isParsingFile}
                />
                <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full border border-white/10 bg-white/10">
                  <Upload className="h-5 w-5 text-cyan-100" />
                </div>
                <p className="mt-4 text-lg font-semibold text-white">
                  {isParsingFile ? "Reading resume..." : "Choose resume file"}
                </p>
                <p className="mt-2 text-sm text-slate-300">
                  PDF, TXT, and Markdown are supported.
                </p>
                {uploadedFileName && (
                  <p className="mt-3 text-sm font-semibold text-cyan-100">
                    Loaded: {uploadedFileName}
                  </p>
                )}
              </label>

              <div className="mt-5 space-y-4">
                <div>
                  <label className="mb-2 block text-sm font-bold uppercase tracking-[0.18em] text-slate-300">
                    Candidate name
                  </label>
                  <input
                    type="text"
                    value={candidateName}
                    onChange={(event) => setCandidateName(event.target.value)}
                    placeholder="Optional"
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
                    rows={12}
                    className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm leading-6 text-white placeholder:text-slate-400 focus:border-cyan-300/40 focus:outline-none"
                  />
                </div>
              </div>
            </section>

            <section className="rounded-[2rem] border border-white/15 bg-slate-950/35 p-5 shadow-2xl backdrop-blur-xl sm:p-6">
              <div className="mb-5">
                <div className="inline-flex rounded-full border border-violet-300/30 bg-violet-400/10 px-3 py-1 text-xs font-bold uppercase tracking-[0.2em] text-violet-100">
                  Step 2
                </div>
                <h2 className="mt-3 text-2xl font-black text-white sm:text-3xl">
                  Add project context
                </h2>
              </div>

              <textarea
                value={projectNotes}
                onChange={(event) => setProjectNotes(event.target.value)}
                placeholder="Write in simple English. Include what the candidate built, what they owned, tools used, and outcomes that are already true."
                rows={12}
                className="w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm leading-6 text-white placeholder:text-slate-400 focus:border-violet-300/40 focus:outline-none"
              />

              {linkedInImportAvailable && (
                <div className="mt-5 rounded-3xl border border-amber-300/15 bg-amber-400/5 p-4">
                  <p className="text-xs font-bold uppercase tracking-[0.18em] text-amber-100">
                    LinkedIn import
                  </p>
                  <p className="mt-2 text-sm leading-6 text-slate-200">
                    Use the approved LinkedIn profile facts pulled via OAuth.
                  </p>
                </div>
              )}
            </section>
          </div>

          <section className="rounded-[2rem] border border-white/15 bg-white/10 p-5 shadow-2xl backdrop-blur-xl sm:p-6">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <p className="text-xs font-bold uppercase tracking-[0.2em] text-emerald-100">
                  Step 3
                </p>
                <h2 className="mt-2 text-2xl font-black text-white sm:text-3xl">
                  Publish the recruiter link
                </h2>
                <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-200">
                  The shared link uses only the approved fact sheet and sends
                  recruiters back to the main site when they finish.
                </p>
              </div>

              <motion.button
                type="button"
                onClick={handleBuildAgent}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                disabled={isBuildingAgent || !resumeText.trim()}
                className="w-full rounded-full bg-[#0071e3] px-6 py-3 text-sm font-bold text-white transition-colors hover:bg-[#0077ed] disabled:cursor-not-allowed disabled:bg-slate-500 sm:w-auto"
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
              <div className="mt-5 grid gap-4 lg:grid-cols-[1fr,auto]">
                <div className="rounded-3xl border border-white/10 bg-black/20 p-5">
                  <p className="text-xs font-bold uppercase tracking-[0.18em] text-cyan-100">
                    Recruiter link
                  </p>
                  <div className="mt-3 inline-flex rounded-full border border-white/10 bg-white/5 px-3 py-1 text-[11px] font-bold uppercase tracking-[0.18em] text-slate-200">
                    {recruiterLinkMode}
                  </div>
                  <p className="mt-3 break-all text-sm leading-6 text-slate-200">
                    {shareUrl}
                  </p>
                  <p className="mt-3 text-xs leading-5 text-slate-400">
                    {shareId
                      ? "Persistent recruiter route created."
                      : "Portable recruiter token created."}{" "}
                    {usedModel
                      ? "AI-enhanced build completed."
                      : "Build completed."}
                  </p>
                </div>

                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1">
                  <motion.button
                    type="button"
                    onClick={handleCopyShareUrl}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
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
                    onClick={() =>
                      layoutExperiment.trackMetric("share_link_opened", {
                        recruiter_link_mode: recruiterLinkMode,
                      })
                    }
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="rounded-full border border-cyan-300/30 bg-cyan-400/10 px-5 py-3 text-center text-sm font-bold text-white transition-colors hover:bg-cyan-400/15"
                  >
                    Open recruiter link
                  </motion.a>
                </div>
              </div>
            )}
          </section>

          {agentProfile && (
            <div className="grid gap-6 lg:grid-cols-[0.88fr,1.12fr]">
              <section className="rounded-[2rem] border border-white/15 bg-white/10 p-5 shadow-2xl backdrop-blur-xl sm:p-6">
                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <p className="text-xs font-bold uppercase tracking-[0.2em] text-cyan-100">
                      Approved facts
                    </p>
                    <h2 className="mt-2 text-2xl font-black text-white sm:text-3xl">
                      What the agent can use
                    </h2>
                  </div>
                  <div className="rounded-full border border-white/10 bg-white/10 px-4 py-2 text-sm font-bold text-white">
                    {approvedFactCount} facts
                  </div>
                </div>

                <div className="mt-5 space-y-4">
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
                            className="text-sm leading-6 text-slate-200"
                          >
                            - {bullet}
                          </p>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </section>

              <section className="space-y-4">
                <div className="rounded-[2rem] border border-white/15 bg-slate-950/35 p-5 shadow-2xl backdrop-blur-xl sm:p-6">
                  <p className="text-xs font-bold uppercase tracking-[0.2em] text-emerald-100">
                    Recruiter preview
                  </p>
                  <h2 className="mt-2 text-2xl font-black text-white sm:text-3xl">
                    Preview the shared recruiter experience
                  </h2>
                  <p className="mt-3 text-sm leading-6 text-slate-200">
                    Test the same grounded agent that recruiters will see on the
                    published link.
                  </p>
                </div>

                <RecruiterAgentChat
                  profile={agentProfile}
                  mode="preview"
                  shareUrl={shareUrl}
                />
              </section>
            </div>
          )}
        </motion.div>
      </div>
    </SubpageLayout>
  );
}
