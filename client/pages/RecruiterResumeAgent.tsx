import React, { useEffect, useMemo, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { motion } from "framer-motion";
import SubpageLayout from "@/components/SubpageLayout";
import RecruiterAgentChat from "@/components/resume-agent/RecruiterAgentChat";
import { decodeResumeAgentProfile } from "@shared/resume-agent";
import { fetchResumeAgent } from "@/api/resume-agent";

export default function RecruiterResumeAgent() {
  const { agentId } = useParams<{ agentId: string }>();
  const [searchParams] = useSearchParams();
  const tokenProfile = useMemo(() => {
    const token = searchParams.get("agent");
    return token ? decodeResumeAgentProfile(token) : null;
  }, [searchParams]);
  const [storedProfileLoaded, setStoredProfileLoaded] = useState(!agentId);
  const [storedProfile, setStoredProfile] = useState(tokenProfile);

  useEffect(() => {
    let isMounted = true;

    if (!agentId) {
      setStoredProfileLoaded(true);
      return undefined;
    }

    setStoredProfileLoaded(false);

    fetchResumeAgent(agentId).then((response) => {
      if (!isMounted) {
        return;
      }

      setStoredProfile(response?.profile ?? null);
      setStoredProfileLoaded(true);
    });

    return () => {
      isMounted = false;
    };
  }, [agentId]);

  const profile = storedProfile ?? tokenProfile;

  if (agentId && !storedProfileLoaded) {
    return (
      <SubpageLayout
        route="/resume-builder"
        eyebrow="Recruiter Resume Agent"
        title="Loading recruiter chat"
        description="Fetching the persisted recruiter agent."
        accent="blue"
        metrics={[
          { value: "1", label: "Requested link" },
          { value: "0", label: "Profile loaded" },
          { value: "1", label: "Route lookup" },
          { value: "100%", label: "Grounding enforced" },
        ]}
      >
        <div className="container mx-auto px-6 py-10 sm:py-12">
          <div className="mx-auto max-w-3xl rounded-[2rem] border border-white/15 bg-white/10 p-8 text-center backdrop-blur-xl">
            <h2 className="text-3xl font-black text-white">
              Loading recruiter agent
            </h2>
            <p className="mt-4 text-base leading-relaxed text-slate-200">
              Pulling the persisted candidate material for this recruiter link.
            </p>
          </div>
        </div>
      </SubpageLayout>
    );
  }

  if (!profile) {
    return (
      <SubpageLayout
        route="/resume-builder"
        eyebrow="Recruiter Resume Agent"
        title="Recruiter Link Not Available"
        description="This recruiter chat link is missing or invalid."
        accent="blue"
        metrics={[
          { value: "—", label: "Candidate profile", detail: "Not loaded" },
          { value: "3", label: "Ways to recover", detail: "Options below" },
        ]}
      >
        <div className="container mx-auto px-6 py-10 sm:py-12">
          <div className="mx-auto max-w-3xl space-y-6 rounded-[1.75rem] border border-white/10 bg-white/[0.06] p-7 backdrop-blur-2xl shadow-[0_20px_48px_rgba(8,12,24,0.45),inset_0_1px_0_rgba(255,255,255,0.06)]">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-amber-300">
                Link not recognized
              </p>
              <h2 className="mt-2 text-2xl font-semibold text-white sm:text-3xl">
                This recruiter chat can't load a candidate profile.
              </h2>
              <p className="mt-3 text-sm leading-relaxed text-slate-300">
                The link is missing its payload, the stored session has expired,
                or the URL was copied incompletely. Pick any of the recovery
                paths below.
              </p>
            </div>

            <ol className="space-y-3 text-sm text-slate-200">
              <li className="flex gap-3 rounded-2xl border border-white/10 bg-white/5 p-4">
                <span className="font-mono text-xs font-semibold text-cyan-300">
                  01
                </span>
                <div>
                  <p className="font-semibold text-white">
                    Re-open the candidate's link
                  </p>
                  <p className="mt-1 text-xs leading-relaxed text-slate-400">
                    Double-check the full URL including everything after{" "}
                    <code className="rounded bg-white/10 px-1 py-0.5 text-[11px] text-cyan-200">
                      ?agent=
                    </code>
                    — a truncated paste is the most common cause.
                  </p>
                </div>
              </li>
              <li className="flex gap-3 rounded-2xl border border-white/10 bg-white/5 p-4">
                <span className="font-mono text-xs font-semibold text-cyan-300">
                  02
                </span>
                <div>
                  <p className="font-semibold text-white">
                    Ask the candidate to regenerate the link
                  </p>
                  <p className="mt-1 text-xs leading-relaxed text-slate-400">
                    Links expire when the underlying candidate material is
                    cleared. A fresh link always loads.
                  </p>
                </div>
              </li>
              <li className="flex gap-3 rounded-2xl border border-white/10 bg-white/5 p-4">
                <span className="font-mono text-xs font-semibold text-cyan-300">
                  03
                </span>
                <div>
                  <p className="font-semibold text-white">
                    Build your own from the resume builder
                  </p>
                  <p className="mt-1 text-xs leading-relaxed text-slate-400">
                    If you're the candidate, open the builder, upload the
                    résumé, and generate a recruiter link.
                  </p>
                </div>
              </li>
            </ol>

            <div className="flex flex-wrap items-center gap-3 pt-1">
              <Link
                to="/resume-builder"
                className="inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-cyan-400 via-sky-400 to-fuchsia-400 px-5 py-2.5 text-sm font-semibold text-slate-950 shadow-[0_10px_30px_rgba(56,189,248,0.35)] transition-transform hover:-translate-y-0.5"
              >
                Open resume builder
              </Link>
              <Link
                to="/"
                className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-5 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-white/10"
              >
                Back to home
              </Link>
            </div>
          </div>
        </div>
      </SubpageLayout>
    );
  }

  return (
    <SubpageLayout
      route="/resume-builder"
      eyebrow="Recruiter Resume Agent"
      title={`Recruiter chat for ${profile.candidateName}`}
      description="Ask grounded questions about this candidate's skills, resume, and projects. The chatbot will only answer from the material shared for this link."
      accent="blue"
      chips={["Recruiter-facing", "Grounded only", "No added facts"]}
      metrics={[
        { value: profile.sections.length.toString(), label: "Grounded sections" },
        {
          value: profile.suggestedQuestions.length.toString(),
          label: "Suggested prompts",
        },
        { value: "1", label: "Chat link" },
        { value: "100%", label: "No invention policy" },
      ]}
    >
      <div className="container mx-auto px-6 py-10 sm:py-12">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mx-auto max-w-6xl"
        >
          <RecruiterAgentChat profile={profile} mode="recruiter" />
        </motion.div>
      </div>
    </SubpageLayout>
  );
}
