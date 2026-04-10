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
          { value: "0", label: "Agent loaded" },
          { value: "1", label: "Action required" },
          { value: "1", label: "Builder route" },
          { value: "100%", label: "Grounding enforced" },
        ]}
      >
        <div className="container mx-auto px-6 py-10 sm:py-12">
          <div className="mx-auto max-w-3xl rounded-[2rem] border border-white/15 bg-white/10 p-8 text-center backdrop-blur-xl">
            <h2 className="text-3xl font-black text-white">
              The recruiter agent link is not usable
            </h2>
            <p className="mt-4 text-base leading-relaxed text-slate-200">
              Ask the candidate to generate a fresh recruiter link from the
              resume builder. The chat only works when the candidate material is
              persisted or embedded in the link.
            </p>
            <Link
              to="/resume-builder"
              className="mt-6 inline-flex rounded-full border border-cyan-300/30 bg-cyan-400/10 px-5 py-3 text-sm font-bold text-white transition-colors hover:bg-cyan-400/15"
            >
              Open resume builder
            </Link>
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
