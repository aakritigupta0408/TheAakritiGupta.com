import "./global.css";

import { lazy, Suspense } from "react";
import { Toaster } from "@/components/ui/toaster";
import { createRoot } from "react-dom/client";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { HashRouter, Routes, Route } from "react-router-dom";
import RouteAnalytics from "@/components/RouteAnalytics";
// The homepage stays eagerly imported so the first paint needs no second
// round-trip; every other route is code-split into its own lazy chunk.
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";

const Games = lazy(() => import("./pages/Games"));
const AIPlayground = lazy(() => import("./pages/AIPlayground"));
const BtcOracleDemo = lazy(() => import("./pages/BtcOracleDemo"));
const VedicAstroDemo = lazy(() => import("./pages/VedicAstroDemo"));
const MovielensRecommenderDemo = lazy(
  () => import("./pages/MovielensRecommenderDemo"),
);
const AIDiscoveries = lazy(() => import("./pages/AIDiscoveries"));
const AITools = lazy(() => import("./pages/AITools"));
const AICompanies = lazy(() => import("./pages/AICompanies"));
const AIProjects = lazy(() => import("./pages/AIProjects"));
const PromptEngineering = lazy(() => import("./pages/PromptEngineering"));
const AIAgentTraining = lazy(() => import("./pages/AIAgentTraining"));
const AIChampions = lazy(() => import("./pages/AIChampions"));
const ResumeBuilder = lazy(() => import("./pages/ResumeBuilder"));
const RecruiterResumeAgent = lazy(() => import("./pages/RecruiterResumeAgent"));
const AIResearcher = lazy(() => import("./pages/talents/AIResearcher"));
const SocialEntrepreneur = lazy(
  () => import("./pages/talents/SocialEntrepreneur"),
);
const Marksman = lazy(() => import("./pages/talents/Marksman"));
const Equestrian = lazy(() => import("./pages/talents/Equestrian"));
const Aviator = lazy(() => import("./pages/talents/Aviator"));
const Motorcyclist = lazy(() => import("./pages/talents/Motorcyclist"));
const Pianist = lazy(() => import("./pages/talents/Pianist"));
const AISafetyReport = lazy(() => import("./pages/AISafetyReport"));

const routeFallback = (
  <div className="flex h-screen items-center justify-center bg-[#f5f5f7]">
    <div className="h-8 w-8 animate-spin rounded-full border-2 border-slate-300 border-t-slate-900" />
  </div>
);

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <HashRouter>
        <RouteAnalytics />
        <Suspense fallback={routeFallback}>
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="/games" element={<Games />} />
            <Route path="/ai-playground" element={<AIPlayground />} />
            <Route
              path="/ai-playground/trade-recommendation-system"
              element={<BtcOracleDemo />}
            />
            <Route
              path="/ai-playground/btc-oracle"
              element={<BtcOracleDemo />}
            />
            <Route
              path="/ai-playground/vedic-astro-ai"
              element={<VedicAstroDemo />}
            />
            <Route
              path="/ai-playground/movielens-recommender"
              element={<MovielensRecommenderDemo />}
            />
            <Route path="/ai-discoveries" element={<AIDiscoveries />} />
            <Route path="/ai-tools" element={<AITools />} />
            <Route path="/ai-companies" element={<AICompanies />} />
            <Route path="/ai-projects" element={<AIProjects />} />
            <Route path="/prompt-engineering" element={<PromptEngineering />} />
            <Route path="/ai-agent-training" element={<AIAgentTraining />} />
            <Route path="/ai-champions" element={<AIChampions />} />
            <Route path="/resume-builder" element={<ResumeBuilder />} />
            <Route
              path="/resume-builder/recruiter"
              element={<RecruiterResumeAgent />}
            />
            <Route
              path="/resume-builder/recruiter/:agentId"
              element={<RecruiterResumeAgent />}
            />
            <Route path="/talent/ai-researcher" element={<AIResearcher />} />
            <Route
              path="/talent/social-entrepreneur"
              element={<SocialEntrepreneur />}
            />
            <Route path="/talent/marksman" element={<Marksman />} />
            <Route path="/talent/equestrian" element={<Equestrian />} />
            <Route path="/talent/aviator" element={<Aviator />} />
            <Route path="/talent/motorcyclist" element={<Motorcyclist />} />
            <Route path="/talent/pianist" element={<Pianist />} />
            <Route path="/ai-safety" element={<AISafetyReport />} />
            {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Suspense>
      </HashRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

createRoot(document.getElementById("root")!).render(<App />);
