import "./global.css";

import { Toaster } from "@/components/ui/toaster";
import { createRoot } from "react-dom/client";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import Games from "./pages/Games";
import AIPlayground from "./pages/AIPlayground";
import AIDiscoveries from "./pages/AIDiscoveries";
import AITools from "./pages/AITools";
import AICompanies from "./pages/AICompanies";
import AIResearcher from "./pages/talents/AIResearcher";
import SocialEntrepreneur from "./pages/talents/SocialEntrepreneur";
import Marksman from "./pages/talents/Marksman";
import Equestrian from "./pages/talents/Equestrian";
import Aviator from "./pages/talents/Aviator";
import Motorcyclist from "./pages/talents/Motorcyclist";
import Pianist from "./pages/talents/Pianist";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/games" element={<Games />} />
          <Route path="/ai-playground" element={<AIPlayground />} />
          <Route path="/ai-discoveries" element={<AIDiscoveries />} />
          <Route path="/ai-tools" element={<AITools />} />
          <Route path="/ai-companies" element={<AICompanies />} />
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
          {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

createRoot(document.getElementById("root")!).render(<App />);
