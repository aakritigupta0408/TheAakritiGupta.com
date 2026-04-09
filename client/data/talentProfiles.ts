export type TalentRoute =
  | "/talent/ai-researcher"
  | "/talent/social-entrepreneur"
  | "/talent/marksman"
  | "/talent/equestrian"
  | "/talent/aviator"
  | "/talent/motorcyclist"
  | "/talent/pianist";

export interface TalentProfile {
  route: TalentRoute;
  slug: string;
  eyebrow: string;
  role: string;
  headline: string;
  summary: string;
  symbol: string;
  accent: {
    text: string;
    softBg: string;
    softBorder: string;
    glow: string;
    surface: string;
  };
  metrics: { value: string; label: string }[];
  focusAreas: { title: string; description: string }[];
  professionalTranslation: { title: string; description: string }[];
  quote: string;
  quoteCredit: string;
}

export const talentProfiles: TalentProfile[] = [
  {
    route: "/talent/ai-researcher",
    slug: "ai-researcher",
    eyebrow: "Research lens",
    role: "AI Researcher",
    headline: "Research discipline translated into modern product and systems work.",
    summary:
      "This page frames AI research as a working discipline: frontier-model curiosity, production realism, and the ability to turn ambiguous technical questions into usable systems.",
    symbol: "◆",
    accent: {
      text: "text-cyan-600",
      softBg: "bg-cyan-500/12",
      softBorder: "border-cyan-200/80",
      glow: "from-cyan-200/60 via-sky-100/70 to-white",
      surface: "from-cyan-500/12 to-sky-500/10",
    },
    metrics: [
      { value: "ICLR", label: "Recognition context" },
      { value: "50+", label: "Models and systems shipped" },
      { value: "10B+", label: "Users affected by production work" },
    ],
    focusAreas: [
      {
        title: "Frontier-to-product thinking",
        description:
          "Balances research exploration with the constraints that matter in real systems: evaluation, latency, reliability, and adoption.",
      },
      {
        title: "Experimental rigor",
        description:
          "Uses benchmarks, ablations, and comparative analysis to separate interesting demos from methods that actually hold up.",
      },
      {
        title: "Applied deployment discipline",
        description:
          "Connects research depth to large-scale product execution across ranking, personalization, recommendation, and ML-enabled workflows.",
      },
    ],
    professionalTranslation: [
      {
        title: "Hypothesis design",
        description:
          "Research training sharpens the ability to ask precise questions, define success metrics, and design useful experiments before writing code.",
      },
      {
        title: "Signal over noise",
        description:
          "Scientific judgment helps prioritize what is actually novel, material, and worth shipping instead of chasing every model trend.",
      },
      {
        title: "Systems-level clarity",
        description:
          "Strong research habits improve architecture reviews, failure analysis, and the ability to explain technical tradeoffs clearly to teams.",
      },
    ],
    quote:
      "The future belongs to those who can make complex things simple.",
    quoteCredit: "Aakriti Gupta, AI researcher and builder",
  },
  {
    route: "/talent/social-entrepreneur",
    slug: "social-entrepreneur",
    eyebrow: "Impact lens",
    role: "Social Entrepreneur",
    headline: "A product mindset grounded in access, inclusion, and real-world usefulness.",
    summary:
      "This page reframes entrepreneurship as the discipline of choosing where technology should create leverage, who it should serve, and how to make ambitious ideas usable at human scale.",
    symbol: "◇",
    accent: {
      text: "text-teal-600",
      softBg: "bg-teal-500/12",
      softBorder: "border-teal-200/80",
      glow: "from-teal-200/60 via-emerald-100/70 to-white",
      surface: "from-teal-500/12 to-emerald-500/10",
    },
    metrics: [
      { value: "3", label: "Company-building threads" },
      { value: "1M+", label: "People targeted for impact" },
      { value: "15", label: "Social initiative patterns explored" },
    ],
    focusAreas: [
      {
        title: "Access-first product framing",
        description:
          "Looks for ways to make sophisticated technology more reachable and more relevant for underserved or fast-growing markets.",
      },
      {
        title: "Purpose with business realism",
        description:
          "Treats impact and sustainability as design constraints, not afterthoughts, so products can scale without losing the original mission.",
      },
      {
        title: "Systems of trust",
        description:
          "Builds around ethics, local context, affordability, and long-term adoption instead of novelty alone.",
      },
    ],
    professionalTranslation: [
      {
        title: "Market sensing",
        description:
          "Entrepreneurial work sharpens judgment around unmet needs, market timing, and what makes a solution credible to real users.",
      },
      {
        title: "Resourceful execution",
        description:
          "Keeps teams focused on the few moves that change outcomes when time, capital, and attention are limited.",
      },
      {
        title: "Cross-functional persuasion",
        description:
          "Improves the ability to align technical, design, and business stakeholders around a coherent product direction.",
      },
    ],
    quote:
      "Profit should be a byproduct of purpose, not the purpose itself.",
    quoteCredit: "Aakriti Gupta, founder and impact-oriented operator",
  },
  {
    route: "/talent/marksman",
    slug: "marksman",
    eyebrow: "Precision lens",
    role: "Marksman",
    headline: "Precision, calm, and tactical clarity translated into technical execution.",
    summary:
      "This page treats marksmanship less as spectacle and more as a model for precision work: controlled focus, disciplined preparation, and clean execution under pressure.",
    symbol: "◎",
    accent: {
      text: "text-rose-600",
      softBg: "bg-rose-500/12",
      softBorder: "border-rose-200/80",
      glow: "from-rose-200/60 via-red-100/70 to-white",
      surface: "from-rose-500/12 to-red-500/10",
    },
    metrics: [
      { value: "95%", label: "Accuracy mindset" },
      { value: "5+", label: "Years of training" },
      { value: "100m", label: "Focus range metaphor" },
    ],
    focusAreas: [
      {
        title: "Controlled execution",
        description:
          "Favors clean, deliberate action over noisy motion, which maps directly to debugging, architecture, and production decision-making.",
      },
      {
        title: "Pressure handling",
        description:
          "Sustained concentration under stress builds a useful baseline for incident response, high-stakes demos, and hard technical reviews.",
      },
      {
        title: "Tactical reading",
        description:
          "Strong situational awareness helps anticipate failure modes, manage risk, and respond with precision instead of panic.",
      },
    ],
    professionalTranslation: [
      {
        title: "Attention to detail",
        description:
          "Precision training maps naturally to QA, model evaluation, and the ability to catch small details that materially affect outcomes.",
      },
      {
        title: "Decision discipline",
        description:
          "Marksman habits encourage thoughtful action with clear thresholds rather than reactive or impulsive choices.",
      },
      {
        title: "Risk awareness",
        description:
          "A tactical mindset improves system safety reviews, rollback planning, and the discipline to know when not to act.",
      },
    ],
    quote:
      "Precision is not speed slowed down. It is attention made visible.",
    quoteCredit: "Aakriti Gupta, precision-minded builder",
  },
  {
    route: "/talent/equestrian",
    slug: "equestrian",
    eyebrow: "Partnership lens",
    role: "Equestrian",
    headline: "Grace, partnership, and quiet control translated into leadership.",
    summary:
      "This page positions equestrian discipline as a model for leadership through trust: reading nonverbal signals, building partnership, and staying composed in motion.",
    symbol: "◈",
    accent: {
      text: "text-amber-600",
      softBg: "bg-amber-500/12",
      softBorder: "border-amber-200/80",
      glow: "from-amber-200/60 via-orange-100/70 to-white",
      surface: "from-amber-500/12 to-orange-500/10",
    },
    metrics: [
      { value: "8+", label: "Years riding" },
      { value: "15+", label: "Horses trained" },
      { value: "5", label: "Disciplines practiced" },
    ],
    focusAreas: [
      {
        title: "Trust-based coordination",
        description:
          "Strong riding requires subtle communication and mutual trust, which is directly relevant to how teams and partnerships actually work.",
      },
      {
        title: "Balance in motion",
        description:
          "Managing multiple inputs while maintaining control builds composure for dynamic products, ambiguous projects, and changing priorities.",
      },
      {
        title: "Quiet leadership",
        description:
          "Equestrian work rewards calm authority and consistency over force, a better fit for modern cross-functional technical leadership.",
      },
    ],
    professionalTranslation: [
      {
        title: "Stakeholder empathy",
        description:
          "Reading subtle signals improves collaboration, especially where alignment depends on trust more than hierarchy.",
      },
      {
        title: "Grace under pressure",
        description:
          "Handling large, dynamic systems with composure helps in escalations, negotiations, and technical leadership moments.",
      },
      {
        title: "Structured confidence",
        description:
          "Equestrian training reinforces confident direction-setting without overcontrol, which makes teams easier to align and empower.",
      },
    ],
    quote:
      "Leadership is often quiet. The strongest signal is calm control.",
    quoteCredit: "Aakriti Gupta, rider and systems leader",
  },
  {
    route: "/talent/aviator",
    slug: "aviator",
    eyebrow: "Systems lens",
    role: "Aviator",
    headline: "Navigation, checklists, and judgment under pressure translated into architecture thinking.",
    summary:
      "This page uses aviation as a model for system design: mission planning, instrumentation, contingency preparation, and the discipline to make good decisions with incomplete information.",
    symbol: "◉",
    accent: {
      text: "text-blue-600",
      softBg: "bg-blue-500/12",
      softBorder: "border-blue-200/80",
      glow: "from-blue-200/60 via-sky-100/70 to-white",
      surface: "from-blue-500/12 to-sky-500/10",
    },
    metrics: [
      { value: "PPL", label: "Flight credential context" },
      { value: "100+", label: "Flight hours" },
      { value: "5", label: "Aircraft categories" },
    ],
    focusAreas: [
      {
        title: "Preflight discipline",
        description:
          "Checklists, preparation, and operational hygiene map directly to production readiness and engineering quality.",
      },
      {
        title: "Navigation thinking",
        description:
          "Flight work sharpens route planning, instrument reading, and a systems view that fits architecture and reliability work well.",
      },
      {
        title: "Contingency judgment",
        description:
          "Aviation rewards fast but structured thinking when conditions change, which mirrors incident response and technical decision-making.",
      },
    ],
    professionalTranslation: [
      {
        title: "Reliability mindset",
        description:
          "Aviation habits improve fault tolerance, monitoring, and the instinct to prepare for the edge cases that matter.",
      },
      {
        title: "Operational clarity",
        description:
          "Complex systems become easier to manage when procedures, instrumentation, and escalation paths are explicit.",
      },
      {
        title: "High-stakes composure",
        description:
          "Calm under changing conditions helps in launches, outages, and moments where teams need confident technical leadership.",
      },
    ],
    quote:
      "The sky rewards preparation, not improvisation without fundamentals.",
    quoteCredit: "Aakriti Gupta, aviator and systems architect",
  },
  {
    route: "/talent/motorcyclist",
    slug: "motorcyclist",
    eyebrow: "Performance lens",
    role: "Motorcyclist",
    headline: "Mechanical feel, speed, and controlled risk translated into performance engineering.",
    summary:
      "This page reframes riding as a performance discipline: reading machines, responding quickly, and balancing speed with control instead of chasing adrenaline for its own sake.",
    symbol: "◐",
    accent: {
      text: "text-violet-600",
      softBg: "bg-violet-500/12",
      softBorder: "border-violet-200/80",
      glow: "from-violet-200/60 via-fuchsia-100/70 to-white",
      surface: "from-violet-500/12 to-fuchsia-500/10",
    },
    metrics: [
      { value: "10+", label: "Years riding" },
      { value: "200+", label: "Speed envelope metaphor" },
      { value: "6", label: "Bike styles mastered" },
    ],
    focusAreas: [
      {
        title: "Machine sensitivity",
        description:
          "Riding well requires feeling small system changes early, which is a useful instinct in performance tuning and debugging.",
      },
      {
        title: "Fast but controlled response",
        description:
          "At speed, overcorrection is costly. That same discipline matters in operational decision-making and incident handling.",
      },
      {
        title: "Calculated risk",
        description:
          "Motorcycling sharpens the habit of balancing ambition with safety margins, a useful frame for production change management.",
      },
    ],
    professionalTranslation: [
      {
        title: "Performance tuning",
        description:
          "Mechanical awareness maps naturally to infrastructure optimization, latency reduction, and efficient system operation.",
      },
      {
        title: "Safety with ambition",
        description:
          "A strong risk model makes it easier to move quickly without accepting avoidable failure modes.",
      },
      {
        title: "Feedback-loop thinking",
        description:
          "Riding builds constant adjustment based on live signals, which mirrors healthy engineering loops and product iteration.",
      },
    ],
    quote:
      "Speed only matters when control comes with it.",
    quoteCredit: "Aakriti Gupta, rider and performance-oriented engineer",
  },
  {
    route: "/talent/pianist",
    slug: "pianist",
    eyebrow: "Creative lens",
    role: "Pianist",
    headline: "Pattern recognition, timing, and expression translated into product craft.",
    summary:
      "This page positions music as both a technical and expressive discipline: repetition, structure, interpretation, and the ability to make complex systems feel intuitive.",
    symbol: "◑",
    accent: {
      text: "text-emerald-600",
      softBg: "bg-emerald-500/12",
      softBorder: "border-emerald-200/80",
      glow: "from-emerald-200/60 via-green-100/70 to-white",
      surface: "from-emerald-500/12 to-green-500/10",
    },
    metrics: [
      { value: "15+", label: "Years playing" },
      { value: "50+", label: "Pieces learned" },
      { value: "8", label: "Genres explored" },
    ],
    focusAreas: [
      {
        title: "Technical precision",
        description:
          "Piano training develops disciplined repetition, fine control, and reliable execution, which translates well to engineering practice.",
      },
      {
        title: "Pattern recognition",
        description:
          "Musical structure strengthens instincts for rhythm, motif, and composition, all of which connect to code, systems, and data.",
      },
      {
        title: "Expressive judgment",
        description:
          "Performance is not only about correctness. It is about shaping an experience, which maps directly to product polish and interface craft.",
      },
    ],
    professionalTranslation: [
      {
        title: "Design sensitivity",
        description:
          "Musical training improves taste, pacing, and the ability to shape interactions that feel coherent rather than merely functional.",
      },
      {
        title: "Structured creativity",
        description:
          "Working within form while still creating expression is a strong model for product design and technical problem-solving.",
      },
      {
        title: "Practice discipline",
        description:
          "Consistency, iteration, and attention to nuance all improve when craft is treated as something refined over time.",
      },
    ],
    quote:
      "Elegance is structure made expressive.",
    quoteCredit: "Aakriti Gupta, pianist and product-minded technologist",
  },
];

export function getTalentProfile(route: TalentRoute) {
  const profile = talentProfiles.find((item) => item.route === route);

  if (!profile) {
    throw new Error(`Unknown talent route: ${route}`);
  }

  return profile;
}
