import { useEffect, useMemo } from "react";
import { useLocation } from "react-router-dom";
import { trackEvent } from "@/lib/analytics";

type VariantDefinition = {
  key: string;
  weight: number;
};

type ExperimentDefinition = {
  key: string;
  variants: VariantDefinition[];
  defaultVariant: string;
};

const VISITOR_KEY = "tag:visitor-id:v1";
const ASSIGNMENTS_KEY = "tag:experiment-assignments:v1";
const EXPOSED_KEY = "tag:experiment-exposures:v1";

const experimentCatalog = {
  "resume-builder-layout": {
    key: "resume-builder-layout",
    defaultVariant: "compact",
    variants: [
      { key: "compact", weight: 1 },
      { key: "guided", weight: 1 },
    ],
  },
} as const satisfies Record<string, ExperimentDefinition>;

type ExperimentKey = keyof typeof experimentCatalog;

function canUseDom() {
  return typeof window !== "undefined";
}

function readStorageRecord(key: string) {
  if (!canUseDom()) {
    return {};
  }

  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as Record<string, string>) : {};
  } catch {
    return {};
  }
}

function writeStorageRecord(key: string, value: Record<string, string>) {
  if (!canUseDom()) {
    return;
  }

  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // storage unavailable (private browsing, quota exceeded, test env)
  }
}

function getVisitorId() {
  if (!canUseDom()) {
    return "static-render";
  }

  try {
    const existing = window.localStorage.getItem(VISITOR_KEY);
    if (existing) {
      return existing;
    }

    const next = window.crypto?.randomUUID?.() || `visitor-${Date.now()}`;
    window.localStorage.setItem(VISITOR_KEY, next);
    return next;
  } catch {
    return `visitor-${Date.now()}`;
  }
}

function hashToRatio(input: string) {
  let hash = 2166136261;

  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }

  return (hash >>> 0) / 4294967295;
}

function chooseVariant(definition: ExperimentDefinition, seed: string) {
  const totalWeight = definition.variants.reduce(
    (sum, variant) => sum + variant.weight,
    0,
  );
  const bucket = hashToRatio(seed) * totalWeight;
  let cursor = 0;

  for (const variant of definition.variants) {
    cursor += variant.weight;
    if (bucket <= cursor) {
      return variant.key;
    }
  }

  return definition.defaultVariant;
}

function getOverrideVariant(
  definition: ExperimentDefinition,
  search: string,
) {
  const searchParams = new URLSearchParams(search);
  const override = searchParams.get(`exp-${definition.key}`);

  if (!override) {
    return null;
  }

  return definition.variants.some((variant) => variant.key === override)
    ? override
    : null;
}

function resolveVariant(definition: ExperimentDefinition, search: string) {
  const override = getOverrideVariant(definition, search);

  if (override) {
    return {
      variant: override,
      source: "override" as const,
    };
  }

  if (!canUseDom()) {
    return {
      variant: definition.defaultVariant,
      source: "default" as const,
    };
  }

  const assignments = readStorageRecord(ASSIGNMENTS_KEY);
  const assigned = assignments[definition.key];

  if (assigned) {
    return {
      variant: assigned,
      source: "stored" as const,
    };
  }

  const variant = chooseVariant(definition, `${getVisitorId()}:${definition.key}`);
  assignments[definition.key] = variant;
  writeStorageRecord(ASSIGNMENTS_KEY, assignments);

  return {
    variant,
    source: "assigned" as const,
  };
}

function markExposure(experimentKey: string, variant: string, source: string) {
  if (!canUseDom()) {
    return;
  }

  const exposures = readStorageRecord(EXPOSED_KEY);
  const exposureKey = `${experimentKey}:${variant}`;

  if (exposures[exposureKey]) {
    return;
  }

  exposures[exposureKey] = new Date().toISOString();
  writeStorageRecord(EXPOSED_KEY, exposures);
  trackEvent("experiment_exposure", {
    experiment_key: experimentKey,
    variant_key: variant,
    assignment_source: source,
  });
}

export function useExperiment(key: ExperimentKey) {
  const definition = experimentCatalog[key];
  const location = useLocation();
  const assignment = useMemo(
    () => resolveVariant(definition, location.search),
    [definition, location.search],
  );

  useEffect(() => {
    markExposure(definition.key, assignment.variant, assignment.source);
  }, [assignment.source, assignment.variant, definition.key]);

  return {
    variant: assignment.variant,
    isOverride: assignment.source === "override",
    trackMetric(metricName: string, params?: Record<string, unknown>) {
      trackEvent("experiment_metric", {
        experiment_key: definition.key,
        variant_key: assignment.variant,
        metric_name: metricName,
        ...params,
      });
    },
  };
}
