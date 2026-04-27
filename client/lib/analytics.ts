const LOCAL_EVENT_KEY = "tag:analytics-events:v1";
const GA_MEASUREMENT_ID = import.meta.env.VITE_GA_MEASUREMENT_ID?.trim() || "";

declare global {
  interface Window {
    dataLayer?: unknown[];
    gtag?: (...args: unknown[]) => void;
  }
}

interface AnalyticsEventPayload {
  name: string;
  params?: Record<string, unknown>;
  ts: string;
}

function canUseDom() {
  return typeof window !== "undefined" && typeof document !== "undefined";
}

function readLocalEvents() {
  if (!canUseDom()) {
    return [];
  }

  try {
    const raw = window.localStorage.getItem(LOCAL_EVENT_KEY);
    return raw ? (JSON.parse(raw) as AnalyticsEventPayload[]) : [];
  } catch {
    return [];
  }
}

function persistLocalEvent(event: AnalyticsEventPayload) {
  if (!canUseDom()) {
    return;
  }

  const events = readLocalEvents();
  events.push(event);

  try {
    window.localStorage.setItem(LOCAL_EVENT_KEY, JSON.stringify(events.slice(-250)));
  } catch {
    // storage unavailable (private browsing, quota exceeded, test env)
  }
}

function injectGaScript() {
  if (!canUseDom() || !GA_MEASUREMENT_ID) {
    return;
  }

  if (document.querySelector(`script[data-ga-id="${GA_MEASUREMENT_ID}"]`)) {
    return;
  }

  const externalScript = document.createElement("script");
  externalScript.async = true;
  externalScript.src = `https://www.googletagmanager.com/gtag/js?id=${GA_MEASUREMENT_ID}`;
  externalScript.dataset.gaId = GA_MEASUREMENT_ID;
  document.head.appendChild(externalScript);

  const inlineScript = document.createElement("script");
  inlineScript.dataset.gaId = `${GA_MEASUREMENT_ID}-config`;
  inlineScript.text = `
    window.dataLayer = window.dataLayer || [];
    function gtag(){window.dataLayer.push(arguments);}
    window.gtag = window.gtag || gtag;
    gtag('js', new Date());
    gtag('config', '${GA_MEASUREMENT_ID}', { send_page_view: false });
  `;
  document.head.appendChild(inlineScript);
}

export function initAnalytics() {
  injectGaScript();
}

export function trackEvent(name: string, params?: Record<string, unknown>) {
  const payload: AnalyticsEventPayload = {
    name,
    params,
    ts: new Date().toISOString(),
  };

  persistLocalEvent(payload);

  if (canUseDom() && window.gtag) {
    window.gtag("event", name, params || {});
  }
}

export function trackPageView(path: string) {
  trackEvent("page_view", {
    page_path: path,
    page_location: canUseDom() ? window.location.href : path,
    page_title: canUseDom() ? document.title : "",
  });
}
