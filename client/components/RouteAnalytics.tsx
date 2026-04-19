import { useEffect } from "react";
import { useLocation } from "react-router-dom";
import { initAnalytics, trackPageView } from "@/lib/analytics";

export default function RouteAnalytics() {
  const location = useLocation();

  useEffect(() => {
    initAnalytics();
  }, []);

  useEffect(() => {
    window.scrollTo(0, 0);
    trackPageView(`${location.pathname}${location.search}`);
  }, [location.pathname, location.search]);

  return null;
}
