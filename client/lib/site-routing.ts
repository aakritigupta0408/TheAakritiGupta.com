export function normalizeAppPath(path: string) {
  if (!path) {
    return "/";
  }

  return path.startsWith("/") ? path : `/${path}`;
}

export function buildHashRoute(path: string) {
  return `#${normalizeAppPath(path)}`;
}

export function buildStaticSiteUrl(
  path: string,
  params?: URLSearchParams | Record<string, string>,
) {
  if (typeof window === "undefined") {
    return "";
  }

  const url = new URL(window.location.origin);
  const search =
    params instanceof URLSearchParams
      ? params.toString()
      : params
        ? new URLSearchParams(params).toString()
        : "";

  url.hash = search
    ? `${normalizeAppPath(path)}?${search}`
    : normalizeAppPath(path);

  return url.toString();
}
