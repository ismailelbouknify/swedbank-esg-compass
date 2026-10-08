import { useEffect, useState } from "react";

export const PRODUCT_NAME = "Swedbank ESG Compass";
export const PRODUCT_SUBTITLE = "Evidence-based sustainability assessment";

// Only an approved local asset is used: frontend/public/swedbank-logo.svg (served at /swedbank-logo.svg).
// The logo is never downloaded, generated or redrawn; without the file the UI uses text-only branding.
export const LOGO_SRC = "/swedbank-logo.svg";

/** "Google — Swedbank ESG Compass", or just the product name. */
export function useDocumentTitle(page?: string | null) {
  useEffect(() => {
    document.title = page ? `${page} — ${PRODUCT_NAME}` : PRODUCT_NAME;
  }, [page]);
}

/**
 * Whether the local logo file is available. A missing file makes the SPA fallback answer with
 * HTML, which fails to load as an image, so the check only succeeds for a real image.
 */
let logoState: "unknown" | "ok" | "missing" = "unknown";
const waiting: ((ok: boolean) => void)[] = [];

export function useLogoAvailable(): boolean {
  const [ok, setOk] = useState(logoState === "ok");
  useEffect(() => {
    if (logoState !== "unknown") {
      setOk(logoState === "ok");
      return;
    }
    waiting.push(setOk);
    if (waiting.length === 1) {
      const img = new Image();
      const done = (result: boolean) => {
        logoState = result ? "ok" : "missing";
        waiting.splice(0).forEach((cb) => cb(result));
      };
      img.onload = () => done(img.naturalWidth > 0);
      img.onerror = () => done(false);
      img.src = LOGO_SRC;
    }
    return () => {
      const i = waiting.indexOf(setOk);
      if (i >= 0) waiting.splice(i, 1);
    };
  }, []);
  return ok;
}
