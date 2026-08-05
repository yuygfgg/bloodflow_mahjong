import { createHash } from "node:crypto";
import { readdir } from "node:fs/promises";
import { resolve } from "node:path";
import react from "@vitejs/plugin-react";
import type { Plugin } from "vite";
import { defineConfig } from "vitest/config";

async function publicFiles(directory: string, prefix = ""): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true });
  const files: string[] = [];
  for (const entry of entries) {
    const relative = `${prefix}${entry.name}`;
    if (entry.name === "sw.js") continue;
    if (entry.isDirectory()) {
      files.push(
        ...(await publicFiles(resolve(directory, entry.name), `${relative}/`)),
      );
    } else {
      files.push(relative);
    }
  }
  return files;
}

function offlineBundle(): Plugin {
  return {
    name: "bloodflow-offline-bundle",
    apply: "build",
    async generateBundle(_options, bundle) {
      const staticFiles = await publicFiles(
        resolve(import.meta.dirname, "public"),
      );
      const paths = [
        "",
        "index.html",
        ...Object.keys(bundle),
        ...staticFiles,
      ].map((path) => `./${path}`);
      const version = createHash("sha256")
        .update(paths.join("\n"))
        .digest("hex")
        .slice(0, 12);
      this.emitFile({
        type: "asset",
        fileName: "sw.js",
        source: `const CACHE = "bloodflow-${version}";
const PRECACHE = ${JSON.stringify(paths)};
self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(PRECACHE)));
  self.skipWaiting();
});
self.addEventListener("activate", (event) => {
  event.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((key) => key !== CACHE).map((key) => caches.delete(key)))));
  self.clients.claim();
});
self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET" || new URL(event.request.url).origin !== self.location.origin) return;
  event.respondWith(caches.match(event.request).then((cached) => cached ?? fetch(event.request).then((response) => {
    if (response.ok) caches.open(CACHE).then((cache) => cache.put(event.request, response.clone()));
    return response;
  })));
});
`,
      });
    },
  };
}

export default defineConfig({
  base: "./",
  plugins: [react(), offlineBundle()],
  assetsInclude: ["**/*.onnx", "**/*.wasm"],
  build: {
    target: "es2022",
    assetsInlineLimit: 0,
    chunkSizeWarningLimit: 25_000,
  },
  worker: { format: "es" },
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
  },
});
