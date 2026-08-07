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
      const bundlePaths = Object.keys(bundle);
      const versionPaths = [
        "",
        "index.html",
        ...bundlePaths,
        ...staticFiles,
      ].map((path) => `./${path}`);
      const precachePaths = versionPaths.filter(
        (path) => !path.endsWith(".onnx"),
      );
      const version = createHash("sha256")
        .update(versionPaths.join("\n"))
        .digest("hex")
        .slice(0, 12);
      this.emitFile({
        type: "asset",
        fileName: "sw.js",
        source: `const CACHE_PREFIX = "bloodflow-";
const CACHE = CACHE_PREFIX + "${version}";
const PRECACHE = ${JSON.stringify(precachePaths)};
self.addEventListener("install", (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    await cache.addAll(PRECACHE);
    await self.skipWaiting();
  })());
});
self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter((key) => key.startsWith(CACHE_PREFIX) && key !== CACHE).map((key) => caches.delete(key)));
    await self.clients.claim();
  })());
});
self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET" || new URL(event.request.url).origin !== self.location.origin) return;
  event.respondWith((async () => {
    const cache = await caches.open(CACHE);
    const cached = await cache.match(event.request);
    if (cached != null) return cached;
    const response = await fetch(event.request);
    if (response.ok) {
      try {
        await cache.put(event.request, response.clone());
      } catch {
        // A cache quota failure must not make the online request fail.
      }
    }
    return response;
  })());
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
