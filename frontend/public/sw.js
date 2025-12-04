// Minimal service worker to make the app installable as a PWA.
// Intentionally lightweight — it installs and immediately claims clients.
self.addEventListener('install', (event) => {
  // Activate fast
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

// Optional fetch handler — here as a pass-through so the SW is functional
self.addEventListener('fetch', (event) => {
  // We don't cache here; this is a lightweight surrogate SW just to enable installability.
});
