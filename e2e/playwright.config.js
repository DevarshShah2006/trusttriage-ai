// @ts-check
const { defineConfig, devices } = require("@playwright/test");

module.exports = defineConfig({
  testDir: "./tests",
  timeout: 45_000,
  retries: 1, // serverless cold starts can be slow on the first request
  reporter: [["list"], ["html", { open: "never", outputFolder: "report" }]],
  use: {
    baseURL: process.env.BASE_URL || "https://support-ticket-triage-eight.vercel.app",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  projects: [
    { name: "desktop-light", use: { ...devices["Desktop Chrome"], colorScheme: "light" } },
    { name: "desktop-dark", use: { ...devices["Desktop Chrome"], colorScheme: "dark" }, testMatch: /ui\.spec/ },
    { name: "mobile", use: { ...devices["Pixel 7"] }, testMatch: /ui\.spec/ },
  ],
});
