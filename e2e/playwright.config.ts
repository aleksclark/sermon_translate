import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  retries: 1,
  use: {
    baseURL: process.env.BASE_URL || "http://localhost:4173",
    headless: true,
    screenshot: "only-on-failure",
  },
  reporter: [["list"], ["html", { outputFolder: "results/html-report", open: "never" }]],
  outputDir: "results/test-results",
});
