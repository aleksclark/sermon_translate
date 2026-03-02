import { test, expect } from "@playwright/test";

test.describe("App shell", () => {
  test("loads and shows title", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("text=Sermon Translate")).toBeVisible();
  });

  test("shows server stats panel", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("Server", { exact: true })).toBeVisible();
    await expect(page.getByText("Uptime")).toBeVisible({ timeout: 10_000 });
  });

  test("shows sessions panel", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("Sessions", { exact: true })).toBeVisible();
    await expect(page.getByText("No sessions yet")).toBeVisible();
  });

  test("has dark mode toggle", async ({ page }) => {
    await page.goto("/");
    const toggle = page.locator('button[aria-label="Toggle color scheme"]');
    await expect(toggle).toBeVisible();
    await toggle.click();
  });
});
