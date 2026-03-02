import { test, expect } from "@playwright/test";

test.describe("Session creation modal", () => {
  test("opens modal from + button and shows form", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("Sessions", { exact: true })).toBeVisible();

    const addButton = page.locator('button[aria-label="New session"]');
    await expect(addButton).toBeVisible();
    await addButton.click();

    await expect(page.getByText("New Session")).toBeVisible();
    await expect(page.getByText("Pipeline", { exact: true })).toBeVisible();
    await expect(page.getByText("Audio Input", { exact: true })).toBeVisible();
    await expect(page.getByText("Audio Output", { exact: true })).toBeVisible();
    await expect(page.locator('button:has-text("Start Session")')).toBeVisible();
    await expect(page.locator('button:has-text("Cancel")')).toBeVisible();
  });

  test("cancel closes modal", async ({ page }) => {
    await page.goto("/");
    await page.locator('button[aria-label="New session"]').click();
    await expect(page.getByText("New Session")).toBeVisible();
    await page.locator('button:has-text("Cancel")').click();
    await expect(page.getByText("New Session")).not.toBeVisible();
  });
});
