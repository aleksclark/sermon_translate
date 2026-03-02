import { test, expect } from "@playwright/test";

test.describe("Server stats via API", () => {
  test("GET /api/stats returns valid stats", async ({ request }) => {
    const res = await request.get("/api/stats");
    expect(res.ok()).toBe(true);
    const data = await res.json();
    expect(data).toHaveProperty("uptime_seconds");
    expect(data).toHaveProperty("active_sessions");
    expect(data).toHaveProperty("available_pipelines");
    expect(data.available_pipelines).toBeGreaterThanOrEqual(1);
  });

  test("GET /api/pipelines lists echo pipeline", async ({ request }) => {
    const res = await request.get("/api/pipelines");
    expect(res.ok()).toBe(true);
    const data = await res.json();
    expect(data.length).toBeGreaterThanOrEqual(1);
    const echo = data.find((p: { id: string }) => p.id === "echo");
    expect(echo).toBeDefined();
    expect(echo.name).toBeTruthy();
  });
});
