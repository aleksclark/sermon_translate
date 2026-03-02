import { test, expect } from "@playwright/test";

test.describe("Session CRUD via API", () => {
  test("create, read, update, delete session", async ({ request }) => {
    // Create
    const create = await request.post("/api/sessions", {
      data: { pipeline_id: "echo", label: "e2e-test" },
    });
    expect(create.status()).toBe(201);
    const session = await create.json();
    expect(session.pipeline_id).toBe("echo");
    expect(session.label).toBe("e2e-test");
    expect(session.status).toBe("created");
    const id = session.id;

    // Read
    const get = await request.get(`/api/sessions/${id}`);
    expect(get.ok()).toBe(true);
    const fetched = await get.json();
    expect(fetched.id).toBe(id);

    // List
    const list = await request.get("/api/sessions");
    expect(list.ok()).toBe(true);
    const sessions = await list.json();
    expect(sessions.some((s: { id: string }) => s.id === id)).toBe(true);

    // Update label
    const patch = await request.patch(`/api/sessions/${id}`, {
      data: { label: "renamed" },
    });
    expect(patch.ok()).toBe(true);
    expect((await patch.json()).label).toBe("renamed");

    // Update status to closed
    const close = await request.patch(`/api/sessions/${id}`, {
      data: { status: "closed" },
    });
    expect(close.ok()).toBe(true);
    expect((await close.json()).status).toBe("closed");

    // Delete
    const del = await request.delete(`/api/sessions/${id}`);
    expect(del.status()).toBe(204);

    // Verify gone
    const gone = await request.get(`/api/sessions/${id}`);
    expect(gone.status()).toBe(404);
  });
});
