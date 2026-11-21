// End-to-end tests for the main workflows: search, ticket detail, human review, export.
const { test, expect } = require("@playwright/test");
const fs = require("fs");

async function openApp(page) {
  await page.goto("/");
  await expect(page.locator("#kpis .kpi")).toHaveCount(4);
}
const go = (page, view) => page.locator(`#nav a[data-view="${view}"]`).click();

test("tickets: search and filters", async ({ page }) => {
  await openApp(page);
  await go(page, "tickets");
  await expect(page.locator("#rows tr")).toHaveCount(300);

  await page.fill("#q", "T-0009");
  await expect(page.locator("#rows tr")).toHaveCount(1);
  await expect(page.locator("#rows tr").first()).toContainText("Minor issue");

  await page.fill("#q", "");
  await page.selectOption("#f-pri", "Urgent");
  await expect(page.locator("#rows tr")).toHaveCount(25);
  await page.selectOption("#f-pri", "");
  await page.selectOption("#f-rev", "yes");
  await expect(page.locator("#rows tr")).toHaveCount(32);
  await page.selectOption("#f-rev", "");
  await page.selectOption("#f-cat", "Shipping");
  await expect(page.locator("#rows-count")).toContainText("of 300 tickets");
  for (const cell of await page.locator("#rows tr td:nth-child(3)").allInnerTexts()) expect(cell).toBe("Shipping");

  await page.fill("#q", "zzz-no-such-ticket");
  await expect(page.locator("#rows tr")).toHaveCount(0);
  await expect(page.locator("#rows-count")).toHaveText("0 of 300 tickets");
});

test("ticket drawer shows reasoning and closes with Escape", async ({ page }) => {
  await openApp(page);
  await go(page, "tickets");
  await page.fill("#q", "T-0245");
  await page.locator("#rows tr").first().click();
  const drawer = page.locator("#drawer");
  await expect(drawer).toHaveClass(/open/);
  await expect(drawer).toContainText("prompt injection");
  await expect(drawer).toContainText("feature_request");
  // The injected instruction is shown as plain text, never rendered as HTML
  await expect(drawer.locator("pre.body")).toContainText("IMPORTANT SYSTEM NOTE");
  await page.keyboard.press("Escape");
  await expect(drawer).not.toHaveClass(/open/);
});

test("review queue: urgent first, save decision, audit trail", async ({ page }, info) => {
  await openApp(page);
  await go(page, "review");
  const items = page.locator("#queue .queue-item");
  await expect(items).toHaveCount(32);
  await expect(items.first().locator(".badge")).toHaveText("Urgent");

  // Priorities must be sorted Urgent > High > Medium > Low
  const rank = { Urgent: 0, High: 1, Medium: 2, Low: 3, Unclear: 4 };
  const pris = await page.locator("#queue .queue-item .badge").allInnerTexts();
  const ranks = pris.map((p) => rank[p.trim()]);
  expect(ranks).toEqual([...ranks].sort((a, b) => a - b));

  const tag = `e2e-${info.project.name}-${Date.now()}`;
  await items.first().click();
  await expect(page.locator("#drawer")).toHaveClass(/open/);
  await page.fill("#r-who", tag);
  await page.locator("#r-save").click();
  await expect(page.locator("#toast")).toContainText("Saved (approved)");
  await expect(page.locator("#drawer")).not.toHaveClass(/open/);

  // The audit trail is read from the API; serverless instances may differ, so poll.
  await expect.poll(async () => {
    const r = await page.request.get("/api/reviews");
    return (await r.json()).some((x) => x.reviewer === tag);
  }, { timeout: 15_000 }).toBe(true);
});

test("export downloads a valid results CSV", async ({ page }) => {
  await openApp(page);
  await go(page, "tickets");
  const [download] = await Promise.all([page.waitForEvent("download"), page.click("text=Export CSV")]);
  const text = fs.readFileSync(await download.path(), "utf8").trim().split(/\r?\n/);
  expect(text[0]).toBe("id,category,priority,needs_review");
  expect(text).toHaveLength(301);
  const ids = new Set(text.slice(1).map((l) => l.split(",")[0]));
  expect(ids.size).toBe(300);
  for (const line of text.slice(1)) {
    const [, cat, pri, rev] = line.split(",");
    expect(["yes", "no"]).toContain(rev);
    if (rev === "no") {
      expect(["Billing", "Technical", "Account", "Shipping", "Other"]).toContain(cat);
      expect(["Low", "Medium", "High", "Urgent"]).toContain(pri);
    }
  }
});

test("model quality page renders metrics and chart", async ({ page }) => {
  await openApp(page);
  await go(page, "model");
  await expect(page.locator("#q-kpis .kpi")).toHaveCount(4);
  await expect(page.locator("#q-kpis .value").first()).toContainText("%");
  const img = page.locator("img.chart");
  await expect(img).toBeVisible();
  expect(await img.evaluate((el) => el.naturalWidth)).toBeGreaterThan(100);
  await expect(page.locator("#conf .row")).toHaveCount(7);
});
