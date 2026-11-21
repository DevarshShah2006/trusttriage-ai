// End-to-end UI tests for the TrustTriage AI dashboard.
const { test, expect } = require("@playwright/test");

/** Fail the test on any page error, console error or failed request. */
function watchErrors(page) {
  const errors = [];
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("console", (m) => m.type() === "error" && errors.push(`console: ${m.text()}`));
  page.on("response", (r) => r.status() >= 400 && errors.push(`HTTP ${r.status()} ${r.url()}`));
  return errors;
}

async function openApp(page) {
  await page.goto("/");
  await expect(page.locator("#kpis .kpi")).toHaveCount(4);
}

async function go(page, view) {
  const link = page.locator(`#nav a[data-view="${view}"]`);
  if (!(await link.isVisible())) await link.scrollIntoViewIfNeeded();
  await link.click();
  await expect(page.locator(`#v-${view}`)).toBeVisible();
}

test("overview shows correct totals with no errors", async ({ page }, info) => {
  const errors = watchErrors(page);
  await openApp(page);
  const values = await page.locator("#kpis .value").allInnerTexts();
  expect(values).toEqual(["300", "268", "32", "25"]);
  await expect(page.locator("#ov-sub")).toContainText("300 tickets processed");
  await expect(page.locator("#urgent tr")).toHaveCount(25);
  await expect(page.locator("#pri-bars .row")).toHaveCount(5);
  await expect(page.locator("#cat-bars .row")).toHaveCount(6);
  await expect(page.locator("#hourly svg rect").first()).toBeVisible();
  await page.screenshot({ path: `screenshots/${info.project.name}-overview.png`, fullPage: true });
  expect(errors).toEqual([]);
});

test("navigation shows exactly one section at a time", async ({ page }) => {
  await openApp(page);
  for (const view of ["tickets", "review", "try", "incidents", "model", "experiments", "overview"]) {
    await go(page, view);
    await expect(page.locator("main > section:visible")).toHaveCount(1);
    await expect(page.locator(`#nav a[data-view="${view}"]`)).toHaveClass(/active/);
  }
});

test("no horizontal page scroll", async ({ page }) => {
  await openApp(page);
  for (const view of ["overview", "tickets", "review", "try", "incidents", "model", "experiments"]) {
    await go(page, view);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflow, `horizontal overflow on ${view}`).toBeLessThanOrEqual(1);
  }
});

test("try-a-ticket examples give the expected decisions", async ({ page }, info) => {
  await openApp(page);
  await go(page, "try");
  const cases = [
    { ex: 0, cat: "Other", pri: "Low", review: "Yes", reason: "prompt injection" },
    { ex: 1, cat: "Other", pri: "Low", review: "No", note: "angry tone" },
    { ex: 2, cat: "Technical", pri: "High", review: "Yes", reason: "several requests" },
    { ex: 3, cat: "Billing", pri: "High", review: "No", note: "non-English" },
  ];
  for (const c of cases) {
    await page.locator(`[data-ex="${c.ex}"]`).click();
    const vals = page.locator("#t-out .result .value");
    await expect(vals.nth(0)).toHaveText(c.cat);
    await expect(vals.nth(1)).toHaveText(c.pri);
    await expect(vals.nth(2)).toHaveText(c.review);
    if (c.reason) await expect(page.locator("#t-out .reason").first()).toContainText(c.reason);
    if (c.note) await expect(page.locator("#t-out")).toContainText(c.note);
  }
  await page.screenshot({ path: `screenshots/${info.project.name}-try.png`, fullPage: true });
});
