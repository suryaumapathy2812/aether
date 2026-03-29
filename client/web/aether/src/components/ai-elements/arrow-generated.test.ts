import { describe, expect, it } from "vitest";
import { extractArrowSandboxSource } from "./arrow-generated";

describe("extractArrowSandboxSource", () => {
  it("wraps default component exports into a renderable root", () => {
    const result = extractArrowSandboxSource(`
      import { html, component } from '@arrow-js/core'

      export default component(() => {
        return html\`<div>hello</div>\`
      })
    `);

    expect(result?.source["main.ts"]).toContain(
      "const __AETHER_DEFAULT_COMPONENT = component(",
    );
    expect(result?.source["main.ts"]).toContain(
      "export default html`${__AETHER_DEFAULT_COMPONENT()}`",
    );
  });

  it("rewrites inline mapped html templates into reusable components", () => {
    const result = extractArrowSandboxSource(`
      import { html } from '@arrow-js/core'

      const data = [20, 50, 80, 40, 90]

      export default html\`
        <div>
          \${data.map(h => html\`<div style="height: \${h}%"></div>\`)}
        </div>
      \`
    `);

    const main = result?.source["main.ts"] ?? "";
    expect(main).toContain("import { component, html } from '@arrow-js/core'");
    expect(main).toContain(
      "const __AETHER_MAP_COMPONENT_0 = component(h => html`<div style=\"height: ${h}%\"></div>`)",
    );
    expect(main).toContain("data.map(h => __AETHER_MAP_COMPONENT_0(h))");
    expect(main).not.toContain("data.map(h => html`");
  });

  it("ensures html is imported when wrapping a default component export", () => {
    const result = extractArrowSandboxSource(`
      import { component } from '@arrow-js/core'

      export default component(() => {
        return html\`<div>hello</div>\`
      })
    `);

    const main = result?.source["main.ts"] ?? "";
    expect(main).toContain("import { component, html } from '@arrow-js/core'");
    expect(main).toContain(
      "const __AETHER_DEFAULT_COMPONENT = component(",
    );
    expect(main).toContain(
      "export default html`${__AETHER_DEFAULT_COMPONENT()}`",
    );
  });

  it("rewrites destructured map callbacks into reusable components", () => {
    const result = extractArrowSandboxSource(`
      import { html } from '@arrow-js/core'

      const items = [{ name: "a" }, { name: "b" }]

      export default html\`
        <ul>
          \${items.map(({ name }) => html\`<li>\${name}</li>\`)}
        </ul>
      \`
    `);

    const main = result?.source["main.ts"] ?? "";
    expect(main).toContain("import { component, html } from '@arrow-js/core'");
    expect(main).toContain(
      'const __AETHER_MAP_COMPONENT_0 = component(({ name }) => html`<li>${name}</li>`)',
    );
    expect(main).toContain("items.map(__item => __AETHER_MAP_COMPONENT_0(__item))");
    expect(main).not.toContain("items.map(({ name }) => html`");
  });

  it("normalizes React attributes in structured object source values", () => {
    const result = extractArrowSandboxSource({
      sandbox: {
        source: {
          "main.ts":
            'import { html } from "@arrow-js/core"\nexport default html`<div className="box" tabIndex={0}></div>`',
        },
      },
    });

    const main = result?.source["main.ts"] ?? "";
    expect(main).toContain('class="box"');
    expect(main).toContain("tabindex=");
    expect(main).not.toContain("className");
    expect(main).not.toContain("tabIndex");
  });

  it("normalizes Arrow transforms for structured sandbox sources", () => {
    const result = extractArrowSandboxSource(`
      import { html } from '@arrow-js/core'

      const props = { label: 'ok' }

      export default html\`
        <div>\${props.label}</div>
      \`
    `);

    const main = result?.source["main.ts"] ?? "";
    expect(main).toContain("import { html } from '@arrow-js/core'");
    expect(main).not.toContain("import { html, props } from '@arrow-js/core'");
  });

  it("applies map rewriting to structured sandbox sources", () => {
    const result = extractArrowSandboxSource({
      sandbox: {
        source: {
          "main.ts":
            'import { html } from "@arrow-js/core"\nconst data = [20, 50]\nexport default html`<div>${data.map(h => html`<div className="bar" style="height: ${h}%"></div>`)}<\/div>`',
        },
      },
    });

    const main = result?.source["main.ts"] ?? "";
    expect(main).toContain("import { component, html } from '@arrow-js/core'");
    expect(main).toContain("__AETHER_MAP_COMPONENT_0");
    expect(main).toContain('class="bar"');
    expect(main).toContain("data.map(h => __AETHER_MAP_COMPONENT_0(h))");
    expect(main).not.toContain("data.map(h => html`");
  });

  it("does not invent imports for local identifiers", () => {
    const result = extractArrowSandboxSource(`
      import { html } from '@arrow-js/core'

      const props = { count: 0 }

      export default html\`
        <div>\${props.count}</div>
      \`
    `);

    const main = result?.source["main.ts"] ?? "";
    expect(main).toContain("import { html } from '@arrow-js/core'");
    expect(main).not.toContain("props } from '@arrow-js/core'");
  });
});
