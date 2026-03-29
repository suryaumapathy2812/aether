---
name: arrow-ui
description: Generate sandbox-safe Arrow UI modules for rich answer rendering, interactive cards, forms, lists, dashboards, and compact result explorers. Use this whenever the user asks for Arrow UI, interactive rendering, live widgets, visual result presentation, embedded forms, or any UI response that should render in the Aether chat instead of plain prose.
---

# Arrow UI

Generate UI for the Aether chat sandbox using `@arrow-js/core`.

Use this skill when UI is materially better than prose. Do not emit Arrow just because you can.

## Goal

Produce a single self-contained Arrow module that:

- renders correctly in the sandbox
- uses only supported runtime patterns
- stays visually clear without app-global CSS
- can send narrow host actions back to chat when needed

## Runtime contract

- Import only from `@arrow-js/core` unless the prompt explicitly documents another sandbox bridge.
- Export a renderable root.
- Return only module source when emitting Arrow. No prose, no markdown fences, no explanation before or after the module.
- Use inline `style=""` attributes for styling.
- Use `reactive()` for mutable state.
- Use Arrow event bindings like `@click=${...}`, `@input=${...}`, `@change=${...}`.
- Use reactive expressions like `${() => state.value}` for dynamic content.

## Safe root patterns

Static root:

```ts
import { html } from '@arrow-js/core'

export default html`<section>...</section>`
```

Stateful app root:

```ts
import { html, reactive, component } from '@arrow-js/core'

const App = component(() => {
  const state = reactive({ count: 0 })
  return html`<button @click=${() => { state.count += 1 }}>${() => state.count}</button>`
})

export default html`${App()}`
```

## Forbidden patterns

Do not produce these:

- React code
- JSX with React imports
- Tailwind-only styling that depends on host CSS
- `export default component(() => { ... })`
- `items.map(item => html\`...\`)`
- prose followed by a code block
- fake approval flows that should use `ask_user`

The sandbox requires templates to be precompiled. That means nested `html\`...\`` creation inside loops, callbacks, or `map()` bodies will fail at runtime.

## Repeated content

For rows, cards, options, tabs, chips, and repeated fields:

1. Define a reusable Arrow component once.
2. Pass plain props into that component.
3. Render repeated items with `items.map(item => Row(item))`.

Correct pattern:

```ts
import { html, reactive, component } from '@arrow-js/core'

const state = reactive({ selectedId: '1' })

const EventRow = component((props) => html`
  <button
    style="display:flex;justify-content:space-between;align-items:center;width:100%;padding:12px;border:none;border-radius:10px;cursor:pointer;background:${() => state.selectedId === props.id ? '#ffffff' : 'rgba(255,255,255,0.06)'};color:${() => state.selectedId === props.id ? '#111111' : '#ffffff'}"
    @click=${() => { state.selectedId = props.id }}
  >
    <span>${props.summary}</span>
    <span>${props.start}</span>
  </button>
`)

const events = [
  { id: '1', summary: 'Design Review', start: '2026-03-30' },
  { id: '2', summary: 'Weekly Demo', start: '2026-04-03' },
]

export default html`<div style="display:grid;gap:8px">${events.map((event) => EventRow(event))}</div>`
```

Incorrect pattern:

```ts
${events.map((event) => html`<div>${event.summary}</div>`)}
```

## Interactive components

If the user would benefit from interaction, build the interaction directly with Arrow state and DOM events.

### Form pattern

Use local reactive state, update it from inputs, and submit through the host bridge only when the action is explicit.

```ts
import { html, reactive, component } from '@arrow-js/core'

const ExampleForm = component(() => {
  const state = reactive({
    name: '',
    role: 'designer',
    subscribe: true,
  })

  const submit = () => {
    output({
      type: 'chat.submit',
      text: `Form submitted for ${state.name || 'unknown user'}`,
      data: {
        name: state.name,
        role: state.role,
        subscribe: state.subscribe,
      },
    })
  }

  return html`<div style="display:grid;gap:12px;padding:16px;border:1px solid rgba(255,255,255,0.12);border-radius:16px">
    <label style="display:grid;gap:6px">
      <span style="font-size:13px;color:rgba(255,255,255,0.72)">Name</span>
      <input
        value="${() => state.name}"
        style="height:40px;padding:0 12px;border-radius:10px;border:1px solid rgba(255,255,255,0.12);background:#111;color:#fff"
        @input=${(event) => { state.name = event.target?.value ?? '' }}
      />
    </label>

    <label style="display:grid;gap:6px">
      <span style="font-size:13px;color:rgba(255,255,255,0.72)">Role</span>
      <select
        value="${() => state.role}"
        style="height:40px;padding:0 12px;border-radius:10px;border:1px solid rgba(255,255,255,0.12);background:#111;color:#fff"
        @change=${(event) => { state.role = event.target?.value ?? 'designer' }}
      >
        <option value="designer">Designer</option>
        <option value="engineer">Engineer</option>
        <option value="founder">Founder</option>
      </select>
    </label>

    <label style="display:flex;align-items:center;gap:8px">
      <input
        type="checkbox"
        checked="${() => state.subscribe}"
        @change=${(event) => { state.subscribe = !!event.target?.checked }}
      />
      <span style="font-size:13px;color:rgba(255,255,255,0.72)">Subscribe to updates</span>
    </label>

    <button
      style="height:40px;border:none;border-radius:10px;background:#ffffff;color:#111111;font-weight:600;cursor:pointer"
      @click=${submit}
    >
      Submit
    </button>
  </div>`
})

export default html`${ExampleForm()}`
```

### Compact interactive card

```ts
import { html, reactive, component } from '@arrow-js/core'

const Counter = component(() => {
  const state = reactive({ count: 0 })

  return html`<div style="display:grid;gap:12px;padding:16px;border:1px solid rgba(255,255,255,0.12);border-radius:16px">
    <div style="font-size:14px;color:rgba(255,255,255,0.72)">Counter</div>
    <div style="font-size:28px;font-weight:600">${() => state.count}</div>
    <button
      style="height:40px;border:none;border-radius:10px;background:#ffffff;color:#111111;font-weight:600;cursor:pointer"
      @click=${() => { state.count += 1 }}
    >
      Increment
    </button>
  </div>`
})

export default html`${Counter()}`
```

## Host actions

The sandbox may call `output(...)` for a small, explicit set of host actions.

Supported payloads:

- `output({ type: 'open-url', url: 'https://example.com' })`
- `output({ type: 'copy', text: 'value to copy' })`
- `output({ type: 'chat.submit', text: '...', data: { ... } })`

Do not invent custom action types unless the prompt explicitly documents them.

Use `chat.submit` only when a UI submission should intentionally become a new user turn in chat. Do not auto-submit on every keystroke or selection change.

## Product guidance

- Use Arrow when interaction, compact visualization, or scannable structure materially improves the answer.
- Use plain text when a normal answer is clearer.
- For blocking approvals or decisions, use `ask_user` instead of inventing your own approval widget.
- Keep modules self-contained and robust. Prefer one file. Avoid clever abstractions that reduce reliability.
- If the request is simple, favor static or lightly interactive UI over a complex mini-app.

## Troubleshooting

If you are about to generate any of these patterns, stop and rewrite:

- default-exported `component(...)`
- `map(() => html\`...\`)`
- React imports
- host-only assumptions like routing helpers, global CSS classes, or framework APIs

When in doubt, build:

1. a single `App` component
2. one or two small reusable row/card components
3. a root `export default html\`${App()}\``
