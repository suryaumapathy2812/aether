import { html, reactive } from '@arrow-js/core'

const raw = `{{DATA}}`
let data = {}
try { data = JSON.parse(raw) } catch {}

const results = Array.isArray(data.results) ? data.results : []

const state = reactive({ results })

export default html`
  <div style="font-family:system-ui,sans-serif;display:flex;flex-direction:column;gap:8px;">
    ${() => state.results.map((r, i) => html`
      <a href="${r.url || '#'}" target="_blank" rel="noreferrer" style="display:block;border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:12px;text-decoration:none;" .key="${i}">
        <div style="font-size:13px;font-weight:500;color:rgba(255,255,255,.9);">${r.title || 'Untitled'}</div>
        <div style="font-size:11px;color:#38bdf8;margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${r.url || ''}</div>
        ${r.snippet ? html`<div style="font-size:12px;color:rgba(255,255,255,.45);margin-top:4px;">${r.snippet}</div>` : ''}
      </a>`
    )}
  </div>
`
