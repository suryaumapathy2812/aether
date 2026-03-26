import { html, reactive } from '@arrow-js/core'

const raw = `{{DATA}}`
let data = {}
try { data = JSON.parse(raw) } catch {}

const state = reactive({ link: data.htmlLink || '' })

export default html`
  <div style="font-family:system-ui,sans-serif;border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:16px;display:flex;align-items:flex-start;gap:12px;">
    <div style="width:32px;height:32px;border-radius:50%;background:rgba(16,185,129,.1);color:#10b981;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>
    </div>
    <div>
      <div style="font-size:13px;font-weight:500;color:rgba(255,255,255,.85);">Event created</div>
      ${() => state.link ? html`<a href="${state.link}" target="_blank" rel="noreferrer" style="font-size:11px;color:#38bdf8;margin-top:4px;display:inline-block;text-decoration:none;">Open in Google Calendar →</a>` : ''}
    </div>
  </div>
`
