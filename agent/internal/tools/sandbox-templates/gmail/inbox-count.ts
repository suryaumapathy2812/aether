import { html, reactive } from '@arrow-js/core'

const raw = `{{DATA}}`
let data = {}
try { data = JSON.parse(raw) } catch {}

const state = reactive({
  total: Number(data.messagesTotal) || 0,
  unread: Number(data.messagesUnread) || 0,
  threads: Number(data.threadsTotal) || 0,
  unreadThreads: Number(data.threadsUnread) || 0,
})

export default html`
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;font-family:system-ui,sans-serif;">
    <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:16px;">
      <div style="font-size:28px;font-weight:600;color:rgba(255,255,255,.9);">${() => state.total.toLocaleString()}</div>
      <div style="font-size:12px;color:rgba(255,255,255,.45);margin-top:4px;">Messages</div>
    </div>
    <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:16px;">
      <div style="font-size:28px;font-weight:600;color:${() => state.unread > 0 ? 'rgba(56,189,248,.9)' : 'rgba(255,255,255,.9)'};">${() => state.unread.toLocaleString()}</div>
      <div style="font-size:12px;color:rgba(255,255,255,.45);margin-top:4px;">Unread</div>
    </div>
    <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:16px;">
      <div style="font-size:28px;font-weight:600;color:rgba(255,255,255,.9);">${() => state.threads.toLocaleString()}</div>
      <div style="font-size:12px;color:rgba(255,255,255,.45);margin-top:4px;">Threads</div>
    </div>
    <div style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:16px;">
      <div style="font-size:28px;font-weight:600;color:${() => state.unreadThreads > 0 ? 'rgba(56,189,248,.9)' : 'rgba(255,255,255,.9)'};">${() => state.unreadThreads.toLocaleString()}</div>
      <div style="font-size:12px;color:rgba(255,255,255,.45);margin-top:4px;">Unread threads</div>
    </div>
  </div>
`
