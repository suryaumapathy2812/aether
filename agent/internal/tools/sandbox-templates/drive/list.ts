import { html, reactive } from '@arrow-js/core'

const raw = `{{DATA}}`
let data = {}
try { data = JSON.parse(raw) } catch {}

const files = Array.isArray(data.files) ? data.files : (Array.isArray(data) ? data : [])

function formatSize(bytes) {
  const b = Number(bytes) || 0
  if (b < 1024) return b + ' B'
  if (b < 1024*1024) return (b/1024).toFixed(1) + ' KB'
  return (b/(1024*1024)).toFixed(1) + ' MB'
}

function fileIcon(mime) {
  if (!mime) return '📄'
  if (mime.includes('folder')) return '📁'
  if (mime.includes('spreadsheet')) return '📊'
  if (mime.includes('document')) return '📝'
  if (mime.includes('presentation')) return '📽️'
  if (mime.includes('pdf')) return '📕'
  if (mime.startsWith('image/')) return '🖼️'
  return '📄'
}

const state = reactive({ files })

export default html`
  <div style="font-family:system-ui,sans-serif;">
    <div style="padding:0 4px 8px;font-size:12px;font-weight:500;color:rgba(255,255,255,.6);">
      ${state.files.length} file${state.files.length !== 1 ? 's' : ''}
    </div>
    ${() => state.files.length === 0
      ? html`<div style="text-align:center;padding:32px;color:rgba(255,255,255,.3);font-size:13px;">No files found.</div>`
      : html`<div style="border:1px solid rgba(255,255,255,.06);border-radius:12px;overflow:hidden;">
        ${state.files.map((f, i) => html`
          <div style="display:flex;align-items:center;gap:10px;padding:10px 12px;border-bottom:1px solid rgba(255,255,255,.04);" .key="${f.id || i}">
            <span style="font-size:16px;flex-shrink:0;">${fileIcon(f.mimeType)}</span>
            <div style="min-width:0;flex:1;">
              <div style="font-size:12px;font-weight:500;color:rgba(255,255,255,.8);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${f.name || 'Untitled'}</div>
              <div style="font-size:10px;color:rgba(255,255,255,.35);margin-top:2px;">${f.mimeType || ''} ${f.size ? '· ' + formatSize(f.size) : ''}</div>
            </div>
            ${f.webViewLink ? html`<a href="${f.webViewLink}" target="_blank" rel="noreferrer" style="font-size:10px;color:#38bdf8;text-decoration:none;">Open →</a>` : ''}
          </div>`
        )}
      </div>`
    }
  </div>
`
