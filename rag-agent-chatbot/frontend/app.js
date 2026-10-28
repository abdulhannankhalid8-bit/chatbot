const API = "/api";
const $ = (s) => document.querySelector(s);

const els = {
  messages: $("#messages"), empty: $("#empty"), chat: $("#chat"),
  input: $("#input"), send: $("#send"), form: $("#composer"),
  model: $("#model"), status: $("#status-bar"),
  chunkStat: $("#chunk-stat"), ingestBtn: $("#ingest-btn"),
  ingestMsg: $("#ingest-msg"), fileInput: $("#file-input"),
  keyWarning: $("#key-warning"),
};

const history = []; // {role, content} for multi-turn

marked.setOptions({ gfm: true, breaks: true });
const md = (t) => DOMPurify.sanitize(marked.parse(t || ""));

function scrollDown() {
  els.chat.scrollTop = els.chat.scrollHeight;
}

function addUser(text) {
  els.empty.hidden = true;
  const row = document.createElement("div");
  row.className = "msg user";
  row.innerHTML = `<div class="bubble"></div>`;
  row.querySelector(".bubble").textContent = text;
  els.messages.appendChild(row);
  scrollDown();
}

function addAssistantShell() {
  const row = document.createElement("div");
  row.className = "msg assistant";
  row.innerHTML = `
    <div class="tool-line" hidden></div>
    <div class="bubble"></div>
    <div class="sources" hidden></div>
  `;
  els.messages.appendChild(row);
  scrollDown();
  return {
    row,
    tool: row.querySelector(".tool-line"),
    bubble: row.querySelector(".bubble"),
    sources: row.querySelector(".sources"),
  };
}

function setStatus(msg) {
  if (!msg) {
    els.status.hidden = true;
    els.status.textContent = "";
    return;
  }
  els.status.hidden = false;
  els.status.textContent = msg;
}

function updateSend() {
  els.send.disabled = !els.input.value.trim() || els.send.dataset.busy === "1";
}

async function refreshStats() {
  try {
    const h = await (await fetch(`${API}/health`)).json();
    els.chunkStat.textContent = `Chunks: ${h.chunks ?? 0}`;
    els.keyWarning.hidden = h.has_key;
  } catch {
    els.chunkStat.textContent = "Chunks: ?";
  }
}

async function sendMessage(text) {
  text = text.trim();
  if (!text || els.send.dataset.busy === "1") return;

  addUser(text);
  history.push({ role: "user", content: text });
  els.input.value = "";
  updateSend();

  const ui = addAssistantShell();
  els.send.dataset.busy = "1";
  updateSend();

  let reply = "";
  try {
    const res = await fetch(`${API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        model: els.model.value,
        history: history.slice(0, -1),
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const line = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 2);
        if (!line.startsWith("data:")) continue;
        const ev = JSON.parse(line.slice(5));

        if (ev.type === "status") {
          setStatus(ev.message);
        } else if (ev.type === "tool") {
          ui.tool.hidden = false;
          ui.tool.textContent = `🔍 Searching: ${ev.query}`;
          setStatus(`Searching docs for “${ev.query}”…`);
        } else if (ev.type === "delta") {
          setStatus(null);
          reply += ev.content;
          ui.bubble.innerHTML = md(reply);
          scrollDown();
        } else if (ev.type === "done") {
          setStatus(null);
          if (ev.sources && ev.sources.length) {
            ui.sources.hidden = false;
            ui.sources.innerHTML =
              "Sources: " +
              ev.sources.map((s) => `<span>${s}</span>`).join("");
          }
        } else if (ev.type === "error") {
          ui.bubble.innerHTML = `<div class="error">${ev.message}</div>`;
        }
      }
    }

    if (reply) history.push({ role: "assistant", content: reply });
  } catch (e) {
    ui.bubble.innerHTML = `<div class="error">${e.message || "Network error"}</div>`;
  } finally {
    els.send.dataset.busy = "0";
    updateSend();
    setStatus(null);
  }
}

els.form.addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage(els.input.value);
});
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage(els.input.value);
  }
});
els.input.addEventListener("input", updateSend);

document.querySelectorAll(".chip").forEach((c) =>
  c.addEventListener("click", () => sendMessage(c.dataset.q))
);

els.ingestBtn.addEventListener("click", async () => {
  els.ingestMsg.textContent = "Ingesting…";
  try {
    const res = await fetch(`${API}/ingest`, { method: "POST" });
    const data = await res.json();
    els.ingestMsg.textContent = `Done: ${data.files} files, ${data.chunks} chunks`;
    refreshStats();
  } catch (e) {
    els.ingestMsg.textContent = "Ingest failed — is the server running?";
  }
});

els.fileInput.addEventListener("change", async () => {
  const file = els.fileInput.files[0];
  if (!file) return;
  els.ingestMsg.textContent = `Uploading ${file.name}…`;
  const fd = new FormData();
  fd.append("file", file);
  try {
    const res = await fetch(`${API}/upload`, { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");
    els.ingestMsg.textContent = `Uploaded ${data.file}: ${data.chunks} chunks`;
    refreshStats();
  } catch (e) {
    els.ingestMsg.textContent = e.message || "Upload failed";
  }
  els.fileInput.value = "";
});

(async function init() {
  try {
    const { models, default: def } = await (await fetch(`${API}/models`)).json();
    models.forEach((m) => els.model.add(new Option(m.label, m.id)));
    els.model.value = def;
  } catch {
    els.model.add(new Option("Server offline", ""));
  }
  await refreshStats();
  els.input.focus();
})();
