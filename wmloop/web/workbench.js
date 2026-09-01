const state = { modes: [], mode: "hybrid", campaigns: [], graph: { nodes: [], edges: [] }, transform: { x: 0, y: 0, scale: 1 } };
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}

function selectMode(mode) {
  state.mode = mode;
  $$("#mode-switch button").forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
  const item = state.modes.find((row) => row.mode === mode);
  $("#mode-description").textContent = item?.description || "";
  $("#mode-state").textContent = mode === "hybrid" ? "推荐" : mode === "causal_discovery" ? "需要因果证据" : "冷启动";
  $$(".causal-field").forEach((field) => field.classList.toggle("hidden", mode === "quick_start"));
}

function renderModes() {
  const host = $("#mode-switch");
  host.innerHTML = "";
  state.modes.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.mode = item.mode;
    button.setAttribute("role", "radio");
    button.textContent = ({ quick_start: "快速启动", causal_discovery: "因果发现", hybrid: "混合模式" })[item.mode] || item.label;
    button.addEventListener("click", () => selectMode(item.mode));
    host.appendChild(button);
  });
  selectMode(state.mode);
}

function modeLabel(mode) {
  return ({ quick_start: "快速启动", causal_discovery: "因果发现", hybrid: "混合模式" })[mode] || mode || "兼容模式";
}

function nodeLabel(node) { return node.display_label || node.value || node.id; }
function nodeKind(node) { return node.display_kind || node.kind; }

function evidenceStage(campaign) {
  const stages = campaign.research_mode_plan?.stages || [];
  const current = stages.find((row) => row.state === "active") || stages.find((row) => row.state === "pending");
  return current?.stage || "-";
}

function renderCampaigns() {
  const body = $("#campaign-table");
  body.innerHTML = "";
  $("#campaign-count").textContent = `${state.campaigns.length} 个任务`;
  $("#empty-campaigns").style.display = state.campaigns.length ? "none" : "block";
  $("#overview-total").textContent = state.campaigns.length;
  $("#overview-running").textContent = state.campaigns.filter((item) => item.status === "running").length;
  $("#overview-queued").textContent = state.campaigns.filter((item) => ["created", "confirmed", "queued"].includes(item.status)).length;
  $("#overview-blocked").textContent = state.campaigns.filter((item) => ["blocked", "failed"].includes(item.status)).length;
  state.campaigns.slice().reverse().forEach((campaign) => {
    const row = document.createElement("tr");
    const canCancel = ["created", "confirmed", "queued", "running"].includes(campaign.status);
    row.innerHTML = `<td><button class="table-link" type="button"><strong>${escapeHtml(campaign.campaign_id)}</strong><small>${escapeHtml((campaign.revision_id || "").slice(0, 18))}</small></button></td><td>${escapeHtml(modeLabel(campaign.research_mode))}</td><td><span class="badge ${escapeHtml(campaign.status)}">${escapeHtml(statusLabel(campaign.status))}</span></td><td>${escapeHtml(campaign.goal)}</td><td>${escapeHtml(evidenceStage(campaign))}</td><td></td>`;
    row.querySelector(".table-link").addEventListener("click", () => showCampaignDetail(campaign));
    if (canCancel) {
      const button = document.createElement("button"); button.className = "row-action"; button.textContent = "取消";
      button.addEventListener("click", async () => { await api(`/api/campaigns/${campaign.campaign_id}/cancel`, { method: "POST", body: "{}" }); await refreshCampaigns(); });
      row.lastElementChild.appendChild(button);
    }
    body.appendChild(row);
  });
}

function statusLabel(status) { return ({ created: "草稿", confirmed: "已确认", queued: "排队中", running: "运行中", completed: "已完成", blocked: "已阻断", failed: "失败", cancelled: "已取消" })[status] || status; }
function showCampaignDetail(campaign) {
  const detail = $("#campaign-detail");
  const result = campaign.execution_result || {};
  const manifest = result.pipeline_manifest || {};
  detail.hidden = false;
  detail.innerHTML = `<div class="detail-head"><div><span class="eyebrow">研究任务</span><h3>${escapeHtml(campaign.campaign_id)}</h3></div><button type="button" class="detail-close" aria-label="关闭">×</button></div><div class="detail-grid"><div><span>状态</span><strong class="badge ${escapeHtml(campaign.status)}">${escapeHtml(statusLabel(campaign.status))}</strong></div><div><span>目标指标</span><strong>${escapeHtml((campaign.target_metrics || []).join(", ") || "由评测契约决定")}</strong></div><div><span>适配器</span><strong>${escapeHtml(campaign.adapter_profile || "自动选择")}</strong></div><div><span>修订</span><strong>${escapeHtml(campaign.revision_id || "-")}</strong></div></div>${manifest.blocked_stage ? `<div class="detail-callout"><strong>阻断阶段：${escapeHtml(manifest.blocked_stage)}</strong><p>评测资产或运行条件尚未满足，系统没有继续消耗 GPU。</p></div>` : ""}<p class="detail-goal">${escapeHtml(campaign.goal)}</p>`;
  detail.querySelector(".detail-close").addEventListener("click", () => { detail.hidden = true; });
  detail.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function refreshCampaigns() {
  const result = await api("/api/campaigns"); state.campaigns = result.items || []; renderCampaigns();
}

async function refreshGraph() {
  state.graph = await api("/api/graph");
  const kinds = [...new Set((state.graph.nodes || []).map((node) => node.kind))].sort();
  const select = $("#kind-filter"); select.innerHTML = '<option value="">全部节点</option>' + kinds.map((kind) => `<option value="${escapeHtml(kind)}">${escapeHtml(kind)}</option>`).join("");
  initializeGraph();
}

function initializeGraph() {
  const canvas = $("#graph-canvas"), box = canvas.getBoundingClientRect(), ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(box.width * ratio)); canvas.height = Math.max(1, Math.floor(box.height * ratio));
  const width = box.width, height = box.height;
  (state.graph.nodes || []).forEach((node, index) => { const angle = index * 2.39996; const radius = 35 + Math.sqrt(index) * 27; node.x = width / 2 + Math.cos(angle) * radius; node.y = height / 2 + Math.sin(angle) * radius; node.vx = 0; node.vy = 0; });
  state.transform = { x: 0, y: 0, scale: 1 }; $("#graph-empty").style.display = state.graph.nodes?.length ? "none" : "block";
  simulateGraph();
}

function visibleGraph() {
  const kind = $("#kind-filter").value; const nodes = (state.graph.nodes || []).filter((node) => !kind || node.kind === kind); const ids = new Set(nodes.map((node) => node.id)); const edges = (state.graph.edges || []).filter((edge) => ids.has(edge.source) && ids.has(edge.target)); return { nodes, edges };
}

function simulateGraph() {
  const { nodes, edges } = visibleGraph(); const byId = new Map(nodes.map((node) => [node.id, node]));
  for (let tick = 0; tick < 90; tick += 1) {
    nodes.forEach((a, i) => { for (let j = i + 1; j < nodes.length; j += 1) { const b = nodes[j], dx = a.x - b.x || .1, dy = a.y - b.y || .1, d2 = dx * dx + dy * dy, f = Math.min(2, 900 / d2); a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f; } });
    edges.forEach((edge) => { const a = byId.get(edge.source), b = byId.get(edge.target); if (!a || !b) return; const dx = b.x - a.x, dy = b.y - a.y, distance = Math.hypot(dx, dy) || 1, f = (distance - 105) * .002; a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f; });
    nodes.forEach((node) => { node.vx += (400 - node.x) * .0005; node.vy += (300 - node.y) * .0005; node.vx *= .82; node.vy *= .82; node.x += node.vx; node.y += node.vy; });
  }
  drawGraph();
}

function nodeColor(kind) { const colors = { artifact: "#61706a", campaign: "#176b4d", probe: "#2f5f8f", candidate: "#9a6415", verdict: "#a33a31", research_source: "#775b91", model: "#3c756f" }; return colors[kind] || "#7a8580"; }
function drawGraph() {
  const canvas = $("#graph-canvas"), context = canvas.getContext("2d"), ratio = window.devicePixelRatio || 1, { nodes, edges } = visibleGraph(), byId = new Map(nodes.map((node) => [node.id, node])), t = state.transform;
  context.setTransform(ratio, 0, 0, ratio, 0, 0); context.clearRect(0, 0, canvas.width / ratio, canvas.height / ratio); context.save(); context.translate(t.x, t.y); context.scale(t.scale, t.scale);
  context.strokeStyle = "#cbd3cf"; context.lineWidth = 1 / t.scale; edges.forEach((edge) => { const a = byId.get(edge.source), b = byId.get(edge.target); if (!a || !b) return; context.beginPath(); context.moveTo(a.x, a.y); context.lineTo(b.x, b.y); context.stroke(); });
  nodes.forEach((node) => { context.fillStyle = nodeColor(node.kind); context.beginPath(); context.arc(node.x, node.y, 6.5, 0, Math.PI * 2); context.fill(); if (t.scale > .65) { context.fillStyle = "#27312d"; context.font = "11px system-ui"; context.fillText(String(nodeLabel(node)).slice(0, 32), node.x + 10, node.y + 4); } }); context.restore();
}

function graphPoint(event) { const rect = $("#graph-canvas").getBoundingClientRect(), t = state.transform; return { x: (event.clientX - rect.left - t.x) / t.scale, y: (event.clientY - rect.top - t.y) / t.scale }; }
function bindGraph() {
  const canvas = $("#graph-canvas"); let drag = null, pan = null;
  canvas.addEventListener("pointerdown", (event) => { const point = graphPoint(event), { nodes } = visibleGraph(); drag = nodes.find((node) => Math.hypot(node.x - point.x, node.y - point.y) < 14 / state.transform.scale); if (!drag) pan = { x: event.clientX - state.transform.x, y: event.clientY - state.transform.y }; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener("pointermove", (event) => { if (drag) { const point = graphPoint(event); drag.x = point.x; drag.y = point.y; drawGraph(); } else if (pan) { state.transform.x = event.clientX - pan.x; state.transform.y = event.clientY - pan.y; drawGraph(); } });
  canvas.addEventListener("pointerup", (event) => { if (drag) showNode(drag); drag = null; pan = null; canvas.releasePointerCapture(event.pointerId); });
  canvas.addEventListener("wheel", (event) => { event.preventDefault(); state.transform.scale = Math.max(.25, Math.min(3, state.transform.scale * (event.deltaY < 0 ? 1.12 : .89))); drawGraph(); }, { passive: false });
}

function showNode(node) { const edges = (state.graph.edges || []).filter((edge) => edge.source === node.id || edge.target === node.id); $("#node-detail").innerHTML = `<h3>${escapeHtml(nodeLabel(node))}</h3><dl><dt>类型</dt><dd>${escapeHtml(nodeKind(node))}</dd><dt>技术标识</dt><dd>${escapeHtml(node.id)}</dd><dt>关系</dt><dd>${edges.length}</dd><dt>证据来源</dt><dd><ul class="source-list">${(node.sources || []).slice(0, 12).map((source) => `<li>${escapeHtml(source)}</li>`).join("") || "<li>无</li>"}</ul></dd></dl>`; }
function escapeHtml(value) { return String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char])); }

$("#campaign-form").addEventListener("submit", async (event) => { event.preventDefault(); const data = Object.fromEntries(new FormData(event.currentTarget)); const payload = { goal: data.goal, research_mode: state.mode, queue: true }; ["model", "dataset", "budget", "adapter", "cpbe_request", "cpbe_history", "target_metrics"].forEach((key) => { if (data[key]) payload[key] = data[key]; }); try { $("#form-error").textContent = ""; await api("/api/campaigns", { method: "POST", body: JSON.stringify(payload) }); event.currentTarget.reset(); await refreshCampaigns(); } catch (error) { $("#form-error").textContent = error.message; } });
$("#run-queue").addEventListener("click", async () => { try { $("#form-error").textContent = "正在执行队列..."; await api("/api/dispatch", { method: "POST", body: JSON.stringify({ max_parallel: 1 }) }); $("#form-error").textContent = ""; await refreshCampaigns(); } catch (error) { $("#form-error").textContent = error.message; } });
$("#refresh").addEventListener("click", async () => { await Promise.all([refreshCampaigns(), refreshGraph()]); });
$("#kind-filter").addEventListener("change", simulateGraph); $("#fit-graph").addEventListener("click", initializeGraph);
$$('.nav-item').forEach((button) => button.addEventListener("click", async () => { $$('.nav-item').forEach((item) => item.classList.toggle("active", item === button)); $$('.view').forEach((view) => view.classList.remove("active")); $(`#${button.dataset.view}-view`).classList.add("active"); if (button.dataset.view === "graph") { await refreshGraph(); } }));
window.addEventListener("resize", () => { if ($("#graph-view").classList.contains("active")) initializeGraph(); });

async function boot() { try { const [modes, project] = await Promise.all([api("/api/modes"), api("/api/project"), refreshCampaigns()]); state.modes = modes.items; renderModes(); bindGraph(); const values = project.values || {}; ["model", "dataset", "budget", "adapter", "target_metrics"].forEach((key) => { if (values[key] && $( `[name="${key}"]`)) $( `[name="${key}"]`).value = Array.isArray(values[key]) ? values[key].join(" ") : values[key]; }); $("#project-state").textContent = project.configured ? "已加载项目配置；默认值由系统内部管理。" : "未发现 verdiwm.toml；可在高级设置中提供项目路径。"; $("#api-state").textContent = "本地服务已连接"; $("#api-state").classList.add("ready"); } catch (error) { $("#api-state").textContent = error.message; } }
boot();
