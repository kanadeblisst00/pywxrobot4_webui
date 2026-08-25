const state = { models: [], settings: null, summaries: [], currentJob: null };

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try { const payload = await response.json(); detail = payload.detail || detail; } catch {}
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return response.status === 204 ? null : response.json();
}

function toast(message, type = "success") {
  const element = document.createElement("div");
  element.className = `toast ${type}`;
  element.textContent = message;
  $("#toast-region").append(element);
  setTimeout(() => element.remove(), 3500);
}

const viewMeta = {
  dashboard: ["OVERVIEW", "工作台"], summary: ["SUMMARY LAB", "摘要试运行"],
  models: ["MODEL ROUTER", "模型管理"], history: ["ARCHIVE", "历史记录"],
  settings: ["PIPELINE", "流水线设置"],
};

function showView(name) {
  $$(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${name}`));
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === name));
  $("#page-eyebrow").textContent = viewMeta[name][0];
  $("#page-title").textContent = viewMeta[name][1];
  if (name === "history") loadHistory();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function modelName(id) {
  return state.models.find((item) => item.id === id)?.name || id || "—";
}

function statusLabel(status) {
  return ({ pending: "等待中", running: "总结中", completed: "已完成", failed: "失败" })[status] || status;
}

function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("zh-CN", { hour12: false });
}

async function loadHealth() {
  try {
    await api("/api/v1/health");
    $("#health-dot").classList.add("ok");
    $("#health-text").textContent = "服务运行正常";
  } catch {
    $("#health-dot").classList.remove("ok");
    $("#health-text").textContent = "服务连接失败";
  }
}

async function loadDashboard() {
  const metrics = await api("/api/v1/dashboard");
  $("#metric-total").textContent = metrics.total;
  $("#metric-completed").textContent = metrics.completed;
  $("#metric-running").textContent = metrics.running;
  $("#metric-models").textContent = metrics.enabled_models;
  renderActiveModel();
  renderRecent();
}

function renderActiveModel() {
  const profile = state.models.find((item) => item.id === state.settings?.default_model_profile_id);
  const target = $("#active-model-card");
  if (!profile) { target.innerHTML = '<span class="muted">未配置默认模型</span>'; return; }
  target.innerHTML = `<div class="model-icon">Q</div><div><b>${escapeHtml(profile.name)}</b><small>${escapeHtml(profile.model)} · ${escapeHtml(profile.base_url)}</small></div>`;
}

function renderRecent() {
  const target = $("#recent-list");
  if (!state.summaries.length) { target.innerHTML = '<p class="empty">还没有摘要任务</p>'; return; }
  target.innerHTML = state.summaries.slice(0, 4).map((item) => `<div class="recent-item" data-summary-id="${item.id}"><span><b>${escapeHtml(item.room_name)}</b><small>${formatDate(item.created_at)}</small></span><span class="status ${item.status}">${statusLabel(item.status)}</span></div>`).join("");
}

function populateModelSelects() {
  const enabled = state.models.filter((item) => item.enabled);
  const options = enabled.map((item) => `<option value="${item.id}">${escapeHtml(item.name)} · ${escapeHtml(item.model)}</option>`).join("");
  $("#summary-model").innerHTML = options;
  $("#default-model").innerHTML = options;
  if (state.settings?.default_model_profile_id) {
    $("#summary-model").value = state.settings.default_model_profile_id;
    $("#default-model").value = state.settings.default_model_profile_id;
  }
}

async function loadModels() {
  state.models = await api("/api/v1/model-profiles");
  populateModelSelects();
  renderModels();
}

function renderModels() {
  $("#model-grid").innerHTML = state.models.map((profile) => `
    <article class="model-card ${profile.enabled ? "" : "disabled"}" data-profile="${profile.id}">
      <i class="model-state ${profile.enabled ? "enabled" : ""}"></i>
      <div class="model-card-top"><div class="model-icon">${profile.provider === "llama.cpp" ? "L" : profile.provider === "ollama" ? "O" : "AI"}</div><div><h3>${escapeHtml(profile.name)}</h3><span class="status neutral">${profile.is_builtin ? "内置预设" : "自定义"}</span></div></div>
      <p>${escapeHtml(profile.description || "暂无说明")}</p>
      <div class="model-meta"><span>模型 <b title="${escapeHtml(profile.model)}">${escapeHtml(profile.model)}</b></span><span>接口 <b title="${escapeHtml(profile.base_url)}">${escapeHtml(profile.base_url)}</b></span><span>结构化输出 <b>${profile.supports_json_schema ? "JSON Schema" : "提示词约束"}</b></span></div>
      <div class="model-actions"><button class="button ghost" data-action="test-model">检测连接</button><button class="button ghost" data-action="edit-model">编辑</button>${profile.is_builtin ? "" : '<button class="button ghost" data-action="delete-model">删除</button>'}</div>
      <div class="model-test-result"></div>
    </article>`).join("");
}

function openModelDialog(profile = null) {
  $("#model-dialog-title").textContent = profile ? "编辑模型配置" : "添加模型配置";
  $("#model-profile-id").value = profile?.id || "";
  $("#model-name").value = profile?.name || "";
  $("#model-provider").value = profile?.provider || "llama.cpp";
  $("#model-base-url").value = profile?.base_url || "http://127.0.0.1:18080/v1";
  $("#model-name-id").value = profile?.model || "";
  $("#model-api-key").value = "";
  $("#model-description").value = profile?.description || "";
  $("#model-enabled").checked = profile?.enabled ?? true;
  $("#model-schema").checked = profile?.supports_json_schema ?? true;
  $("#model-dialog").showModal();
}

async function saveModel(event) {
  event.preventDefault();
  const id = $("#model-profile-id").value;
  const payload = {
    name: $("#model-name").value.trim(), provider: $("#model-provider").value,
    base_url: $("#model-base-url").value.trim(), model: $("#model-name-id").value.trim(),
    enabled: $("#model-enabled").checked, supports_json_schema: $("#model-schema").checked,
    description: $("#model-description").value.trim(),
  };
  const key = $("#model-api-key").value.trim();
  if (!id || key) payload.api_key = key;
  try {
    await api(id ? `/api/v1/model-profiles/${id}` : "/api/v1/model-profiles", { method: id ? "PUT" : "POST", body: JSON.stringify(payload) });
    $("#model-dialog").close();
    await loadModels();
    await loadDashboard();
    toast("模型配置已保存");
  } catch (error) { toast(error.message, "error"); }
}

async function testModel(card, id) {
  const output = $(".model-test-result", card);
  output.textContent = "正在检测接口和已安装模型…";
  try {
    const result = await api(`/api/v1/model-profiles/${id}/test`, { method: "POST" });
    output.textContent = result.detail;
    output.style.color = result.installed ? "#2e6c55" : result.ok ? "#b47b24" : "#b74646";
  } catch (error) { output.textContent = error.message; output.style.color = "#b74646"; }
}

async function loadSettings() {
  state.settings = await api("/api/v1/settings");
  $("#chunk-max-chars").value = state.settings.chunk_max_chars;
  $("#chunk-overlap").value = state.settings.chunk_overlap_messages;
  $("#max-output-tokens").value = state.settings.max_output_tokens;
  $("#temperature").value = state.settings.temperature;
  $("#ignored-types").value = state.settings.ignored_message_types.join(", ");
  $("#default-instruction").value = state.settings.custom_instruction;
  $("#keep-raw").checked = state.settings.keep_raw_messages;
  populateModelSelects();
}

async function saveSettings(event) {
  event.preventDefault();
  const payload = {
    default_model_profile_id: $("#default-model").value,
    chunk_max_chars: Number($("#chunk-max-chars").value),
    chunk_overlap_messages: Number($("#chunk-overlap").value),
    max_output_tokens: Number($("#max-output-tokens").value),
    temperature: Number($("#temperature").value),
    keep_raw_messages: $("#keep-raw").checked,
    ignored_message_types: $("#ignored-types").value.split(",").map((item) => item.trim()).filter(Boolean),
    custom_instruction: $("#default-instruction").value.trim(),
  };
  try { state.settings = await api("/api/v1/settings", { method: "PUT", body: JSON.stringify(payload) }); renderActiveModel(); toast("流水线设置已保存"); }
  catch (error) { toast(error.message, "error"); }
}

function parseMessages() {
  const value = $("#messages-json").value.trim();
  if (!value) return [];
  const parsed = JSON.parse(value);
  if (!Array.isArray(parsed)) throw new Error("消息 JSON 顶层必须是数组");
  return parsed;
}

function updateMessageCounter() {
  try { const messages = parseMessages(); $("#message-counter").textContent = `${messages.length} 条消息`; }
  catch { $("#message-counter").textContent = "JSON 格式待修正"; }
}

const exampleMessages = [
  { id: "m101", sender_id: "u1", sender_name: "林雨", timestamp: "2026-08-24T09:12:00+08:00", content: "登录模块联调完成了，今天可以合进测试分支。", message_type: "text" },
  { id: "m102", sender_id: "u2", sender_name: "周远", timestamp: "2026-08-24T09:14:00+08:00", content: "支付回调还有重复入账风险，我下午补幂等校验。", message_type: "text" },
  { id: "m103", sender_id: "u3", sender_name: "陈澄", timestamp: "2026-08-24T09:18:00+08:00", content: "那就定了：周远今天 18 点前提交修复，明早一起回归。", message_type: "text" },
  { id: "m104", sender_id: "u2", sender_name: "周远", timestamp: "2026-08-24T15:42:00+08:00", content: "幂等修复已提交 MR !284，等待陈澄审核。", message_type: "text" },
  { id: "m105", sender_id: "u3", sender_name: "陈澄", timestamp: "2026-08-24T16:10:00+08:00", content: "我来审核，是否需要补一条并发退款用例还没确定。", message_type: "text" },
];

async function submitSummary(event) {
  event.preventDefault();
  let messages;
  try { messages = parseMessages(); if (!messages.length) throw new Error("请至少提供一条消息"); }
  catch (error) { toast(error.message, "error"); return; }
  $("#run-summary").disabled = true;
  try {
    const record = await api("/api/v1/summaries", { method: "POST", body: JSON.stringify({ room_id: $("#room-id").value.trim(), room_name: $("#room-name").value.trim(), model_profile_id: $("#summary-model").value, custom_instruction: $("#custom-instruction").value.trim() || null, messages }) });
    state.currentJob = record.id;
    setJobState(record);
    pollSummary(record.id);
  } catch (error) { toast(error.message, "error"); $("#run-summary").disabled = false; }
}

async function pollSummary(id) {
  for (let attempt = 0; attempt < 720; attempt += 1) {
    await new Promise((resolve) => setTimeout(resolve, 1500));
    try {
      const record = await api(`/api/v1/summaries/${id}`);
      setJobState(record);
      if (["completed", "failed"].includes(record.status)) {
        $("#run-summary").disabled = false;
        await refreshData();
        return;
      }
    } catch (error) { toast(error.message, "error"); }
  }
  $("#run-summary").disabled = false;
  toast("轮询超时，任务仍可能在后台运行", "error");
}

function setJobState(record) {
  const badge = $("#job-status");
  badge.className = `status ${record.status}`;
  badge.textContent = statusLabel(record.status);
  if (record.status === "completed") renderResult(record.result, $("#summary-result"));
  else if (record.status === "failed") $("#summary-result").innerHTML = `<div class="result-empty"><div class="empty-orbit">!</div><h4>摘要失败</h4><p>${escapeHtml(record.error)}</p></div>`;
  else $("#summary-result").innerHTML = '<div class="result-empty"><div class="empty-orbit">✦</div><h4>模型正在整理消息</h4><p>长记录会先分段提取，再进行层级归并。</p></div>';
}

function renderItems(title, items, formatter) {
  if (!items?.length) return "";
  return `<section class="result-section"><h4>${title} · ${items.length}</h4>${items.map((item) => `<div class="result-item">${formatter(item)}<span class="evidence">证据：${escapeHtml((item.evidence_message_ids || []).join(", "))}</span></div>`).join("")}</section>`;
}

function renderResult(result, target) {
  if (!result) { target.innerHTML = '<p class="empty">没有可展示的结果</p>'; return; }
  target.innerHTML = `<div class="result-overview">${escapeHtml(result.overview || "未生成概览")}</div>
    <div class="result-stats"><span>${result.stats?.included_message_count || 0} 条消息</span><span>${result.stats?.participant_count || 0} 位成员</span><span>${result.stats?.chunk_count || 0} 个分段</span></div>
    ${renderItems("核心话题", result.topics, (item) => `<strong>${escapeHtml(item.title)}</strong>${escapeHtml(item.summary)}${item.participants?.length ? `<br><small>参与：${escapeHtml(item.participants.join("、"))}</small>` : ""}`)}
    ${renderItems("明确决定", result.decisions, (item) => `<strong>${escapeHtml(item.content)}</strong>${item.owner ? `负责人：${escapeHtml(item.owner)}<br>` : ""}`)}
    ${renderItems("行动项", result.action_items, (item) => `<strong>${escapeHtml(item.task)}</strong>负责人：${escapeHtml(item.owner || "未明确")} · 截止：${escapeHtml(item.deadline || "未明确")} · 状态：${escapeHtml(item.status)}`)}
    ${renderItems("未决问题", result.open_questions, (item) => `<strong>${escapeHtml(item.question)}</strong>${item.owner ? `跟进人：${escapeHtml(item.owner)}<br>` : ""}`)}
    ${renderItems("风险", result.risks, (item) => `<strong>${escapeHtml(item.content)}</strong>等级：${escapeHtml(item.level)}`)}`;
}

async function loadHistory() {
  state.summaries = await api("/api/v1/summaries?limit=200");
  const body = $("#history-body");
  body.innerHTML = state.summaries.length ? state.summaries.map((item) => `<tr data-id="${item.id}"><td><b>${escapeHtml(item.room_name)}</b><br><small class="muted">${escapeHtml(item.room_id || "无群 ID")}</small></td><td>${escapeHtml(modelName(item.model_profile_id))}</td><td><span class="status ${item.status}">${statusLabel(item.status)}</span></td><td>${formatDate(item.created_at)}</td><td>查看 →</td></tr>`).join("") : '<tr><td colspan="5" class="empty">还没有摘要历史</td></tr>';
  renderRecent();
}

async function showDetail(id) {
  try {
    const record = await api(`/api/v1/summaries/${id}`);
    $("#detail-title").textContent = record.room_name;
    const target = $("#detail-content");
    if (record.status === "completed") renderResult(record.result, target);
    else target.innerHTML = `<div class="result-empty"><span class="status ${record.status}">${statusLabel(record.status)}</span><p>${escapeHtml(record.error || "任务尚未完成")}</p></div>`;
    $("#detail-dialog").showModal();
  } catch (error) { toast(error.message, "error"); }
}

async function refreshData() {
  try { await Promise.all([loadHealth(), loadModels(), loadSettings(), loadHistory()]); await loadDashboard(); }
  catch (error) { toast(`刷新失败：${error.message}`, "error"); }
}

$("#main-nav").addEventListener("click", (event) => { const button = event.target.closest("[data-view]"); if (button) showView(button.dataset.view); });
document.addEventListener("click", async (event) => {
  const goto = event.target.closest("[data-goto]"); if (goto) showView(goto.dataset.goto);
  const recent = event.target.closest("[data-summary-id]"); if (recent) showDetail(recent.dataset.summaryId);
  const historyRow = event.target.closest("tr[data-id]"); if (historyRow) showDetail(historyRow.dataset.id);
  const action = event.target.closest("[data-action]");
  if (action) {
    const card = action.closest("[data-profile]"); const id = card.dataset.profile; const profile = state.models.find((item) => item.id === id);
    if (action.dataset.action === "test-model") testModel(card, id);
    if (action.dataset.action === "edit-model") openModelDialog(profile);
    if (action.dataset.action === "delete-model" && confirm(`确认删除模型配置“${profile.name}”吗？`)) { try { await api(`/api/v1/model-profiles/${id}`, { method: "DELETE" }); await loadModels(); toast("模型配置已删除"); } catch (error) { toast(error.message, "error"); } }
  }
});
$("#refresh-button").addEventListener("click", refreshData);
$("#add-model").addEventListener("click", () => openModelDialog());
$("#model-form").addEventListener("submit", saveModel);
$$('[data-close-dialog]').forEach((button) => button.addEventListener("click", () => $("#model-dialog").close()));
$$('[data-close-detail]').forEach((button) => button.addEventListener("click", () => $("#detail-dialog").close()));
$("#settings-form").addEventListener("submit", saveSettings);
$("#summary-form").addEventListener("submit", submitSummary);
$("#messages-json").addEventListener("input", updateMessageCounter);
$("#load-example").addEventListener("click", () => { $("#messages-json").value = JSON.stringify(exampleMessages, null, 2); updateMessageCounter(); });

refreshData();
