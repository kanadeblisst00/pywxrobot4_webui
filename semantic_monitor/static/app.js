const state = { topics: [], catalog: null, settings: null, datasets: [], datasetCatalog: [], runs: [] };
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.body && !(options.body instanceof FormData)) headers["Content-Type"] = "application/json";
  const token = $("#api-token")?.value.trim();
  if (token) headers["X-API-Token"] = token;
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail || `请求失败 (${response.status})`);
  }
  return response.status === 204 ? null : response.json();
}

function toast(message) {
  const el = $("#toast"); el.textContent = message; el.classList.add("show");
  clearTimeout(toast.timer); toast.timer = setTimeout(() => el.classList.remove("show"), 2600);
}

const titles = { overview: "监测概览", topics: "监测主题", playground: "消息测试台", events: "告警与复核", datasets: "训练数据", training: "训练与评测", models: "模型设置" };
$$('.nav-item').forEach(button => button.addEventListener('click', () => {
  $$('.nav-item').forEach(item => item.classList.toggle('active', item === button));
  $$('.view').forEach(view => view.classList.toggle('active', view.id === `view-${button.dataset.view}`));
  $('#page-title').textContent = titles[button.dataset.view];
  if (button.dataset.view === 'events') loadEvents();
  if (button.dataset.view === 'topics') loadTopics();
  if (button.dataset.view === 'models') loadModels();
  if (button.dataset.view === 'datasets') loadDatasets();
  if (button.dataset.view === 'training') loadTraining();
}));

$('#api-token').value = localStorage.getItem('semanticMonitorApiToken') || '';
$('#api-token').addEventListener('change', event => localStorage.setItem('semanticMonitorApiToken', event.target.value.trim()));

async function loadStats() {
  const value = await api('/api/v1/stats');
  $('#stat-today').textContent = value.events_today;
  $('#stat-pending').textContent = value.pending_review;
  $('#stat-topics').textContent = value.enabled_topics;
  $('#stat-fpr').textContent = `${Math.round(value.false_positive_rate * 100)}%`;
}

function resultMarkup(result) {
  const top = result.matched ? `<strong>${result.risk_level.toUpperCase()} 风险</strong><p>命中 ${result.matches.length} 个监测主题 · ${result.processing_ms} ms</p>` : `<strong>未命中</strong><p>当前消息未达到任何主题阈值 · ${result.processing_ms} ms</p>`;
  return `<div class="result-summary">${top}</div>${result.matches.map(item => `<div class="match-card"><div class="match-head"><b>${escapeHtml(item.topic_name)}</b><span>${Math.round(item.confidence * 100)}%</span></div><div class="score-bar"><i style="width:${item.confidence * 100}%"></i></div><p>${escapeHtml(item.evidence)}</p><small>语义 ${item.semantic_score ?? '—'} · 规则 ${item.rule_score} · 复核 ${item.classifier_score ?? '—'}</small></div>`).join('')}`;
}

$('#quick-form').addEventListener('submit', async event => {
  event.preventDefault(); const box = $('#quick-result'); box.textContent = '正在分析…';
  try { const result = await api('/api/v1/messages/analyze', { method: 'POST', body: JSON.stringify({ room_id: 'quick-test', sender_name: '测试成员', text: $('#quick-text').value, persist: false }) }); box.innerHTML = result.matched ? `命中 <b>${escapeHtml(result.matches[0].topic_name)}</b> · 置信度 ${Math.round(result.matches[0].confidence * 100)}%` : '未命中任何主题'; } catch (error) { box.textContent = error.message; }
});

$('#analyze-form').addEventListener('submit', async event => {
  event.preventDefault(); const form = new FormData(event.currentTarget); const target = $('#analysis-result'); target.innerHTML = '<p>正在完成规则、语义与上下文分析…</p>';
  try { const result = await api('/api/v1/messages/analyze', { method: 'POST', body: JSON.stringify({ room_id: form.get('room_id'), sender_name: form.get('sender_name'), text: form.get('text'), context: String(form.get('context')).split('\n').filter(Boolean), persist: false }) }); target.classList.remove('empty-state'); target.innerHTML = resultMarkup(result); } catch (error) { target.innerHTML = `<p>${escapeHtml(error.message)}</p>`; }
});

async function loadVisionStatus() {
  try {
    const status = await api('/api/v1/vision/status');
    const label = item => !item.enabled ? '关闭' : item.available ? '可用' : '缺少依赖';
    $('#vision-status').textContent = `二维码 ${label(status.qrcode)} · OCR ${label(status.ocr)} · 色情检测 ${label(status.nsfw)}`;
    if ($('#vision-model-status')) $('#vision-model-status').textContent = `二维码 ${label(status.qrcode)} · OCR ${label(status.ocr)} · 色情 ${label(status.nsfw)}`;
  } catch (error) { $('#vision-status').textContent = error.message; }
}

function visionCard(title, result, body) {
  const tone = result.status === 'ok' ? '' : ' error';
  return `<article class="vision-card${tone}"><h4>${title}</h4>${body}<small>${escapeHtml(result.provider)} · ${escapeHtml(result.status)}${result.error ? ' · ' + escapeHtml(result.error) : ''}</small></article>`;
}

$('#analyze-image').addEventListener('click', async () => {
  const file = $('#image-file').files[0]; if (!file) return toast('请先选择图片');
  const button = $('#analyze-image'); button.disabled = true; button.textContent = '正在执行三路检测…';
  const body = new FormData(); body.append('file', file); body.append('room_id', 'image-test'); body.append('sender_name', '测试成员'); body.append('persist', 'false');
  try {
    const result = await api('/api/v1/images/analyze', { method: 'POST', body });
    const qrBody = `<strong>${result.qrcode.detected ? '检测到' : '未检测到'}</strong><p>${result.qrcode.count} 个二维码；未读取二维码内容</p>`;
    const nsfwBody = `<strong>${result.nsfw.score == null ? '—' : Math.round(result.nsfw.score * 100) + '%'}</strong><p>${result.nsfw.matched ? '达到色情风险阈值' : '未达到色情风险阈值'}</p>`;
    const ocrBody = `<strong>${result.ocr.lines.length} 行</strong><p>平均置信度 ${result.ocr.average_confidence == null ? '—' : Math.round(result.ocr.average_confidence * 100) + '%'}</p>`;
    $('#image-result').className = 'image-result';
    $('#image-result').innerHTML = `${visionCard('二维码存在性', result.qrcode, qrBody)}${visionCard('图片色情检测', result.nsfw, nsfwBody)}${visionCard('海报文字 OCR', result.ocr, ocrBody)}<div class="ocr-text">${result.ocr.text ? escapeHtml(result.ocr.text) : '未提取到文字'}${result.text_analysis ? `\n\n文本审核：${result.text_analysis.matched ? '命中 ' + escapeHtml(result.text_analysis.matches.map(x=>x.topic_name).join('、')) : '未命中'}` : ''}</div>`;
  } catch (error) { $('#image-result').className = 'image-result empty-image'; $('#image-result').textContent = error.message; }
  finally { button.disabled = false; button.textContent = '开始图片审核'; }
});

function escapeHtml(value) { const div = document.createElement('div'); div.textContent = String(value ?? ''); return div.innerHTML; }
function lines(value) { return String(value || '').split('\n').map(item => item.trim()).filter(Boolean); }

async function loadTopics() {
  state.topics = await api('/api/v1/topics');
  $('#topic-grid').innerHTML = state.topics.map(topic => `<article class="topic-card"><div class="topic-top"><span class="topic-icon">${escapeHtml(topic.name.slice(0,1))}</span><span class="severity severity-${topic.severity}">${{low:'低风险',medium:'中风险',high:'高风险',critical:'严重'}[topic.severity]}</span></div><h3>${escapeHtml(topic.name)}</h3><p>${escapeHtml(topic.description || '尚未填写主题说明')}</p><div class="topic-stats"><span><b>${topic.keywords.length}</b> 关键词</span><span><b>${topic.examples.filter(x=>x.polarity==='positive').length}</b> 正例</span><span>阈值 <b>${topic.semantic_threshold}</b></span></div><div class="topic-actions"><button class="button ghost edit-topic" data-id="${topic.id}">编辑</button><button class="button ghost delete-topic" data-id="${topic.id}">删除</button></div></article>`).join('') || '<div class="empty-state"><p>还没有监测主题</p></div>';
}

function openTopic(topic = null) {
  const form = $('#topic-form'); form.reset(); form.elements.id.value = topic?.id || ''; form.elements.name.value = topic?.name || ''; form.elements.severity.value = topic?.severity || 'medium'; form.elements.description.value = topic?.description || ''; form.elements.keywords.value = (topic?.keywords || []).join('\n'); form.elements.positive_examples.value = (topic?.examples || []).filter(x=>x.polarity==='positive').map(x=>x.text).join('\n'); form.elements.negative_examples.value = (topic?.examples || []).filter(x=>x.polarity==='negative').map(x=>x.text).join('\n'); form.elements.semantic_threshold.value = topic?.semantic_threshold ?? .66; form.elements.review_threshold.value = topic?.review_threshold ?? .46; form.elements.context_enabled.checked = topic?.context_enabled ?? true; $('#topic-dialog-title').textContent = topic ? '编辑监测主题' : '新建监测主题'; $('#topic-dialog').showModal();
}
$('#add-topic').addEventListener('click', () => openTopic());
$('#topic-grid').addEventListener('click', async event => { const edit = event.target.closest('.edit-topic'); const del = event.target.closest('.delete-topic'); if (edit) openTopic(state.topics.find(item => item.id === Number(edit.dataset.id))); if (del && confirm('确定删除这个监测主题及其事件吗？')) { await api(`/api/v1/topics/${del.dataset.id}`, { method: 'DELETE' }); toast('监测主题已删除'); await loadTopics(); loadStats(); } });
$('#save-topic').addEventListener('click', async event => { event.preventDefault(); const form = $('#topic-form'); if (!form.reportValidity()) return; const id = form.elements.id.value; const payload = { name: form.elements.name.value, description: form.elements.description.value, enabled: true, severity: form.elements.severity.value, keywords: lines(form.elements.keywords.value), regex_patterns: [], exclude_patterns: [], examples: [...lines(form.elements.positive_examples.value).map(text=>({text,polarity:'positive'})), ...lines(form.elements.negative_examples.value).map(text=>({text,polarity:'negative'}))], semantic_threshold: Number(form.elements.semantic_threshold.value), review_threshold: Number(form.elements.review_threshold.value), context_enabled: form.elements.context_enabled.checked }; try { await api(id ? `/api/v1/topics/${id}` : '/api/v1/topics', { method: id ? 'PUT' : 'POST', body: JSON.stringify(payload) }); $('#topic-dialog').close(); toast(id ? '监测主题已更新' : '监测主题已创建'); await loadTopics(); loadStats(); } catch(error) { toast(error.message); } });

async function loadEvents() {
  const value = await api('/api/v1/events?page_size=50');
  $('#events-body').innerHTML = value.items.map(item => `<tr><td>${new Date(item.created_at).toLocaleString()}<small>${escapeHtml(item.room_id)}</small></td><td class="message-cell">${escapeHtml(item.text)}<small>${escapeHtml(item.sender_name || item.sender_id)}</small></td><td><span class="severity severity-${item.severity}">${escapeHtml(item.topic_name)}</span></td><td><b>${Math.round(item.confidence*100)}%</b><small>${item.stage}</small></td><td class="message-cell">${escapeHtml(item.evidence)}</td><td>${item.feedback ? `<span class="tag">${{correct:'准确',false_positive:'误报',missed:'漏报'}[item.feedback]}</span>` : `<div class="feedback-actions"><button class="button feedback" data-id="${item.id}" data-verdict="correct">准确</button><button class="button feedback" data-id="${item.id}" data-verdict="false_positive">误报</button></div>`}</td></tr>`).join('') || '<tr><td colspan="6">暂无告警事件</td></tr>';
}
$('#refresh-events').addEventListener('click', loadEvents);
$('#events-body').addEventListener('click', async event => { const button = event.target.closest('.feedback'); if (!button) return; await api(`/api/v1/events/${button.dataset.id}/feedback`, { method:'POST', body:JSON.stringify({verdict:button.dataset.verdict,note:''}) }); toast('复核反馈已保存'); loadEvents(); loadStats(); });

async function loadModels() {
  if (!state.catalog) [state.catalog, state.settings] = await Promise.all([api('/api/v1/models/catalog'), api('/api/v1/models/settings')]);
  renderProfiles(); renderModelOptions('embedding'); renderModelOptions('classifier'); fillSettings();
}
function renderProfiles() { $('#model-profiles').innerHTML = (state.catalog.profiles || []).map(item => `<button type="button" class="profile-option ${state.settings.profile === item.id ? 'selected' : ''}" data-profile="${item.id}"><b>${escapeHtml(item.name)}</b><p>${escapeHtml(item.description)}</p><small>${item.gpu_required ? '需要独立显卡' : 'CPU 可运行'} · 建议内存 ${item.recommended_ram_gb} GB+</small></button>`).join(''); }
$('#model-profiles').addEventListener('click', event => { const button = event.target.closest('[data-profile]'); if (!button) return; const profile = state.catalog.profiles.find(item => item.id === button.dataset.profile); state.settings.profile = profile.id; state.settings.classifier_provider = profile.classifier_provider; state.settings.classifier_model = profile.classifier_model; renderProfiles(); renderModelOptions('classifier'); $('#model-form').elements.classifier_model.value = profile.classifier_model; });
function renderModelOptions(kind) { const selectedProvider = state.settings[`${kind}_provider`]; const selectedModel = state.settings[`${kind}_model`]; $(`#${kind}-options`).innerHTML = state.catalog[kind].map((item,index) => `<label class="model-option ${item.provider===selectedProvider && item.model===selectedModel?'selected':''}"><input type="radio" name="${kind}_preset" value="${index}" ${item.provider===selectedProvider && item.model===selectedModel?'checked':''}><span><b>${escapeHtml(item.name)}</b><small>${escapeHtml(item.description)}</small></span></label>`).join(''); }
function fillSettings(){ const form=$('#model-form'); ['embedding_model','embedding_api_base','embedding_api_key','classifier_model','classifier_api_base','classifier_api_key','ocr_provider','nsfw_provider','nsfw_threshold','max_image_megapixels'].forEach(key=>form.elements[key].value=state.settings[key]??''); form.elements.qrcode_enabled.checked=Boolean(state.settings.qrcode_enabled); loadVisionStatus(); }
$('#model-form').addEventListener('change', event => { const radio=event.target.closest('[name$="_preset"]'); if(!radio)return; const kind=radio.name.split('_')[0], item=state.catalog[kind][Number(radio.value)]; state.settings[`${kind}_provider`]=item.provider; state.settings[`${kind}_model`]=item.model; renderModelOptions(kind); $(`#model-form [name="${kind}_model"]`).value=item.model; });
$('#model-form').addEventListener('submit', async event => { event.preventDefault(); const form=event.currentTarget; const payload={...state.settings}; ['embedding_model','embedding_api_base','embedding_api_key','classifier_model','classifier_api_base','classifier_api_key','ocr_provider','nsfw_provider'].forEach(key=>payload[key]=form.elements[key].value); payload.qrcode_enabled=form.elements.qrcode_enabled.checked; payload.nsfw_threshold=Number(form.elements.nsfw_threshold.value); payload.max_image_megapixels=Number(form.elements.max_image_megapixels.value); try { state.settings=await api('/api/v1/models/settings',{method:'PUT',body:JSON.stringify(payload)}); $('#model-status').textContent='设置已保存并立即生效'; toast('模型设置已保存'); renderModelOptions('embedding'); renderModelOptions('classifier'); loadVisionStatus(); } catch(error){toast(error.message)} });
$$('.probe').forEach(button=>button.addEventListener('click',async()=>{button.textContent='测试中…';try{const value=await api('/api/v1/models/probe',{method:'POST',body:JSON.stringify({kind:button.dataset.kind})});toast(`${value.detail} · ${value.latency_ms}ms`)}catch(error){toast(error.message)}finally{button.textContent='测试连接'}}));

async function loadDatasets() {
  [state.datasetCatalog, state.datasets] = await Promise.all([api('/api/v1/datasets/catalog'), api('/api/v1/datasets')]);
  $('#dataset-grid').innerHTML = state.datasetCatalog.map(item => `<article class="dataset-card ${item.recommended ? 'recommended' : ''}"><div class="dataset-meta"><span>${escapeHtml(item.license)}</span><span>·</span><span>${item.gated ? '需在来源页授权' : '公开来源'}</span></div><h3>${escapeHtml(item.name)}</h3><p>${escapeHtml(item.description)}</p><div class="dataset-labels">${item.labels.map(label => `<span>${escapeHtml(label)}</span>`).join('')}</div><div class="dataset-actions">${item.loaded ? `<span class="tag success">已加载 ${item.sample_count} 条</span>` : item.auto_load ? `<button class="button primary load-dataset" data-slug="${escapeHtml(item.slug)}" data-accept="${item.requires_acceptance ? '1' : '0'}">加载数据集</button>` : `<a class="button ghost" href="${escapeHtml(item.homepage)}" target="_blank" rel="noreferrer">查看来源 ↗</a>`}</div></article>`).join('');
}

$('#dataset-grid').addEventListener('click', async event => {
  const button = event.target.closest('.load-dataset'); if (!button) return;
  let accepted = false;
  if (button.dataset.accept === '1') { accepted = confirm('该来源没有标准 SPDX 许可证。请先阅读作者声明，并确认会标明来源与引用论文。是否继续？'); if (!accepted) return; }
  button.disabled = true; button.textContent = '正在下载并导入…';
  try { const result = await api(`/api/v1/datasets/${button.dataset.slug}/load?accepted=${accepted}`, { method: 'POST' }); toast(`已导入 ${result.inserted} 条，跳过 ${result.skipped} 条`); await loadDatasets(); loadStats(); } catch (error) { toast(error.message); button.disabled = false; button.textContent = '加载数据集'; }
});

$('#toggle-import').addEventListener('click', () => { $('#import-box').hidden = !$('#import-box').hidden; });
$('#import-dataset').addEventListener('click', async () => {
  const file = $('#import-file').files[0]; if (!file) return toast('请先选择数据文件');
  const body = new FormData(); body.append('file', file); body.append('options_json', JSON.stringify({ name: $('#import-name').value, slug: $('#import-slug').value, version: 'local', license: 'user-provided', text_column: $('#import-text-column').value, label_column: $('#import-label-column').value, split_column: 'split', default_label: 'normal', label_mapping: {} }));
  try { const result = await api('/api/v1/datasets/import', { method: 'POST', body }); toast(`成功导入 ${result.inserted} 条样本`); $('#import-box').hidden = true; await loadDatasets(); } catch (error) { toast(error.message); }
});

async function loadTraining() {
  [state.datasets, state.runs] = await Promise.all([api('/api/v1/datasets'), api('/api/v1/training/runs')]);
  $('#training-datasets').innerHTML = state.datasets.map(item => `<label class="dataset-check"><input type="checkbox" value="${item.id}" ${item.sample_count ? '' : 'disabled'}><span><b>${escapeHtml(item.name)}</b><small>${item.sample_count} 条 · ${escapeHtml(item.labels.join(', '))}</small></span></label>`).join('') || '<div class="empty-state"><p>请先加载或导入数据集</p></div>';
  $('#training-runs').innerHTML = state.runs.map(run => `<div class="run-card"><div class="run-head"><b>${escapeHtml(run.name)}</b><strong class="status-${run.status}">${escapeHtml(run.status)}</strong></div><p>${run.sample_count || 0} 条样本 · ${run.metrics?.vocabulary_size || 0} 个特征 ${run.error ? '· ' + escapeHtml(run.error) : ''}</p>${run.status === 'completed' ? `<div class="run-metrics"><span>Micro F1 ${run.metrics.micro_f1}</span><span>Macro F1 ${run.metrics.macro_f1}</span><span>Exact ${run.metrics.exact_match}</span><button class="button primary activate-run" data-id="${run.id}">启用模型</button></div>` : ''}</div>`).join('') || '<div class="empty-state"><p>暂无训练任务</p></div>';
  if (state.runs.some(run => ['queued','running'].includes(run.status))) setTimeout(() => $('#view-training').classList.contains('active') && loadTraining(), 1400);
}

$('#start-training').addEventListener('click', async () => {
  const datasetIds = $$('#training-datasets input:checked').map(item => Number(item.value)); if (!datasetIds.length) return toast('请至少选择一个数据集');
  const payload = { name: `群聊审核模型 ${new Date().toLocaleString()}`, algorithm: 'char_ngram_nb', dataset_ids: datasetIds, test_ratio: Number($('#train-test-ratio').value), min_ngram: 1, max_ngram: 3, min_df: Number($('#train-min-df').value), alpha: 1, threshold: Number($('#train-threshold').value), seed: 42 };
  try { await api('/api/v1/training/runs', { method: 'POST', body: JSON.stringify(payload) }); toast('训练任务已启动'); await loadTraining(); } catch (error) { toast(error.message); }
});
$('#refresh-training').addEventListener('click', loadTraining);
$('#training-runs').addEventListener('click', async event => { const button = event.target.closest('.activate-run'); if (!button) return; try { await api(`/api/v1/models/activate/${button.dataset.id}`, { method: 'POST' }); state.catalog = null; toast('训练模型已启用'); await loadTraining(); } catch (error) { toast(error.message); } });

Promise.all([loadStats(), loadTopics(), loadVisionStatus()]).catch(error => toast(error.message));
