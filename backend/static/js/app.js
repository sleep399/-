const App = {
  // Anthropic 设计 Token 色值 (用于 JS 内联样式)
  _colors: {
    error: '#C0453A',
    warning: '#C9943A',
    success: '#6B8F47',
    info: '#6A9BCC',
    accent: '#D97757',
    muted: '#9B9890',
  },
  token: localStorage.getItem('token') || '',
  streamModule: null,
  streamInterval: null,
  wsAlerts: null,
  wsStream: null,
  sseSource: null,
  assistantHistory: [],
  assistantThinking: false,
  assistantRecognition: null,
  assistantVoiceEnabled: false,
  _agentVoice: null,
  _agentSpeaking: false,
  alertTypes: [],
  currentReplayId: null,
  alertTimelineSkip: 0,
  alertTimelineHasMore: false,
  replayEvents: [],
  replayStepIndex: 0,
  replayPlayTimer: null,
  agentOpenCount: 0,
  agentDragMoved: false,
  agentSpeechTimer: null,
  _agentPointerId: null,
  agentMonitorTimer: null,
  agentBriefing: null,
  agentLastBriefKey: '',
  focusedAlert: null,
  currentView: '',
  logSseSource: null,
  LOG_CATEGORY_LABELS: {
    lpr: '车牌识别',
    police_gesture: '交警手势',
    owner_gesture: '车主手势',
    alert: '告警',
    user: '用户操作',
    system: '系统运行',
    agent: '智能体决策',
  },
  LOG_CATEGORY_COLORS: {
    lpr: '#6A9BCC',
    police_gesture: '#C9943A',
    owner_gesture: '#6B8F47',
    alert: '#C0453A',
    user: '#8B7EC8',
    system: '#64748b',
    agent: '#0891b2',
  },

  /** 构建助手请求：仅携带用户显式选定的告警，不再静默绑定「最近一条」 */
  buildAssistantPayload(question) {
    const body = { question };
    const alertId = this.getExplicitAlertId();
    if (alertId) body.alert_id = alertId;
    return body;
  },

  /** 当前显式选定的告警 ID（仅以 focusedAlert 为准；回放页打开不等于助手上下文） */
  getExplicitAlertId() {
    return this.focusedAlert?.id ?? null;
  },

  /** 用户显式选定某条告警作为对话上下文 */
  setFocusedAlert(alert) {
    if (!alert || !alert.id) return;
    this.focusedAlert = {
      id: alert.id,
      title: alert.title || '系统提醒',
      level: alert.level || 'info',
    };
    this.updateAssistantContextUI();
  },

  /** 取消当前选定的告警上下文（不影响告警回放面板本身） */
  clearFocusedAlert() {
    this.focusedAlert = null;
    this.updateAssistantContextUI();
  },

  updateAssistantContextUI() {
    const bar = document.getElementById('assistant-context-bar');
    const titleEl = document.getElementById('assistant-context-title');
    const subtitle = document.getElementById('agent-subtitle');

    if (this.focusedAlert && bar && titleEl) {
      bar.classList.remove('hidden');
      titleEl.textContent = this.focusedAlert.title;
      if (subtitle) subtitle.textContent = `正在讨论：${this.focusedAlert.title}`;
      return;
    }

    if (bar) bar.classList.add('hidden');
    if (titleEl) titleEl.textContent = '';
    if (subtitle && !this.agentOpenCount) {
      subtitle.textContent = '感知 · 决策 · 告警推送';
    } else if (subtitle && this.agentOpenCount > 0) {
      subtitle.textContent = `发现 ${this.agentOpenCount} 条未处理告警（请先选定一条再提问）`;
    }
  },

  /**
   * 快捷提问（根因/建议/升级/影响）
   * requireAlert=true 时若无选定告警，由后端返回「您指的是哪条告警？」
   */
  askAboutAlert(question) {
    this.askAssistant(question);
  },

  /** 用户能听懂的简短告警话术 */
  alertToUserSpeech(alert) {
    const title = alert.title || '系统提醒';
    const summary = alert.summary || '';
    if (summary && summary.length < 60) return summary;
    return title;
  },

  init() {
    this.bindTabs();
    this.bindNav();
    this.bindFileInputs();
    this.initSelectChevrons();
    this.initAssistant();
    if (this.token) this.showMain();
    else document.getElementById('login-page').classList.add('active');
  },

  headers() {
    const h = { 'Content-Type': 'application/json' };
    if (this.token) h['Authorization'] = `Bearer ${this.token}`;
    return h;
  },

  async api(path, opts = {}) {
    const res = await fetch(path, { ...opts, headers: { ...this.headers(), ...opts.headers } });
    if (!res.ok) {
      let detail = res.statusText;
      try { const err = await res.json(); detail = err.detail || detail; } catch (e) {}
      throw new Error(detail || '请求失败');
    }
    return res.json();
  },

  bindTabs() {
    document.querySelectorAll('.tab').forEach(tab => {
      tab.onclick = () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
      };
    });
  },

  bindNav() {
    document.querySelectorAll('.nav-item[data-view]').forEach(item => {
      item.onclick = (e) => {
        e.preventDefault();
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
        item.classList.add('active');
        document.getElementById('view-' + item.dataset.view).classList.add('active');
        this.onViewChange(item.dataset.view);
      };
    });
  },

  bindFileInputs() {
    document.getElementById('lpr-file').onchange = (e) => this.uploadFile('lpr', e.target.files[0]);
    document.getElementById('police-file').onchange = (e) => this.uploadFile('police', e.target.files[0]);
    document.getElementById('owner-file').onchange = (e) => this.uploadFile('owner', e.target.files[0]);
  },

  onViewChange(view) {
    const prev = this.currentView;
    this.currentView = view;
    if (prev === 'logs' && view !== 'logs') this.disconnectLogStream();
    if (view === 'dashboard') this.loadDashboard();
    if (view === 'lpr') { this.loadLprHistory(); }
    if (view === 'police') { this.loadPoliceGestures(); this.loadPoliceHistory(); }
    if (view === 'owner') { this.loadOwnerGestures(); this.loadVehicleState(); }
    if (view === 'alerts') {
      this.connectAlertWs();
      this.connectSSE();
      this.loadAlerts();
      this.loadAlertTypes();
      this.loadAlertAnalytics();
      this.loadAgentActivity();
      this.loadAlertNotifications();
      this.loadAlertConfig();
    }
    if (view === 'logs') {
      this.resetLogFilters(false);
      this.syncLogDatetimeState();
      this.loadLogs();
      this.connectLogStream();
    }
  },

  // ── 认证 ──
  async login() {
    const username = document.getElementById('login-user').value;
    const password = document.getElementById('login-pass').value;
    try {
      const data = await this.api('/api/auth/login', { method: 'POST', body: JSON.stringify({ username, password }) });
      this.token = data.access_token;
      localStorage.setItem('token', this.token);
      this.showMain();
    } catch (e) { alert(e.message); }
  },

  async sendCode() {
    const target = document.getElementById('code-target').value;
    try {
      const data = await this.api('/api/auth/send-code', { method: 'POST', body: JSON.stringify({ target, target_type: target.includes('@') ? 'email' : 'phone' }) });
      alert('验证码: ' + data.code + ' (演示模式直接显示)');
    } catch (e) { alert(e.message); }
  },

  async loginCode() {
    const target = document.getElementById('code-target').value;
    const code = document.getElementById('code-input').value;
    try {
      const data = await this.api('/api/auth/login-code', { method: 'POST', body: JSON.stringify({ target, code, target_type: target.includes('@') ? 'email' : 'phone' }) });
      this.token = data.access_token;
      localStorage.setItem('token', this.token);
      this.showMain();
    } catch (e) { alert(e.message); }
  },

  async wechatLogin() {
    try {
      const session = await this.api('/api/auth/wechat/qrcode', { method: 'POST' });
      const qrBox = document.getElementById('qr-box');
      qrBox.innerHTML = `微信扫码登录<br><small>${session.session_id.slice(0, 8)}</small><div class="qr-placeholder">二维码已生成，当前为演示模式</div>`;
      const poll = setInterval(async () => {
        const res = await fetch(session.poll_url);
        const data = await res.json();
        if (data.status === 'confirmed') {
          clearInterval(poll);
          this.token = data.access_token;
          localStorage.setItem('token', this.token);
          this.showMain();
        }
      }, 1500);
    } catch (e) { alert(e.message); }
  },

  skipLogin() { this.showMain(); },

  showMain() {
    document.getElementById('login-page').classList.remove('active');
    document.getElementById('main-page').classList.add('active');
    this.loadDashboard();
    this.connectAlertWs();
    this.connectSSE();
    this.refreshAgentStats();
    this.startAgentMonitorLoop();
    if (this.token) {
      this.api('/api/auth/me').then(u => {
        document.getElementById('user-info').textContent = u.username;
      }).catch(() => {});
    }
  },

  logout() {
    this.token = '';
    localStorage.removeItem('token');
    this.stopStream();
    this.disconnectSSE();
    this.stopAgentMonitorLoop();
    location.reload();
  },

  // ── WebSocket & SSE ──
  connectAlertWs() {
    if (this.wsAlerts && this.wsAlerts.readyState === WebSocket.OPEN) return;
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    this.wsAlerts = new WebSocket(`${proto}://${location.host}/ws/alerts`);
    this.wsAlerts.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.type === 'alert') {
        this.showToast(data);
        this.prependLiveAlert(data);
        this.onAgentAlert(data);
      }
    };
      this.wsAlerts.onopen = () => { var el = document.getElementById('stat-ws-conn'); if (el) el.textContent = '1'; };
      this.wsAlerts.onclose = () => {
        this.wsAlerts = null;
        var _el = document.getElementById('stat-ws-conn'); if (_el) _el.textContent = '0';
      setTimeout(() => { if (document.getElementById('main-page').classList.contains('active')) this.connectAlertWs(); }, 3000);
    };
  },

  connectSSE() {
    if (this.sseSource) return;
    this.sseSource = new EventSource('/api/monitor/stream');
    this.sseSource.onopen = () => {
      document.getElementById('stat-sse-conn') && (document.getElementById('stat-sse-conn').textContent = '1');
    };
    this.sseSource.addEventListener('connected', (e) => {
      console.log('SSE connected:', JSON.parse(e.data));
    });
    this.sseSource.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === 'alert') {
          this.showToast(data);
          this.prependLiveAlert(data);
          this.onAgentAlert(data);
        }
      } catch (err) {}
    };
    this.sseSource.onerror = () => {
      document.getElementById('stat-sse-conn') && (document.getElementById('stat-sse-conn').textContent = '0');
    };
  },

  disconnectSSE() {
    if (this.sseSource) { this.sseSource.close(); this.sseSource = null; }
    var ssc2 = document.getElementById('stat-sse-conn'); if (ssc2) ssc2.textContent = '0';
  },

  showToast(alert) {
    const el = document.createElement('div');
    el.className = 'toast ' + (alert.level || '');
    el.innerHTML = `<strong>[${(alert.level || '').toUpperCase()}] ${alert.title}</strong><br><small>${alert.summary || ''}</small>`;
    document.getElementById('toast-container').appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; el.style.transition = 'opacity 0.3s'; setTimeout(() => el.remove(), 300); }, 5000);
  },

  prependLiveAlert(alert) {
    const container = document.getElementById('live-alerts');
    if (!container) return;
    const div = document.createElement('div');
    div.className = 'alert-item ' + (alert.level || '');
    const time = alert.created_at ? new Date(alert.created_at).toLocaleString() : new Date().toLocaleString();
    div.innerHTML = `
      <div class="alert-title">${this.escHtml(alert.title)}</div>
      <div class="alert-summary">${this.escHtml(alert.summary || '')}</div>
      <div class="alert-meta">${time} · ${alert.event_type || ''} · ${(alert.channels || '').split(',').join(', ')}</div>
      ${alert.suggestion ? `<div class="alert-suggestion">💡 ${this.escHtml(alert.suggestion)}</div>` : ''}
    `;
    container.prepend(div);
  },

  // ── 仪表盘 ──
  async loadDashboard() {
    try {
      const [lpr, police, owner, stats, connResp, logResp] = await Promise.all([
        this.api('/api/lpr/history?limit=100'),
        this.api('/api/police-gesture/history?limit=100'),
        this.api('/api/owner-gesture/history?limit=100'),
        this.api('/api/monitor/alerts/stats'),
        this.api('/api/monitor/connections'),
        this.api('/api/monitor/logs/stats?hours=24'),
      ]);
      document.getElementById('stat-lpr').textContent = lpr.length;
      document.getElementById('stat-police').textContent = police.length;
      document.getElementById('stat-owner').textContent = owner.length;
      document.getElementById('stat-alerts').textContent = stats.total;
      document.getElementById('stat-logs-total').textContent = logResp.total;
      document.getElementById('stat-ws-conn').textContent = connResp.websocket_clients;
      document.getElementById('stat-sse-conn').textContent = connResp.sse_clients;

      // Token
      if (stats.token_usage) {
        document.getElementById('stat-token-used').textContent = stats.token_usage.used + '/' + stats.token_usage.limit;
      }

      // Dashboard alerts
      const el = document.getElementById('dashboard-alerts');
      el.innerHTML = stats.recent.slice(0, 5).map(a =>
        `<div class="alert-item ${a.level}">
          <div class="alert-title">${this.escHtml(a.title)}</div>
          <div>${this.escHtml(a.summary || '')}</div>
          <div class="alert-meta">${new Date(a.created_at).toLocaleString()}</div>
        </div>`
      ).join('') || '<p style="color:var(--text-muted)">暂无告警</p>';

      // Dashboard alert analytics mini chart
      this.renderAlertAnalytics(stats, 'dashboard-alert-analytics', true);

    } catch (e) { console.error('Dashboard load error:', e); }
  },

  // ── 识别模块 ──
  async uploadFile(module, file) {
    if (!file) return;
    const endpoints = { lpr: '/api/lpr/recognize', police: '/api/police-gesture/recognize', owner: '/api/owner-gesture/recognize' };
    const form = new FormData(); form.append('file', file);
    const headers = {}; if (this.token) headers['Authorization'] = `Bearer ${this.token}`;
    const previewMap = { lpr: 'lpr-preview', police: 'police-preview', owner: 'owner-preview' };
    const resultMap = { lpr: 'lpr-results', police: 'police-result', owner: 'owner-result' };
    const preview = document.getElementById(previewMap[module]);
    const resultBox = document.getElementById(resultMap[module]);
    if (preview && file.type.startsWith('image/')) preview.src = URL.createObjectURL(file);
    if (resultBox) resultBox.innerHTML = '正在识别，请稍候...';
    try {
      const res = await fetch(endpoints[module], { method: 'POST', body: form, headers });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || '识别失败');
      this.renderResult(module, data);
      if (module === 'owner' && data.action) this.loadVehicleState();
    } catch (e) {
      if (resultBox) resultBox.innerHTML = `识别失败：${e.message}`;
      alert(e.message);
    }
  },

  renderResult(module, data) {
    if (module === 'lpr') {
      document.getElementById('lpr-preview').src = 'data:image/jpeg;base64,' + data.annotated_image;
      document.getElementById('lpr-results').innerHTML = `
        <div class="result-banner ${data.success ? 'success' : 'danger'}">
          <div class="result-title">${data.success ? '识别成功' : '未识别到有效车牌'}</div>
          <div class="result-subtitle">共检测到 ${data.plate_count} 个车牌</div>
        </div>`;
      const colorMap = { '蓝牌': 'plate-blue', '绿牌': 'plate-green', '黄牌': 'plate-yellow', '白牌': 'plate-white', '黑牌': 'plate-black' };
      document.getElementById('lpr-plates').innerHTML = data.plates.map(p =>
        `<div class="plate-item"><span class="number">${this.escHtml(p.plate_number)}</span><span class="color ${colorMap[p.plate_color] || ''}">${this.escHtml(p.plate_color)} (${(p.confidence*100).toFixed(0)}%)</span></div>`
      ).join('') || '<p>未检测到车牌</p>';
      this.loadLprHistory();
    } else if (module === 'police') {
      document.getElementById('police-preview').src = 'data:image/jpeg;base64,' + data.annotated_image;
      document.getElementById('police-result').innerHTML = `${this.escHtml(data.gesture_cn)}<br><small>置信度 ${(data.confidence*100).toFixed(0)}%</small>`;
      this.loadPoliceHistory();
    } else if (module === 'owner') {
      document.getElementById('owner-preview').src = 'data:image/jpeg;base64,' + data.annotated_image;
      document.getElementById('owner-result').innerHTML = `${this.escHtml(data.gesture_cn)}${data.action ? '<br><small>→ ' + this.escHtml(data.action) + '</small>' : ''}`;
    }
  },

  async loadLprHistory() {
    try {
      const data = await this.api('/api/lpr/history?limit=10');
      document.getElementById('lpr-history').innerHTML = data.map(r =>
        `<div class="history-item"><span>#${r.id} · ${r.plate_count}个车牌</span><span>${new Date(r.created_at).toLocaleString()}</span></div>`
      ).join('') || '<p>暂无记录</p>';
    } catch (e) {}
  },

  async loadPoliceGestures() {
    try {
      const data = await this.api('/api/police-gesture/gestures');
      document.getElementById('police-gesture-list').innerHTML = data.map(g =>
        `<span class="gesture-tag">${this.escHtml(g.cn)}</span>`
      ).join('');
    } catch (e) {}
  },

  async loadPoliceHistory() {
    try {
      const data = await this.api('/api/police-gesture/history?limit=10');
      document.getElementById('police-history').innerHTML = data.map(r =>
        `<div class="history-item"><span>${this.escHtml(r.gesture_cn)}</span><span>${(r.confidence*100).toFixed(0)}%</span></div>`
      ).join('');
    } catch (e) {}
  },

  async loadOwnerGestures() {
    try {
      const data = await this.api('/api/owner-gesture/gestures');
      document.getElementById('owner-gestures').innerHTML = data.map(g =>
        `<span class="gesture-tag">${this.escHtml(g.cn)} → ${this.escHtml(g.action || '-')}</span>`
      ).join('');
    } catch (e) {}
  },

  async loadVehicleState() {
    try {
      const s = await this.api('/api/owner-gesture/vehicle-state');
      document.getElementById('v-awake').textContent = s.is_awake ? '已唤醒' : '休眠';
      document.getElementById('v-page').textContent = s.current_page;
      document.getElementById('v-volume').value = s.volume;
      document.getElementById('v-volume-val').textContent = s.volume;
      document.getElementById('v-temp').value = s.temperature;
      document.getElementById('v-temp-val').textContent = s.temperature;
      document.getElementById('v-phone').textContent = s.phone_status === 'in_call' ? '通话中' : '空闲';
    } catch (e) {}
  },

  async updateVehicle() {
    const data = {
      volume: +document.getElementById('v-volume').value,
      temperature: +document.getElementById('v-temp').value,
      phone_status: document.getElementById('v-phone').textContent === '通话中' ? 'in_call' : 'idle',
      current_page: document.getElementById('v-page').textContent,
      is_awake: document.getElementById('v-awake').textContent === '已唤醒' ? 1 : 0,
    };
    document.getElementById('v-volume-val').textContent = data.volume;
    document.getElementById('v-temp-val').textContent = data.temperature;
    try { await this.api('/api/owner-gesture/vehicle-state', { method: 'PUT', body: JSON.stringify(data) }); } catch (e) {}
  },

  setPhone(status) {
    document.getElementById('v-phone').textContent = status === 'in_call' ? '通话中' : '空闲';
    this.updateVehicle();
  },

  async startStream(module) {
    this.stopStream();
    this.streamModule = module;
    const video = document.getElementById(module + '-video');
    const canvas = document.getElementById(module + '-canvas');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
      video.hidden = false;
      canvas.hidden = false;
      const ctx = canvas.getContext('2d');
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      this.wsStream = new WebSocket(`${proto}://${location.host}/ws/stream/${module}`);
      this.wsStream.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === 'result') this.renderResult(module, msg.data);
      };
      this.streamInterval = setInterval(() => {
        if (video.readyState >= 2 && this.wsStream && this.wsStream.readyState === WebSocket.OPEN) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          ctx.drawImage(video, 0, 0);
          const dataUrl = canvas.toDataURL('image/jpeg', 0.7);
          this.wsStream.send(JSON.stringify({ type: 'frame', data: dataUrl.split(',')[1] }));
        }
      }, 500);
    } catch (e) { alert('无法访问摄像头: ' + e.message); }
  },

  stopStream() {
    if (this.streamInterval) { clearInterval(this.streamInterval); this.streamInterval = null; }
    if (this.wsStream) { this.wsStream.close(); this.wsStream = null; }
    if (this.streamModule) {
      const video = document.getElementById(this.streamModule + '-video');
      if (video.srcObject) { video.srcObject.getTracks().forEach(t => t.stop()); video.srcObject = null; }
      video.hidden = true;
      this.streamModule = null;
    }
  },

  // ── 告警中心 ──
  _alertFilterParams() {
    const level = (document.getElementById('alert-filter-level') && document.getElementById('alert-filter-level').value) || '';
    const eventType = (document.getElementById('alert-filter-type') && document.getElementById('alert-filter-type').value) || '';
    const status = (document.getElementById('alert-filter-status') && document.getElementById('alert-filter-status').value) || '';
    const startEl = document.getElementById('alert-filter-start');
    const endEl = document.getElementById('alert-filter-end');
    const start = startEl && startEl.value ? startEl.value + 'T00:00:00' : '';
    const end = endEl && endEl.value ? endEl.value + 'T23:59:59' : '';
    let qs = '';
    if (level) qs += '&level=' + encodeURIComponent(level);
    if (eventType) qs += '&event_type=' + encodeURIComponent(eventType);
    if (status) qs += '&status=' + encodeURIComponent(status);
    if (start) qs += '&start=' + encodeURIComponent(start);
    if (end) qs += '&end=' + encodeURIComponent(end);
    return qs;
  },

  _alertFilterSummary() {
    const parts = [];
    const levelEl = document.getElementById('alert-filter-level');
    const typeEl = document.getElementById('alert-filter-type');
    const statusEl = document.getElementById('alert-filter-status');
    const levelLabels = { info: '提示', warning: '警告', critical: '严重' };
    if (levelEl && levelEl.value) parts.push(`级别：${levelLabels[levelEl.value] || levelEl.value}`);
    if (typeEl && typeEl.value) {
      const text = typeEl.selectedOptions[0]?.text || typeEl.value;
      parts.push(`类型：${text.replace(/\s*\([^)]*\)\s*$/, '')}`);
    }
    if (statusEl && statusEl.value) {
      parts.push(`状态：${statusEl.value === 'open' ? '未处理' : '已处理'}`);
    }
    return parts.length ? `筛选：${parts.join(' · ')} · ` : '';
  },

  async onAlertFilterChange() {
    await this.resetAlertTimeline();
    const panel = document.getElementById('alert-timeline');
    if (panel) panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  },

  async refreshAlerts() {
    const btn = document.getElementById('alert-refresh-btn');
    if (btn) {
      btn.disabled = true;
      btn.textContent = '刷新中…';
    }
    try {
      await this.resetAlertTimeline();
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = '刷新';
      }
    }
  },

  renderTimelineItem(a) {
    const levelIcon = { info: 'ℹ️', warning: '⚠️', critical: '🚨' };
    const statusBadge = a.status === 'resolved'
      ? '<span class="badge resolved">已处理</span>'
      : '<span class="badge open">未处理</span>';
    return `
      <div class="timeline-item ${a.level}">
        <div class="timeline-content">
          <div class="timeline-header">
            <span>${levelIcon[a.level] || '•'} <strong>${this.escHtml(a.title)}</strong> ${statusBadge}</span>
            <div class="timeline-actions">
              <button class="btn small" onclick="App.viewReplay(${a.id})" title="事件回放">▶ 回放</button>
              <button class="btn small" onclick="App.showCauseAnalysis(${a.id})" title="根因分析">🔍 根因</button>
              ${a.status !== 'resolved' ? `<button class="btn small primary" onclick="App.resolveAlert(${a.id})">✓ 处理</button>` : ''}
            </div>
          </div>
          <p class="timeline-summary">${this.escHtml(a.summary || '')}</p>
          <div class="timeline-meta">
            <span>🕐 ${new Date(a.created_at).toLocaleString()}</span>
            <span>📌 ${this.escHtml(a.event_type_cn || a.event_type)}</span>
            <span>📡 ${a.channels || a.channels_sent || 'web'}</span>
          </div>
          ${a.root_cause ? `<div class="timeline-cause">🔍 根因：${this.escHtml(a.root_cause)}</div>` : ''}
          ${a.suggestion ? `<div class="timeline-suggestion">💡 建议：${this.escHtml(a.suggestion)}</div>` : ''}
        </div>
      </div>`;
  },

  renderTimelineGroups(groups, append = false) {
    const el = document.getElementById('alert-timeline');
    if (!el) return;
    const html = (groups || []).map(g => `
      <div class="timeline-date-group">
        <div class="timeline-date-header">${g.date}</div>
        ${g.items.map(a => this.renderTimelineItem(a)).join('')}
      </div>
    `).join('');
    if (append) {
      el.innerHTML += html;
    } else {
      el.innerHTML = html || '<p style="color:var(--text-muted);padding:1rem;">暂无告警记录</p>';
    }
  },

  async loadAlerts(append = false) {
    try {
      if (!append) this.alertTimelineSkip = 0;
      const qs = this._alertFilterParams();
      const skip = this.alertTimelineSkip;

      const [stats, timeline] = await Promise.all([
        this.api('/api/monitor/alerts/stats'),
        this.api(`/api/monitor/alerts/timeline?limit=30&skip=${skip}${qs}`),
      ]);

      const statsEl = document.getElementById('alert-stats');
      statsEl.innerHTML = `
        <div class="stat-card"><div class="stat-num">${stats.total}</div><div class="stat-label">总计</div></div>
        <div class="stat-card"><div class="stat-num">${stats.open || 0}</div><div class="stat-label">未处理</div></div>
        <div class="stat-card"><div class="stat-num">${stats.open_critical || 0}</div><div class="stat-label">严重未处理</div></div>
        <div class="stat-card"><div class="stat-num">${stats.today_count || 0}</div><div class="stat-label">今日新增</div></div>
        <div class="stat-card"><div class="stat-num">${stats.resolution_rate || 0}%</div><div class="stat-label">处理率</div></div>
        <div class="stat-card"><div class="stat-num small">${stats.mttr_minutes != null ? stats.mttr_minutes + '分' : '-'}</div><div class="stat-label">平均处理时长</div></div>
      `;

      this.renderDistribution(stats);
      this.renderTimelineGroups(timeline.groups, append);

      this.alertTimelineHasMore = timeline.has_more;
      this.alertTimelineSkip = skip + (timeline.groups || []).reduce((n, g) => n + g.items.length, 0);

      const infoEl = document.getElementById('alert-timeline-info');
      const moreBtn = document.getElementById('alert-load-more');
      if (infoEl) {
        infoEl.textContent = `${this._alertFilterSummary()}共 ${timeline.total} 条，已加载 ${Math.min(this.alertTimelineSkip, timeline.total)} 条`;
      }
      if (moreBtn) moreBtn.style.display = timeline.has_more ? 'inline-block' : 'none';

    } catch (e) {
      console.error('Load alerts error:', e);
      this.showToast({ level: 'critical', title: '加载告警失败', summary: e.message || '请检查网络或稍后重试' });
    }
  },

  async resetAlertTimeline() {
    this.alertTimelineSkip = 0;
    await this.loadAlerts(false);
  },

  loadMoreAlerts() {
    if (this.alertTimelineHasMore) this.loadAlerts(true);
  },

  async loadAlertAnalytics() {
    try {
      const daysEl = document.getElementById('alert-analytics-days');
      const days = daysEl ? daysEl.value : 7;
      const data = await this.api(`/api/monitor/alerts/analytics?days=${days}`);
      this.renderAlertAnalytics(data, 'alert-analytics', false);
    } catch (e) { console.error('Load analytics error:', e); }
  },

  renderAlertAnalytics(data, containerId, compact) {
    const el = document.getElementById(containerId);
    if (!el || !data) return;

    const byLevel = data.by_level || {};
    const maxLevel = Math.max(...Object.values(byLevel), 1);
    const levelColors = { critical: '#C0453A', warning: '#C9943A', info: '#6A9BCC' };

    let levelHtml = '<div class="analytics-section"><h4>级别分布</h4><div class="dist-bars">';
    for (const [level, count] of Object.entries(byLevel)) {
      const pct = Math.round(count / maxLevel * 100);
      levelHtml += `<div class="dist-row"><span class="dist-label">${level}</span>
        <div class="dist-bar-track"><div class="dist-bar-fill" style="width:${pct}%;background:${levelColors[level] || '#9B9890'};"></div></div>
        <span class="dist-count">${count}</span></div>`;
    }
    levelHtml += '</div></div>';

    const ranked = data.by_type_ranked || [];
    const maxType = Math.max(...ranked.map(t => t.count), 1);
    let typeHtml = '<div class="analytics-section"><h4>类型 TOP</h4><div class="dist-bars">';
    for (const t of ranked.slice(0, compact ? 5 : 8)) {
      const pct = Math.round(t.count / maxType * 100);
      typeHtml += `<div class="dist-row"><span class="dist-label dist-label-wide" title="${this.escHtml(t.name)}">${this.escHtml(t.name)}</span>
        <div class="dist-bar-track"><div class="dist-bar-fill" style="width:${pct}%;background:var(--color-accent-orange);"></div></div>
        <span class="dist-count">${t.count}</span></div>`;
    }
    typeHtml += '</div></div>';

    const hourly = data.hourly_distribution || [];
    const maxHour = Math.max(...hourly.map(h => h.count), 1);
    let hourHtml = '<div class="analytics-section"><h4>24 小时分布</h4><div class="hourly-chart">';
    for (const h of hourly) {
      const hPct = Math.max(4, Math.round(h.count / maxHour * 100));
      hourHtml += `<div class="hourly-bar" title="${h.label}: ${h.count}条">
        <div class="hourly-fill" style="height:${hPct}%;"></div>
        <span class="hourly-label">${h.hour % 6 === 0 ? h.label : ''}</span>
      </div>`;
    }
    hourHtml += '</div></div>';

    const trends = data.date_trend || [];
    const maxTrend = Math.max(...trends.map(t => t.count), 1);
    let trendHtml = '<div class="analytics-section"><h4>日期趋势</h4><div class="dist-bars">';
    for (const t of trends.slice(compact ? -7 : -14)) {
      const pct = Math.round(t.count / maxTrend * 100);
      trendHtml += `<div class="dist-row"><span class="dist-label">${t.date.slice(5)}</span>
        <div class="dist-bar-track"><div class="dist-bar-fill" style="width:${pct}%;background:var(--color-info);"></div></div>
        <span class="dist-count">${t.count}</span></div>`;
    }
    trendHtml += '</div></div>';

    const kpiHtml = compact ? '' : `
      <div class="analytics-kpi">
        <div class="analytics-kpi-item"><span class="kpi-num">${data.total || 0}</span><span class="kpi-label">区间总数</span></div>
        <div class="analytics-kpi-item"><span class="kpi-num">${data.resolution_rate || 0}%</span><span class="kpi-label">处理率</span></div>
        <div class="analytics-kpi-item"><span class="kpi-num">${data.mttr_minutes != null ? data.mttr_minutes + '分' : '-'}</span><span class="kpi-label">MTTR</span></div>
        <div class="analytics-kpi-item"><span class="kpi-num">${data.open || 0}</span><span class="kpi-label">未处理</span></div>
      </div>`;

    el.innerHTML = kpiHtml + `<div class="analytics-grid${compact ? ' compact' : ''}">` + levelHtml + typeHtml + hourHtml + trendHtml + '</div>';
  },

  renderCauseAnalysisHtml(cause) {
    if (!cause) return '<p style="color:var(--text-muted);">暂无根因分析数据</p>';
    const chain = (cause.cause_chain || []).map(c => `
      <div class="cause-chain-item cause-${c.type}">
        <div class="cause-chain-step">${c.step}</div>
        <div class="cause-chain-body">
          <strong>${this.escHtml(c.title)}</strong>
          <p>${this.escHtml(c.description || '')}</p>
          ${c.timestamp ? `<span class="cause-time">${new Date(c.timestamp).toLocaleString()}</span>` : ''}
        </div>
      </div>
    `).join('');

    const factors = (cause.contributing_factors || []).map(f =>
      `<li>${this.escHtml(f)}</li>`
    ).join('');

    return `
      <div class="cause-analysis-panel">
        <h5>🔍 根因分析</h5>
        <div class="cause-primary">${this.escHtml(cause.primary_cause)}</div>
        ${cause.impact ? `<div class="cause-impact">⚡ 影响评估：${this.escHtml(cause.impact)}</div>` : ''}
        ${factors ? `<div class="cause-factors"><strong>关联因素</strong><ul>${factors}</ul></div>` : ''}
        ${cause.suggestion ? `<div class="cause-suggestion">💡 ${this.escHtml(cause.suggestion)}</div>` : ''}
        ${chain ? `<div class="cause-chain"><strong>因果链</strong>${chain}</div>` : ''}
        <button class="btn small" onclick="App.askCauseDeep()">🤖 AI 深度分析</button>
      </div>`;
  },

  async showCauseAnalysis(alertId) {
    await this.viewReplay(alertId);
    const panel = document.querySelector('.cause-analysis-panel');
    if (panel) panel.scrollIntoView({ behavior: 'smooth' });
  },

  askCauseDeep() {
    if (!this.getExplicitAlertId()) {
      this.addAssistantMessage('user', '请对这个告警进行深度根因分析，说明因果链、影响范围和推荐处置步骤。');
      this.addAssistantMessage('assistant',
        '您指的是哪条告警？请先在告警回放页打开某条告警，再点「AI 深度分析」。');
      return;
    }
    this.askAboutAlert('请对这个告警进行深度根因分析，说明因果链、影响范围和推荐处置步骤。');
  },

  renderDistribution(stats) {
    const el = document.getElementById('alert-distribution');
    if (!el) return;
    const byLevel = stats.by_level || {};
    const maxVal = Math.max(...Object.values(byLevel), 1);

    let html = '<div class="dist-bars">';
    for (const [level, count] of Object.entries(byLevel)) {
      const pct = Math.round(count / maxVal * 100);
      const color = { critical: '#C0453A', warning: '#C9943A', info: '#6A9BCC' }[level] || '#9B9890';
      html += `
        <div class="dist-row">
          <span class="dist-label">${level}</span>
          <div class="dist-bar-track"><div class="dist-bar-fill" style="width:${pct}%;background:${color};"></div></div>
          <span class="dist-count">${count}</span>
        </div>`;
    }
    html += '</div>';

    // Date trend
    const trends = stats.date_trend || [];
    if (trends.length > 0) {
      html += '<h4 style="margin-top:1rem;margin-bottom:.5rem;">每日趋势</h4><div class="dist-bars">';
      const maxTrend = Math.max(...trends.map(t => t.count), 1);
      for (const t of trends.slice(-14)) {
        const pct = Math.round(t.count / maxTrend * 100);
        html += `
          <div class="dist-row">
            <span class="dist-label">${t.date.slice(5)}</span>
            <div class="dist-bar-track"><div class="dist-bar-fill" style="width:${pct}%;background:var(--color-accent-orange);"></div></div>
            <span class="dist-count">${t.count}</span>
          </div>`;
      }
      html += '</div>';
    }
    el.innerHTML = html;
  },

  async viewReplay(alertId) {
    this.currentReplayId = alertId;
    this.replayStopPlay();
    try {
      const data = await this.api(`/api/monitor/alerts/${alertId}/replay`);
      const panel = document.getElementById('replay-panel');
      panel.style.display = 'block';
      const a = data.alert;
      this.setFocusedAlert(a);
      const causeHtml = this.renderCauseAnalysisHtml(data.cause_analysis);

      panel.querySelector('#replay-content').innerHTML = `
        <div class="replay-alert ${a.level}">
          <h4>${this.escHtml(a.title)}</h4>
          <table class="replay-table">
            <tr><td class="replay-key">级别</td><td><span class="badge ${a.level}">${a.level}</span></td></tr>
            <tr><td class="replay-key">类型</td><td>${this.escHtml(a.event_type_cn || a.event_type)}</td></tr>
            <tr><td class="replay-key">时间</td><td>${new Date(a.created_at).toLocaleString()}</td></tr>
            <tr><td class="replay-key">状态</td><td>${a.status}</td></tr>
            <tr><td class="replay-key">推送渠道</td><td>${a.channels || 'web'}</td></tr>
          </table>
          <h5>📋 摘要</h5><p>${this.escHtml(a.summary || '无')}</p>
          ${causeHtml}
          <h5>📦 事件详情</h5><pre class="replay-detail">${JSON.stringify(a.detail || {}, null, 2)}</pre>
          ${a.resolution_note ? `<h5>✅ 处理说明</h5><p>${this.escHtml(a.resolution_note)}</p>` : ''}
        </div>
        ${(data.related_records && data.related_records.length) ? `
        <h5 style="margin-top:1rem;">🖼️ 关联识别记录 (${data.related_records.length}条)</h5>
        <div class="replay-records">${data.related_records.map(r =>
          `<div class="replay-record">
            <span>${r.type} #${r.id} · ${r.created_at ? new Date(r.created_at).toLocaleString() : ''}</span>
            ${r.gesture_cn ? `<span>${this.escHtml(r.gesture_cn)} (${Math.round((r.confidence || 0) * 100)}%)</span>` : ''}
            ${r.annotated_image ? `<img src="${r.annotated_image}" alt="识别结果" style="max-width:100%;margin-top:0.5rem;border-radius:8px;">` : ''}
          </div>`
        ).join('')}</div>` : ''}
        <h5 style="margin-top:1rem;">📄 关联日志 (${(data.related_logs && data.related_logs.length) || 0}条)</h5>
        <div class="replay-logs">${(data.related_logs || []).map(l =>
          `<div class="log-row">
            <span>${new Date(l.created_at).toLocaleString()}</span>
            <span class="level-${l.level}">${l.level}</span>
            <span>${l.category}</span>
            <span>${this.escHtml(l.message)}</span>
          </div>`
        ).join('') || '<p style="color:var(--text-muted);">无关联日志</p>'}</div>
      `;

      this.replayEvents = data.timeline_events || [];
      this.replayStepIndex = 0;
      const playerEl = document.getElementById('replay-player');
      if (playerEl) {
        playerEl.style.display = this.replayEvents.length > 0 ? 'block' : 'none';
        this.renderReplayStep();
      }

      panel.scrollIntoView({ behavior: 'smooth' });
    } catch (e) { alert('获取回放数据失败: ' + e.message); }
  },

  renderReplayStep() {
    const view = document.getElementById('replay-step-view');
    const info = document.getElementById('replay-step-info');
    if (!view || !this.replayEvents.length) return;

    const idx = this.replayStepIndex;
    const ev = this.replayEvents[idx];
    const typeIcon = { log: '📄', record: '🖼️', health: '🖥️', alert: '🚨' };
    const levelClass = ev.level === 'critical' || ev.level === 'CRITICAL' ? 'critical'
      : (ev.level === 'warning' || ev.level === 'WARN' ? 'warning' : 'info');

    view.innerHTML = `
      <div class="replay-step-card ${levelClass}">
        <div class="replay-step-header">
          <span>${typeIcon[ev.type] || '•'} ${this.escHtml(ev.title)}</span>
          <span class="replay-step-time">${ev.time ? new Date(ev.time).toLocaleString() : ''}</span>
        </div>
        ${ev.image ? `<img src="${ev.image}" alt="回放截图" class="replay-step-img">` : ''}
        ${ev.detail && typeof ev.detail === 'object' ? `<pre class="replay-detail">${JSON.stringify(ev.detail, null, 2)}</pre>` : ''}
      </div>`;

    if (info) info.textContent = `${idx + 1} / ${this.replayEvents.length}`;
  },

  replayStep(delta) {
    if (!this.replayEvents.length) return;
    this.replayStepIndex = Math.max(0, Math.min(this.replayEvents.length - 1, this.replayStepIndex + delta));
    this.renderReplayStep();
  },

  replayTogglePlay() {
    if (this.replayPlayTimer) {
      this.replayStopPlay();
      return;
    }
    const btn = document.getElementById('replay-play-btn');
    if (btn) btn.textContent = '⏸ 暂停';
    this.replayPlayTimer = setInterval(() => {
      if (this.replayStepIndex >= this.replayEvents.length - 1) {
        this.replayStopPlay();
        return;
      }
      this.replayStep(1);
    }, 1500);
  },

  replayStopPlay() {
    if (this.replayPlayTimer) {
      clearInterval(this.replayPlayTimer);
      this.replayPlayTimer = null;
    }
    const btn = document.getElementById('replay-play-btn');
    if (btn) btn.textContent = '▶ 播放';
  },

  closeReplay() {
    this.replayStopPlay();
    document.getElementById('replay-panel').style.display = 'none';
    this.currentReplayId = null;
    this.replayEvents = [];
    this.replayStepIndex = 0;
  },

  async resolveAlert(alertId) {
    try {
      await this.api(`/api/monitor/alerts/${alertId}/resolve`, { method: 'POST', body: JSON.stringify({ resolution_note: '手动处理' }) });
      if (this.focusedAlert && this.focusedAlert.id === alertId) this.clearFocusedAlert();
      if (this.currentReplayId === alertId) this.closeReplay();
      this.loadAlerts();
      this.loadAlertAnalytics();
    } catch (e) { alert(e.message); }
  },

  async testAlert() {
    try {
      const data = await this.api('/api/monitor/alerts/test', { method: 'POST' });
      this.showToast({ level: data.level, title: data.title, summary: data.summary });
      this.loadAlerts();
      this.loadAlertAnalytics();
    } catch (e) { alert(e.message); }
  },

  async loadAlertTypes() {
    try {
      const types = await this.api('/api/monitor/alerts/event-types');
      this.alertTypes = types;
      const filterSelect = document.getElementById('alert-filter-type');
      const testSelect = document.getElementById('test-alert-type');
      const options = types.map(t => `<option value="${t.key}">${t.name} (${t.default_level})</option>`).join('');
      if (filterSelect) filterSelect.innerHTML = '<option value="">全部类型</option>' + options;
      if (testSelect) testSelect.innerHTML = '<option value="">选择测试类型</option>' + options;
    } catch (e) {}
  },

  async triggerTypeAlert() {
    const sel = document.getElementById('test-alert-type');
    if (!sel || !sel.value) return;
    try {
      const data = await this.api(`/api/monitor/alerts/test/${sel.value}`, { method: 'POST' });
      this.showToast({ level: data.level, title: data.title, summary: data.summary });
      this.loadAlerts();
      this.loadAlertAnalytics();
      this.loadAgentActivity();
    } catch (e) { alert(e.message); }
  },

  async loadAgentActivity() {
    const el = document.getElementById('agent-activity');
    if (!el) return;
    try {
      const list = await this.api('/api/monitor/logs?category=agent&limit=15');
      if (!list.length) {
        el.innerHTML = '<p style="color:var(--text-muted)">暂无智能体日志，触发识别或告警后会自动记录</p>';
        return;
      }
      el.innerHTML = list.map(log => {
        const time = log.created_at ? new Date(log.created_at).toLocaleString() : '';
        const levelClass = (log.level || 'INFO').toLowerCase();
        return `<div class="agent-activity-item ${levelClass}">
          <span class="agent-activity-time">${time}</span>
          <span class="agent-activity-level">${log.level || 'INFO'}</span>
          <span class="agent-activity-msg">${this.escHtml(log.message || '')}</span>
        </div>`;
      }).join('');
    } catch (e) {
      el.innerHTML = '<p style="color:var(--text-muted)">加载失败</p>';
    }
  },

  async cleanupNoiseAlerts() {
    if (!confirm('将测试告警和可选配置缺失类历史告警标记为已处理，是否继续？')) return;
    try {
      const data = await this.api('/api/monitor/alerts/cleanup-noise', { method: 'POST' });
      alert(`已清理 ${data.resolved} 条噪声告警`);
      this.loadAlerts();
      this.loadAlertAnalytics();
    } catch (e) { alert(e.message); }
  },

  async loadAlertNotifications() {
    const el = document.getElementById('alert-notifications');
    if (!el) return;
    try {
      const conn = await this.api('/api/monitor/connections');
      const cfg = await this.api('/api/monitor/config');
      el.innerHTML = `
        <div class="notification-card on"><strong>WebSocket</strong><span>${conn.websocket_clients} 个在线连接</span></div>
        <div class="notification-card ${cfg.sse_enabled ? 'on' : 'off'}"><strong>SSE 实时推送</strong><span>${cfg.sse_enabled ? '已启用' : '未启用'}</span></div>
        <div class="notification-card ${cfg.webhook_enabled ? 'on' : 'off'}"><strong>Webhook</strong><span>${cfg.webhook_enabled ? (cfg.webhook_url_configured ? '已启用' : '已启用但未配置 URL') : '未启用'}</span></div>
        <div class="notification-card ${cfg.email_enabled ? 'on' : 'off'}"><strong>邮件通知</strong><span>${cfg.email_enabled ? (cfg.email_configured ? '已启用' : '已启用但 SMTP 不完整') : '未启用'}</span></div>
      `;
    } catch (e) {
      el.innerHTML = '<p style="color:var(--text-muted)">通知状态加载失败</p>';
    }
  },

  async testNotifications(channel) {
    try {
      const data = await this.api(`/api/monitor/notifications/test?channel=${encodeURIComponent(channel || 'all')}`, { method: 'POST' });
      const lines = Object.entries(data.channels || {}).map(([name, result]) => {
        if (typeof result === 'object' && result !== null) {
          if (result.ok) return `${name}: 成功`;
          return `${name}: 失败${result.reason ? `（${result.reason}）` : ''}`;
        }
        return `${name}: ${result ? '成功' : '失败'}`;
      });
      alert('通知测试完成\n\n' + (lines.join('\n') || '无可用渠道'));
      this.loadAlertNotifications();
    } catch (e) { alert(e.message); }
  },

  async loadAlertConfig() {
    const el = document.getElementById('alert-config');
    if (!el) return;
    try {
      const cfg = await this.api('/api/monitor/config');
      el.innerHTML = `
        <div class="config-grid">
          <div><span>连续失败阈值</span><strong>${cfg.failure_threshold}</strong></div>
          <div><span>滑窗秒数</span><strong>${cfg.window_seconds}</strong></div>
          <div><span>冷却秒数</span><strong>${cfg.cooldown_seconds}</strong></div>
          <div><span>低置信度阈值</span><strong>${cfg.low_confidence_threshold}</strong></div>
          <div><span>Token 上限</span><strong>${cfg.token_limit}</strong></div>
          <div><span>LLM 模型</span><strong>${cfg.llm_model || '模板降级'}</strong></div>
          <div><span>LLM 状态</span><strong>${cfg.llm_configured ? '已配置' : '未配置（模板告警）'}</strong></div>
          <div><span>巡检周期</span><strong>60 秒</strong></div>
        </div>`;
    } catch (e) {
      el.innerHTML = '<p style="color:var(--text-muted)">配置加载失败</p>';
    }
  },

  // ── 系统日志 ──
  onLogDatetimeChange(el) {
    if (!el) return;
    el.classList.toggle('has-value', !!el.value);
  },

  syncLogDatetimeState() {
    ['log-start', 'log-end'].forEach(id => {
      const el = document.getElementById(id);
      if (el) this.onLogDatetimeChange(el);
    });
  },

  initSelectChevrons() {
    document.querySelectorAll('.select-input-wrap select').forEach(select => {
      const wrap = select.closest('.select-input-wrap');
      if (!wrap || wrap.dataset.chevronBound) return;
      wrap.dataset.chevronBound = '1';
      const close = () => wrap.classList.remove('is-open');
      select.addEventListener('mousedown', () => wrap.classList.add('is-open'));
      select.addEventListener('blur', close);
      select.addEventListener('change', close);
    });
  },

  resetLogFilters(reload = true) {
    const ids = ['log-category', 'log-level', 'log-search', 'log-user', 'log-start', 'log-end'];
    const defaults = { 'log-category': '', 'log-level': '', 'log-search': '', 'log-user': '', 'log-start': '', 'log-end': '' };
    ids.forEach(id => {
      const el = document.getElementById(id);
      if (el) el.value = defaults[id] ?? '';
    });
    this.syncLogDatetimeState();
    if (reload) this.loadLogs();
  },

  logCategoryLabel(category) {
    return this.LOG_CATEGORY_LABELS[category] || category;
  },

  getLogFilterParams() {
    return {
      cat: (document.getElementById('log-category') && document.getElementById('log-category').value) || '',
      level: (document.getElementById('log-level') && document.getElementById('log-level').value) || '',
      search: (document.getElementById('log-search') && document.getElementById('log-search').value) || '',
      userId: (document.getElementById('log-user') && document.getElementById('log-user').value) || '',
      start: (document.getElementById('log-start') && document.getElementById('log-start').value) || '',
      end: (document.getElementById('log-end') && document.getElementById('log-end').value) || '',
    };
  },

  logMatchesFilters(log, filters) {
    if (!log) return false;
    if (filters.cat && log.category !== filters.cat) return false;
    if (filters.level && log.level !== filters.level) return false;
    if (filters.userId && String(log.user_id || '') !== String(filters.userId)) return false;
    if (filters.search) {
      const q = filters.search.toLowerCase();
      const msg = String(log.message || '').toLowerCase();
      if (!msg.includes(q)) return false;
    }
    if (filters.start) {
      const ts = new Date(log.created_at).getTime();
      if (ts < new Date(filters.start).getTime()) return false;
    }
    if (filters.end) {
      const ts = new Date(log.created_at).getTime();
      if (ts > new Date(filters.end).getTime()) return false;
    }
    return true;
  },

  renderLogRow(log, live) {
    const liveClass = live ? ' log-row-live' : '';
    return `
      <div class="log-row${liveClass}" onclick="App.showLogDetail('${this.escAttr(JSON.stringify(log))}')">
        <span>${new Date(log.created_at).toLocaleString()}</span>
        <span class="level-${log.level}">${log.level}</span>
        <span>${this.escHtml(this.logCategoryLabel(log.category))}</span>
        <span>${this.escHtml(log.message)}</span>
        <span>${log.user_id || '-'}</span>
      </div>`;
  },

  renderLogTable(logs) {
    const table = document.getElementById('log-table');
    if (!table) return;
    table.innerHTML =
      '<div class="log-row header"><span>时间</span><span>级别</span><span>类别</span><span>消息</span><span>用户</span></div>' +
      (logs || []).map(l => this.renderLogRow(l, false)).join('') ||
      '<p style="padding:1rem;color:var(--text-muted);">暂无日志</p>';
  },

  prependLiveLog(log) {
    if (!this.logMatchesFilters(log, this.getLogFilterParams())) return;
    const table = document.getElementById('log-table');
    if (!table) return;
    const empty = table.querySelector('p');
    if (empty) empty.remove();
    if (!table.querySelector('.log-row.header')) {
      table.innerHTML = '<div class="log-row header"><span>时间</span><span>级别</span><span>类别</span><span>消息</span><span>用户</span></div>';
    }
    const header = table.querySelector('.log-row.header');
    header.insertAdjacentHTML('afterend', this.renderLogRow(log, true));
    const rows = table.querySelectorAll('.log-row:not(.header)');
    if (rows.length > 100) rows[rows.length - 1].remove();
    setTimeout(() => {
      const first = table.querySelector('.log-row-live');
      if (first) first.classList.remove('log-row-live');
    }, 2500);
    this.loadLogStats();
  },

  renderLogStats(stats) {
    const statsPanel = document.getElementById('log-stats-panel');
    const statsEl = document.getElementById('log-stats');
    const chartsEl = document.getElementById('log-charts');
    if (!statsPanel || !statsEl || !chartsEl || !stats) return;
    statsPanel.style.display = 'block';

    const catHtml = Object.entries(stats.by_category || {}).map(([k, v]) =>
      `<span class="badge">${this.escHtml(this.logCategoryLabel(k))}: ${v}</span>`
    ).join('');
    statsEl.innerHTML = `
      <span>${stats.hours || 24}h 总计: <b>${stats.total}</b> 条</span>
      ${catHtml}
      ${Object.entries(stats.by_level || {}).map(([k,v]) => `<span class="badge level-${k}">${k}: ${v}</span>`).join('')}
    `;

    const ranked = stats.category_ranked || [];
    const maxCat = Math.max(...ranked.map(c => c.count), 1);
    let categoryHtml = '<div class="log-chart-panel"><h4>类别分布</h4><div class="dist-bars">';
    for (const item of ranked) {
      const pct = Math.round(item.count / maxCat * 100);
      const color = this.LOG_CATEGORY_COLORS[item.key] || '#9B9890';
      categoryHtml += `<div class="dist-row"><span class="dist-label dist-label-wide">${this.escHtml(item.name)}</span>
        <div class="dist-bar-track"><div class="dist-bar-fill" style="width:${pct}%;background:${color};"></div></div>
        <span class="dist-count">${item.count}</span></div>`;
    }
    categoryHtml += ranked.length ? '</div></div>' : '<p class="log-chart-empty">暂无类别数据</p></div>';

    const hourly = stats.hour_trend || [];
    const maxHour = Math.max(...hourly.map(h => h.count), 1);
    let hourHtml = '<div class="log-chart-panel"><h4>时间趋势</h4><div class="hourly-chart log-hourly-chart">';
    for (const h of hourly) {
      const label = h.hour ? h.hour.slice(11, 16) : '';
      const hPct = Math.max(4, Math.round(h.count / maxHour * 100));
      hourHtml += `<div class="hourly-bar" title="${this.escHtml(h.hour || '')}: ${h.count}条">
        <div class="hourly-fill" style="height:${hPct}%;"></div>
        <span class="hourly-label">${label}</span>
      </div>`;
    }
    hourHtml += hourly.length ? '</div></div>' : '<p class="log-chart-empty">暂无趋势数据</p></div>';

    chartsEl.innerHTML = categoryHtml + hourHtml;
  },

  async loadLogStats() {
    try {
      const hoursEl = document.getElementById('log-stats-hours');
      const hours = hoursEl ? hoursEl.value : 24;
      const stats = await this.api(`/api/monitor/logs/stats?hours=${hours}`);
      this.renderLogStats(stats);
    } catch (e) {}
  },

  connectLogStream() {
    if (this.logSseSource) return;
    const statusEl = document.getElementById('log-stream-status');
    const btn = document.getElementById('log-stream-btn');
    this.logSseSource = new EventSource('/api/monitor/logs/stream');
    this.logSseSource.onopen = () => {
      if (statusEl) { statusEl.textContent = '监听中'; statusEl.className = 'conn-status connected'; }
      if (btn) btn.textContent = '停止监听';
    };
    this.logSseSource.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === 'log') this.prependLiveLog(data);
      } catch (err) {}
    };
    this.logSseSource.onerror = () => {
      if (statusEl) { statusEl.textContent = '重连中...'; statusEl.className = 'conn-status reconnecting'; }
    };
  },

  disconnectLogStream() {
    if (this.logSseSource) {
      this.logSseSource.close();
      this.logSseSource = null;
    }
    const statusEl = document.getElementById('log-stream-status');
    const btn = document.getElementById('log-stream-btn');
    if (statusEl) { statusEl.textContent = '未连接'; statusEl.className = 'conn-status'; }
    if (btn) btn.textContent = '实时监听';
  },

  toggleLogStream() {
    if (this.logSseSource) this.disconnectLogStream();
    else this.connectLogStream();
  },

  async loadLogs() {
    try {
      const filters = this.getLogFilterParams();

      let url = '/api/monitor/logs?limit=100';
      if (filters.cat) url += '&category=' + filters.cat;
      if (filters.level) url += '&level=' + filters.level;
      if (filters.search) url += '&search=' + encodeURIComponent(filters.search);
      if (filters.userId) url += '&user_id=' + filters.userId;
      if (filters.start) url += '&start=' + new Date(filters.start).toISOString();
      if (filters.end) url += '&end=' + new Date(filters.end).toISOString();

      const data = await this.api(url);
      this.renderLogTable(data);
      await this.loadLogStats();
    } catch (e) {
      document.getElementById('log-table').innerHTML = `<p style="padding:1rem;color:var(--danger);">加载日志失败: ${e.message}</p>`;
    }
  },

  showLogDetail(jsonStr) {
    try {
      const log = JSON.parse(jsonStr);
      let detailHtml = '';
      if (log.detail_json && typeof log.detail_json === 'object') {
        detailHtml = `<pre class="replay-detail" style="max-height:200px;overflow-y:auto;">${JSON.stringify(log.detail_json, null, 2)}</pre>`;
      }
      alert(`日志详情\n\n时间: ${new Date(log.created_at).toLocaleString()}\n级别: ${log.level}\n类别: ${log.category}\n消息: ${log.message}\n${detailHtml ? '详情: 见下方' : ''}`);
    } catch (e) {}
  },

  exportLogs(format) {
    const rows = document.querySelectorAll('#log-table .log-row:not(.header)');
    if (rows.length === 0) { alert('没有可导出的日志'); return; }
    const data = [];
    rows.forEach(r => {
      const cells = r.querySelectorAll('span');
      data.push({
        time: cells[0] ? cells[0].textContent : '',
        level: cells[1] ? cells[1].textContent : '',
        category: cells[2] ? cells[2].textContent : '',
        message: cells[3] ? cells[3].textContent : '',
        user: cells[4] ? cells[4].textContent : '-',
      });
    });
    let content, mime, ext;
    if (format === 'csv') {
      content = '时间,级别,类别,消息,用户\n' + data.map(d => `"${d.time}","${d.level}","${d.category}","${d.message}","${d.user}"`).join('\n');
      mime = 'text/csv'; ext = 'csv';
    } else {
      content = JSON.stringify(data, null, 2);
      mime = 'application/json'; ext = 'json';
    }
    const blob = new Blob(['\uFEFF' + content], { type: mime + ';charset=utf-8' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
    a.download = `logs_${new Date().toISOString().slice(0,10)}.${ext}`;
    a.click();
  },

  healthLabel(status) {
    const labels = { healthy: '健康', warning: '警告', critical: '严重', unknown: '未知', error: '异常' };
    return labels[status] || status || '-';
  },

  escHtml(text) {
    if (text == null) return '';
    return String(text).replace(/[&<>'"]/g, m => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[m]));
  },

  /** 助手气泡：转义 HTML 并渲染基础 Markdown（**加粗**、换行） */
  formatAssistantText(text) {
    if (text == null) return '';
    let s = this.escHtml(text);
    s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/\n/g, '<br>');
    return s;
  },

  stripMarkdown(text) {
    if (!text) return '';
    return String(text)
      .replace(/\*\*([^*]+)\*\*/g, '$1')
      .replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '$1')
      .replace(/^#+\s*/gm, '');
  },

  escAttr(text) {
    if (text == null) return '';
    return String(text).replace(/[&<>'"]/g, m => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[m]));
  },

  // ── 日期时间默认值 ──
  initDatetimeDefaults() {
    // 日志中心默认不按时间筛选，避免「结束时间」停留在页面打开时刻导致新日志被过滤
  },

  // ── 告警智能体可视化 ──
  initAssistant() {
    const input = document.getElementById('assistant-input');
    if (input) input.addEventListener('keydown', (e) => { if (e.key === 'Enter') this.askAssistant(); });
    this.initAssistantVoice();
    this.renderAssistantHistory();
    this.setAssistantStatus('点击「立即巡检」查看系统状态');
    this.initAgentDrag();
    this.restoreAgentPosition();
    this.setAgentState('idle');
  },

  initAssistantVoice() {
    const saved = localStorage.getItem('assistantVoiceEnabled');
    this.assistantVoiceEnabled = saved === 'true';
    this.updateVoiceToggleUI();

    const loadVoices = () => {
      this._agentVoice = this.pickDoubaoStyleVoice();
    };
    if ('speechSynthesis' in window) {
      loadVoices();
      window.speechSynthesis.onvoiceschanged = loadVoices;
    }
  },

  /** 挑选接近豆包风格的自然中文语音（优先神经网络/在线音色） */
  pickDoubaoStyleVoice() {
    if (!('speechSynthesis' in window)) return null;
    const voices = window.speechSynthesis.getVoices();
    if (!voices.length) return null;

    const zhVoices = voices.filter(v => v.lang && (v.lang.startsWith('zh') || v.lang.includes('CN')));
    const pool = zhVoices.length ? zhVoices : voices;

    const avoid = /kangkang|yunjian|yunxi|yunze|male|guy|david|童|child|junior|yaoyao|huihui|kang/i;

    const priorityPatterns = [
      /xiaoxiao.*(neural|online|natural)/i,
      /晓晓.*(自然|在线|神经)/i,
      /xiaoyi.*(neural|online|natural)/i,
      /晓伊.*(自然|在线|神经)/i,
      /xiaoxuan.*neural/i,
      /xiaomo.*neural/i,
      /yunxia.*neural/i,
      /neural.*zh[- ]?cn/i,
      /online.*natural/i,
      /natural.*zh/i,
      /google.*普通话.*(中国|中国大陆)/i,
      /microsoft.*xiaoxiao/i,
      /microsoft.*xiaoyi/i,
    ];

    for (const pattern of priorityPatterns) {
      const hit = pool.find(v => pattern.test(v.name) && !avoid.test(v.name));
      if (hit) return hit;
    }

    const cloudNatural = pool.filter(v =>
      !v.localService && /xiaoxiao|xiaoyi|neural|natural|online|晓晓|晓伊/i.test(v.name) && !avoid.test(v.name)
    );
    if (cloudNatural.length) return cloudNatural[0];

    let best = null;
    let bestScore = -999;
    for (const v of pool) {
      if (avoid.test(v.name)) continue;
      const name = v.name;
      let score = 0;
      if (v.lang === 'zh-CN' || v.lang === 'cmn-CN') score += 10;
      if (/neural|natural|online/i.test(name)) score += 20;
      if (/xiaoxiao|xiaoyi|晓晓|晓伊/i.test(name)) score += 18;
      if (!v.localService) score += 8;
      if (/google|microsoft/i.test(name)) score += 4;
      if (score > bestScore) {
        bestScore = score;
        best = v;
      }
    }
    return best || pool.find(v => !avoid.test(v.name)) || pool[0];
  },

  onVoiceToggleChange(enabled) {
    this.assistantVoiceEnabled = !!enabled;
    localStorage.setItem('assistantVoiceEnabled', String(this.assistantVoiceEnabled));
    this.updateVoiceToggleUI();
    if (!this.assistantVoiceEnabled) {
      this.stopAssistantSpeech();
      this.setAssistantStatus('语音朗读已关闭');
    } else {
      this.setAssistantStatus('语音朗读已开启');
      this.speakAssistant('语音朗读已开启', { force: true });
    }
  },

  updateVoiceToggleUI() {
    const toggle = document.getElementById('assistant-voice-toggle');
    if (toggle) toggle.checked = this.assistantVoiceEnabled;
    const label = document.getElementById('voice-toggle-status');
    if (label) {
      label.textContent = this.assistantVoiceEnabled ? '朗读开' : '朗读关';
      label.classList.toggle('on', this.assistantVoiceEnabled);
    }
  },

  prepareSpeechText(text) {
    if (!text) return '';
    let t = this.stripMarkdown(text)
      .replace(/[🔔🔍⚠️💡✅🛠📋🧪•]/g, '')
      .replace(/\n+/g, '，')
      .replace(/\s+/g, ' ')
      .trim();
    if (t.length > 200) t = t.slice(0, 200) + '…';
    return t;
  },

  stopAssistantSpeech() {
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    this._agentSpeaking = false;
  },

  startAgentMonitorLoop() {
    this.stopAgentMonitorLoop();
    // 首次进入主动播报一次
    this.runAgentPatrol({ silent: true, speech: true, forceSpeech: true });
    this.agentMonitorTimer = setInterval(
      () => this.runAgentPatrol({ silent: true, speech: true }),
      30000
    );
  },

  stopAgentMonitorLoop() {
    if (this.agentMonitorTimer) {
      clearInterval(this.agentMonitorTimer);
      this.agentMonitorTimer = null;
    }
  },

  updateAgentLiveStatus(briefing) {
    const dot = document.getElementById('agent-live-dot');
    const text = document.getElementById('agent-live-text');
    if (!dot || !text || !briefing) return;

    const open = briefing.open_alerts || 0;
    const warnLogs = (briefing.logs_24h && briefing.logs_24h.warn_or_above) || 0;
    let state = 'ok';
    if (open > 0 || warnLogs > 0) state = 'warn';
    if (open >= 3) state = 'critical';

    dot.className = 'agent-live-dot ' + state;
    text.textContent = open > 0
      ? `监控中 · 有 ${open} 个问题待处理`
      : `一切正常 · 近24小时 ${(briefing.logs_24h && briefing.logs_24h.total) || 0} 次记录`;
  },

  async runAgentPatrol(opts = {}) {
    const { silent = false, speech = false, forceSpeech = false } = opts;
    if (!silent) {
      this.setAgentState('thinking');
      this.setAssistantStatus('正在巡检系统…', true);
    }
    try {
      const data = await this.api('/api/monitor/agent/briefing');
      this.agentBriefing = data;
      this.agentOpenCount = data.open_alerts || 0;
      this.updateAgentBadge();
      this.updateAgentLiveStatus(data);

      const briefKey = `${data.open_alerts}|${(data.logs_24h && data.logs_24h.warn_or_above) || 0}`;
      const statusChanged = briefKey !== this.agentLastBriefKey;
      this.agentLastBriefKey = briefKey;

      const subtitle = document.getElementById('agent-subtitle');
      if (subtitle) {
        subtitle.textContent = data.open_alerts > 0
          ? `发现 ${data.open_alerts} 条未处理告警`
          : '持续监听三路识别与用户操作';
      }

      if (data.open_alerts > 0) {
        this.setAgentState('warning');
      } else if (!this.assistantThinking) {
        this.setAgentState('idle');
      }

      // 仅在状态变化或首次/手动巡检时说话，避免每 30 秒重复播报
      if (speech && (forceSpeech || statusChanged)) {
        const short = data.open_alerts > 0
          ? `提醒您，还有 ${data.open_alerts} 个问题待处理哦`
          : (statusChanged ? '系统运行正常，我会继续帮您看着' : '');
        if (short) this.showAgentSpeech(short, 5000);
      }

      if (!silent) {
        const summary = data.summary_user || data.summary;
        this.addAssistantMessage('assistant', summary);
        if (this.assistantVoiceEnabled) this.speakAssistant(summary);
        if (data.recent_alerts && data.recent_alerts.length) {
          const list = data.recent_alerts.slice(0, 3).map(a =>
            `• ${a.summary_user || a.title || a.event_type_user}`
          ).join('\n');
          this.addAssistantMessage('assistant', `最近的情况：\n${list}\n\n若要问某一条的根因或处理方式，请先在告警中心点「回放」选定，或点击上方最新告警卡片。`);
        }
        this.updateAssistantContextUI();
        this.setAssistantStatus('巡检完成，有问题可以继续问我');
      }

      return data;
    } catch (e) {
      if (!silent) {
        this.addAssistantMessage('assistant', `巡检失败：${e.message}`);
        this.setAssistantStatus('巡检失败，请确认后端服务已启动');
      }
      const text = document.getElementById('agent-live-text');
      if (text) text.textContent = '监控连接失败，请检查服务';
      const dot = document.getElementById('agent-live-dot');
      if (dot) dot.className = 'agent-live-dot critical';
      return null;
    } finally {
      if (!silent && !this.assistantThinking) this.setAgentState(this.agentOpenCount > 0 ? 'warning' : 'idle');
    }
  },

  async agentTriggerTestAlert() {
    this.setAgentState('thinking');
    this.setAssistantStatus('正在触发测试告警…', true);
    try {
      const data = await this.api('/api/monitor/alerts/test', { method: 'POST' });
      const userMsg = `我帮您发了一条测试提醒：「${data.title}」。${data.summary || ''}`;
      this.addAssistantMessage('assistant', userMsg);
      this.showAgentSpeech('已发送一条测试提醒，您可以体验完整流程', 4000);
      this.onAgentAlert({ id: data.id, level: data.level || 'info', title: data.title, summary: data.summary, suggestion: data.suggestion, event_type: data.event_type });
      this.setAssistantStatus('测试提醒已发出，请到告警中心查看');
    } catch (e) {
      this.addAssistantMessage('assistant', `触发失败：${e.message}`);
      this.setAssistantStatus('触发失败');
    } finally {
      if (!this.assistantThinking) this.setAgentState(this.agentOpenCount > 0 ? 'warning' : 'idle');
    }
  },

  initAgentDrag() {
    const bot = document.getElementById('assistant-bot');
    const avatar = document.getElementById('agent-avatar-wrap');
    if (!bot || !avatar) {
      console.warn('Alert Agent: 未找到智能体 DOM 元素');
      return;
    }

    if (avatar.dataset.bound === '1') return;
    avatar.dataset.bound = '1';

    const clampPosition = (left, top) => {
      const w = Math.max(bot.offsetWidth, 110);
      const h = Math.max(bot.offsetHeight, 110);
      return {
        left: Math.max(8, Math.min(window.innerWidth - w - 8, left)),
        top: Math.max(8, Math.min(window.innerHeight - h - 8, top)),
      };
    };

    const applyPosition = (left, top) => {
      const pos = clampPosition(left, top);
      bot.style.left = `${pos.left}px`;
      bot.style.top = `${pos.top}px`;
      bot.style.right = 'auto';
      bot.style.bottom = 'auto';
    };

    const onPointerDown = (e) => {
      if (e.button !== undefined && e.button !== 0) return;
      e.preventDefault();
      e.stopPropagation();

      const rect = bot.getBoundingClientRect();
      this._agentDrag = {
        pointerId: e.pointerId,
        offsetX: e.clientX - rect.left,
        offsetY: e.clientY - rect.top,
        startX: e.clientX,
        startY: e.clientY,
        moved: false,
      };
      this._agentPointerId = e.pointerId;
      bot.classList.add('dragging');
      avatar.classList.add('dragging');
      avatar.setPointerCapture(e.pointerId);
    };

    const onPointerMove = (e) => {
      if (!this._agentDrag || e.pointerId !== this._agentDrag.pointerId) return;
      const dx = Math.abs(e.clientX - this._agentDrag.startX);
      const dy = Math.abs(e.clientY - this._agentDrag.startY);
      if (dx > 5 || dy > 5) {
        this._agentDrag.moved = true;
        this.agentDragMoved = true;
      }
      if (!this._agentDrag.moved) return;
      e.preventDefault();
      applyPosition(e.clientX - this._agentDrag.offsetX, e.clientY - this._agentDrag.offsetY);
    };

    const onPointerUp = (e) => {
      if (!this._agentDrag || e.pointerId !== this._agentDrag.pointerId) return;
      const wasMoved = this._agentDrag.moved;
      bot.classList.remove('dragging');
      avatar.classList.remove('dragging');
      try { avatar.releasePointerCapture(e.pointerId); } catch (err) { /* ignore */ }

      if (wasMoved) {
        this.saveAgentPosition();
      } else {
        this.toggleAssistant();
      }

      this._agentDrag = null;
      this._agentPointerId = null;
      setTimeout(() => { this.agentDragMoved = false; }, 0);
    };

    avatar.addEventListener('pointerdown', onPointerDown);
    avatar.addEventListener('pointermove', onPointerMove);
    avatar.addEventListener('pointerup', onPointerUp);
    avatar.addEventListener('pointercancel', onPointerUp);

    // 兼容旧浏览器：无 Pointer Events 时回退到 mouse
    if (!window.PointerEvent) {
      avatar.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;
        e.preventDefault();
        const rect = bot.getBoundingClientRect();
        this._agentDrag = {
          offsetX: e.clientX - rect.left,
          offsetY: e.clientY - rect.top,
          startX: e.clientX,
          startY: e.clientY,
          moved: false,
        };
        bot.classList.add('dragging');
        const onMouseMove = (ev) => {
          if (!this._agentDrag) return;
          if (Math.abs(ev.clientX - this._agentDrag.startX) > 5 || Math.abs(ev.clientY - this._agentDrag.startY) > 5) {
            this._agentDrag.moved = true;
          }
          if (this._agentDrag.moved) {
            applyPosition(ev.clientX - this._agentDrag.offsetX, ev.clientY - this._agentDrag.offsetY);
          }
        };
        const onMouseUp = () => {
          if (!this._agentDrag) return;
          const moved = this._agentDrag.moved;
          bot.classList.remove('dragging');
          if (moved) this.saveAgentPosition();
          else this.toggleAssistant();
          this._agentDrag = null;
          document.removeEventListener('mousemove', onMouseMove);
          document.removeEventListener('mouseup', onMouseUp);
        };
        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
      });
    }
  },

  saveAgentPosition() {
    const bot = document.getElementById('assistant-bot');
    if (!bot) return;
    const rect = bot.getBoundingClientRect();
    if (rect.left >= 0 && rect.top >= 0 && rect.left < window.innerWidth - 40 && rect.top < window.innerHeight - 40) {
      localStorage.setItem('agentPosition', JSON.stringify({ left: rect.left, top: rect.top }));
    }
  },

  restoreAgentPosition() {
    const bot = document.getElementById('assistant-bot');
    const raw = localStorage.getItem('agentPosition');
    if (!bot || !raw) return;
    try {
      const pos = JSON.parse(raw);
      if (typeof pos.left === 'number' && typeof pos.top === 'number') {
        const w = Math.max(bot.offsetWidth, 110);
        const h = Math.max(bot.offsetHeight, 110);
        const inView = pos.left >= -20 && pos.top >= -20
          && pos.left < window.innerWidth - 40
          && pos.top < window.innerHeight - 40;
        if (inView) {
          bot.style.left = `${pos.left}px`;
          bot.style.top = `${pos.top}px`;
          bot.style.right = 'auto';
          bot.style.bottom = 'auto';
        } else {
          localStorage.removeItem('agentPosition');
        }
      }
    } catch (e) {
      localStorage.removeItem('agentPosition');
    }
  },

  setAgentState(state) {
    const ring = document.getElementById('agent-status-ring');
    const wrap = document.getElementById('agent-avatar-wrap');
    const s = state || 'idle';
    if (ring) ring.className = 'agent-status-ring agent-state-' + s;
    if (wrap) {
      const wasActive = wrap.classList.contains('active');
      const moodMap = {
        idle: 'idle',
        info: 'idle',
        thinking: 'thinking',
        listening: 'listening',
        speaking: 'speaking',
        warning: 'error',
        critical: 'error',
        error: 'error',
      };
      wrap.className = 'agent-avatar-wrap agent-mood-' + (moodMap[s] || 'idle');
      if (wasActive) wrap.classList.add('active');
    }
  },

  showAgentSpeech(text, duration = 6000) {
    const el = document.getElementById('agent-speech');
    if (!el) return;
    el.textContent = text;
    el.classList.add('visible');
    if (this.agentSpeechTimer) clearTimeout(this.agentSpeechTimer);
    this.agentSpeechTimer = setTimeout(() => el.classList.remove('visible'), duration);
    if (this.assistantVoiceEnabled) this.speakAssistant(text);
  },

  onAgentAlert(alert) {
    const level = alert.level || 'info';
    this.setAgentState(level);
    if (alert.id) this.setFocusedAlert(alert);
    this.refreshAgentStats();

    const levelLabel = { info: '提示', warning: '需要注意', critical: '比较紧急' }[level] || '提醒';
    this.showAgentSpeech(this.alertToUserSpeech(alert), 8000);

    const latest = document.getElementById('agent-latest-alert');
    if (latest) {
      latest.className = 'agent-latest-alert ' + level;
      latest.innerHTML = `<strong>${this.escHtml(alert.title || '系统提醒')}</strong>${this.escHtml(alert.summary || '')}${alert.suggestion ? '<br><em>建议：' + this.escHtml(alert.suggestion) + '</em>' : ''}<br><span class="agent-latest-hint">点击此卡片 · 设为当前讨论告警</span>`;
      latest.classList.remove('hidden');
      latest.onclick = () => this.setFocusedAlert(alert);
      latest.title = '点击将此告警设为当前讨论对象';
    }

    const subtitle = document.getElementById('agent-subtitle');
    if (subtitle) subtitle.textContent = `刚发现：${alert.title || '系统异常'}`;

    const userLines = [alert.summary || alert.title || '系统检测到一项异常'];
    if (alert.suggestion) userLines.push('建议：' + alert.suggestion);
    const alertMsg = `🔔 ${levelLabel}提醒\n${userLines.join('\n')}`;
    this.addAssistantMessage('assistant', alertMsg);
    this.loadAgentActivity();

    if (level === 'critical') {
      this.toggleAssistant(true);
    }

    setTimeout(() => {
      if (!this.assistantThinking) this.setAgentState(this.agentOpenCount > 0 ? 'warning' : 'idle');
    }, 12000);
  },

  updateAgentBadge() {
    const badge = document.getElementById('agent-badge');
    if (!badge) return;
    if (this.agentOpenCount > 0) {
      badge.textContent = this.agentOpenCount > 99 ? '99+' : String(this.agentOpenCount);
      badge.classList.remove('hidden');
    } else {
      badge.classList.add('hidden');
    }
  },

  async refreshAgentStats() {
    try {
      const stats = await this.api('/api/monitor/alerts/stats');
      this.agentOpenCount = stats.open || 0;
      this.updateAgentBadge();
      if (stats.open > 0) {
        this.setAgentState('warning');
        this.showAgentSpeech(`当前有 ${stats.open} 条未处理告警`, 5000);
      }
    } catch (e) { /* ignore */ }
  },

  goToAlerts() {
    const nav = document.querySelector('.nav-item[data-view="alerts"]');
    if (nav) nav.click();
    const panel = document.getElementById('assistant-panel');
    if (panel) panel.classList.add('hidden');
  },

  toggleAssistant(forceOpen) {
    const panel = document.getElementById('assistant-panel');
    const avatar = document.getElementById('agent-avatar-wrap');
    if (!panel || !avatar) return;

    let isHidden;
    if (forceOpen === true) {
      isHidden = false;
      panel.classList.remove('hidden');
    } else if (forceOpen === false) {
      isHidden = true;
      panel.classList.add('hidden');
    } else {
      isHidden = panel.classList.toggle('hidden');
    }

    avatar.classList.toggle('active', !isHidden);
    if (!isHidden) {
      this.renderAssistantHistory();
      this.refreshAgentStats();
      this.updateAssistantContextUI();
      if (this.assistantHistory.length === 0) {
        this.addAssistantMessage('assistant',
          '你好，我是小智，您的系统助手。\n\n' +
          '我会帮您盯着车牌识别、手势识别和账号安全。有问题我会用大白话告诉您，不用懂技术也能明白。\n\n' +
          '您可以：\n' +
          '• 点「立即巡检」—— 我帮您看看系统是否正常\n' +
          '• 点「模拟告警」—— 体验我会怎么提醒您\n' +
          '• 在告警中心「回放」选定一条后，再点根因/建议/影响\n' +
          '• 直接问我「系统正常吗」这类整体问题'
        );
      }
      this.runAgentPatrol({ silent: true });
      setTimeout(() => {
        document.addEventListener('click', this.closeAssistantOnOutsideClick);
      }, 0);
    } else {
      document.removeEventListener('click', this.closeAssistantOnOutsideClick);
    }
  },

  closeAssistantOnOutsideClick(evt) {
    const panel = document.getElementById('assistant-panel');
    const avatar = document.getElementById('agent-avatar-wrap');
    if (!panel || !avatar) return;
    if (evt.target instanceof Element && !panel.contains(evt.target) && !avatar.contains(evt.target)) {
      panel.classList.add('hidden');
      avatar.classList.remove('active');
      document.removeEventListener('click', App.closeAssistantOnOutsideClick);
    }
  },

  setAssistantStatus(text, isThinking = false) {
    const el = document.getElementById('assistant-status');
    if (!el) return;
    el.classList.toggle('thinking', isThinking);
    el.innerHTML = isThinking ? `<span class="assistant-thinking"><span></span><span></span><span></span></span> ${text}` : text;
  },

  addAssistantMessage(role, content) {
    this.assistantHistory.push({ role, content });
    if (this.assistantHistory.length > 50) this.assistantHistory.shift();
    this.renderAssistantHistory();
  },

  renderAssistantHistory() {
    const box = document.getElementById('assistant-history');
    if (!box) return;
    box.innerHTML = this.assistantHistory.map(msg => `
      <div class="assistant-msg ${msg.role}">
        <div class="assistant-msg-bubble">${msg.role === 'assistant' ? this.formatAssistantText(msg.content) : this.escHtml(msg.content)}</div>
      </div>
    `).join('');
    box.scrollTop = box.scrollHeight;
  },

  async askAssistant(question) {
    const input = document.getElementById('assistant-input');
    const q = typeof question === 'string' ? question : (input && input.value && input.value.trim());
    if (!q || this.assistantThinking) return;

    this.addAssistantMessage('user', q);
    if (input) input.value = '';
    this.assistantThinking = true;
    this.setAssistantStatus('Alert Agent 正在分析...', true);
    this.setAgentState('thinking');

    const panel = document.getElementById('assistant-panel');
    if (panel) panel.classList.add('assistant-processing');

    try {
      const body = this.buildAssistantPayload(q);
      const data = await this.api('/api/monitor/assistant', { method: 'POST', body: JSON.stringify(body) });
      const answer = data.answer || '我暂时没想好怎么说，您可以换个方式问问，或者先点「立即巡检」。';
      this.addAssistantMessage('assistant', answer);
      if (data.needs_clarification) {
        this.setAssistantStatus('请先选定一条告警');
      } else if (this.focusedAlert) {
        this.setAssistantStatus(`正在讨论：${this.focusedAlert.title}`);
      } else {
        this.setAssistantStatus('说完了，还有问题可以继续问');
      }
      this.setAgentState('idle');
      if (this.assistantVoiceEnabled) this.speakAssistant(answer);
    } catch (e) {
      this.addAssistantMessage('assistant', `抱歉，我没能连上后台：${e.message}。请确认系统已启动后再试。`);
      this.setAssistantStatus('请求失败，请稍后重试');
      this.setAgentState('warning');
    } finally {
      this.assistantThinking = false;
      if (panel) panel.classList.remove('assistant-processing');
    }
  },

  startPanelDrag(evt) {
    const panel = document.getElementById('assistant-panel');
    if (!panel || evt.target.closest('.assistant-close') || evt.target.closest('.assistant-icon-btn') || evt.target.closest('button') || evt.target.closest('input') || evt.target.closest('.assistant-history')) return;
    evt.preventDefault();
    this.dragOffsetX = evt.clientX - panel.getBoundingClientRect().left;
    this.dragOffsetY = evt.clientY - panel.getBoundingClientRect().top;
    panel.classList.add('dragging');
    const onMouseMove = (moveEvt) => {
      panel.style.left = `${moveEvt.clientX - this.dragOffsetX}px`;
      panel.style.top = `${moveEvt.clientY - this.dragOffsetY}px`;
      panel.style.right = 'auto'; panel.style.bottom = 'auto';
    };
    const onMouseUp = () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      panel.classList.remove('dragging');
    };
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  },

  startVoiceInput() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) { this.setAssistantStatus('当前浏览器不支持语音输入'); return; }
    if (this.assistantRecognition) { this.assistantRecognition.stop(); return; }
    const recognition = new SpeechRecognition();
    recognition.lang = 'zh-CN'; recognition.continuous = false; recognition.interimResults = false;
    this.assistantRecognition = recognition;
    this.setAgentState('listening');
    recognition.onstart = () => this.setAssistantStatus('正在聆听...', false);
    recognition.onresult = (event) => {
      const transcript = Array.from(event.results).map(r => r[0].transcript).join('');
      const inp = document.getElementById('assistant-input');
      if (inp) inp.value = transcript;
      this.askAssistant(transcript);
    };
    recognition.onerror = (e) => { this.setAssistantStatus(`语音输入失败: ${e.error}`); };
    recognition.onend = () => {
      this.assistantRecognition = null;
      if (!this.assistantThinking) {
        this.setAgentState('idle');
        this.setAssistantStatus('准备就绪');
      }
    };
    recognition.start();
  },

  speakAssistant(text, opts = {}) {
    const force = opts.force === true;
    if (!force && !this.assistantVoiceEnabled) return;
    if (!('speechSynthesis' in window) || !text) return;

    const prepared = this.prepareSpeechText(text);
    if (!prepared) return;

    this.stopAssistantSpeech();
    if (!this._agentVoice) this._agentVoice = this.pickDoubaoStyleVoice();

    const utterance = new SpeechSynthesisUtterance(prepared);
    utterance.lang = 'zh-CN';
    utterance.pitch = 1.0;
    utterance.rate = 0.92;
    utterance.volume = 1.0;
    if (this._agentVoice) utterance.voice = this._agentVoice;

    const prevState = this.agentOpenCount > 0 ? 'warning' : 'idle';
    this._agentSpeaking = true;
    if (!this.assistantThinking) this.setAgentState('speaking');

    utterance.onend = () => {
      this._agentSpeaking = false;
      if (!this.assistantThinking) {
        this.setAgentState(this.agentOpenCount > 0 ? 'warning' : prevState);
      }
    };
    utterance.onerror = () => {
      this._agentSpeaking = false;
      if (!this.assistantThinking) this.setAgentState(prevState);
    };

    window.speechSynthesis.speak(utterance);
  },

  speakLastAnswer() {
    const last = [...this.assistantHistory].reverse().find(msg => msg.role === 'assistant');
    if (last) this.speakAssistant(last.content, { force: true });
  },
};

document.addEventListener('DOMContentLoaded', () => App.init());
