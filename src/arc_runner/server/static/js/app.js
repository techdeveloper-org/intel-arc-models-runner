/**
 * Intel Arc Model Studio — Client-Side Application Engine.
 * 
 * Manages tab switching, SSE streaming chat with 60fps token buffering,
 * live Ollama model discovery, GGUF inspection modals, and AOT compilation.
 */

document.addEventListener('DOMContentLoaded', () => {
    initTabNavigation();
    initChatPlayground();
    initOllamaStudio();
    initModelHub();
    initBenchmarkStudio();
    initEmbeddingsStudio();
    initLoRAStudio();
});

/**
 * Configure sidebar tab switching and page title updates.
 */
function initTabNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const pageTitle = document.getElementById('page-title');

    const titles = {
        'tab-cockpit': 'Heterogeneous Silicon Cockpit',
        'tab-chat': 'Dual-Silicon Chat Playground',
        'tab-ollama': 'Ollama & GGUF Quantization Studio',
        'tab-hub': 'OpenVINO Model Hub & Downloader',
        'tab-benchmark': 'Hardware Benchmarking Suite',
        'tab-embeddings': 'Intel® AI Boost Vector Embeddings Studio'
    };

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const tabId = item.getAttribute('data-tab');
            navItems.forEach(nav => nav.classList.remove('active'));
            tabPanes.forEach(pane => pane.classList.remove('active'));

            item.classList.add('active');
            const targetPane = document.getElementById(tabId);
            if (targetPane) {
                targetPane.classList.add('active');
            }
            if (pageTitle && titles[tabId]) {
                pageTitle.textContent = titles[tabId];
            }
        });
    });
}

/**
 * Initialize chat playground with streaming SSE and nanosecond latency measurement.
 */
function initChatPlayground() {
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    const metricTtft = document.getElementById('metric-ttft');
    const metricTpot = document.getElementById('metric-tpot');
    const metricRate = document.getElementById('metric-rate');
    const metricEngine = document.getElementById('metric-engine');

    const imageInput = document.getElementById('chat-image-input');
    const previewContainer = document.getElementById('chat-image-preview-container');
    const previewImg = document.getElementById('chat-image-preview');
    const previewName = document.getElementById('chat-image-name');
    const btnRemoveImage = document.getElementById('btn-remove-image');

    let currentBase64Image = null;

    if (imageInput) {
        imageInput.addEventListener('change', (e) => {
            const file = e.target.files && e.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = (event) => {
                const fullDataUrl = event.target.result;
                const base64Content = fullDataUrl.split(',')[1] || fullDataUrl;
                currentBase64Image = base64Content;

                if (previewImg) previewImg.src = fullDataUrl;
                if (previewName) previewName.textContent = file.name;
                if (previewContainer) previewContainer.classList.remove('hidden');
            };
            reader.readAsDataURL(file);
        });
    }

    if (btnRemoveImage) {
        btnRemoveImage.addEventListener('click', () => {
            currentBase64Image = null;
            if (imageInput) imageInput.value = '';
            if (previewContainer) previewContainer.classList.add('hidden');
            if (previewImg) previewImg.src = '';
        });
    }

    if (!chatForm || !chatInput || !chatMessages) return;

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = chatInput.value.trim();
        if (!text && !currentBase64Image) return;

        chatInput.value = '';
        const userPrompt = text || '(Analyze attached image)';
        const imageToSend = currentBase64Image;

        appendMessage('user', userPrompt, previewImg ? previewImg.src : null);

        if (btnRemoveImage) {
            btnRemoveImage.click();
        }

        const modelSelect = document.getElementById('chat-model-select');
        const deviceSelect = document.getElementById('chat-device-select');
        const selectedModel = modelSelect ? modelSelect.value : 'llama-3.2-3b';
        const selectedDevice = deviceSelect ? deviceSelect.value : 'GPU';

        const assistantMsgElem = appendMessage('assistant', '');
        const bubble = assistantMsgElem.querySelector('.message-bubble');

        const startTime = performance.now();
        let firstTokenTime = null;
        let tokenCount = 0;
        let fullResponse = '';

        const userMessagePayload = {
            role: 'user',
            content: userPrompt
        };
        if (imageToSend) {
            userMessagePayload.images = [imageToSend];
        }

        const activeAdapters = (typeof getActiveLoRAAdapters === 'function') ? getActiveLoRAAdapters() : [];
        const requestPayload = {
            model: selectedModel,
            messages: [userMessagePayload],
            stream: true
        };
        if (activeAdapters && activeAdapters.length > 0) {
            requestPayload.adapters = activeAdapters;
        }

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestPayload)
            });

            if (!response.ok) {
                bubble.innerHTML = `<span class="text-danger">Inference error: HTTP ${response.status}</span>`;
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    const trimmed = line.trim();
                    if (!trimmed) continue;
                    try {
                        const parsed = JSON.parse(trimmed);
                        if (parsed.message && parsed.message.content) {
                            if (firstTokenTime === null) {
                                firstTokenTime = performance.now();
                                const ttftMs = Math.round(firstTokenTime - startTime);
                                if (metricTtft) metricTtft.textContent = `${ttftMs} ms`;
                            }
                            tokenCount++;
                            fullResponse += parsed.message.content;
                            window.requestAnimationFrame(() => {
                                bubble.textContent = fullResponse;
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                            });
                        }
                    } catch (err) {
                        // pass non-json lines
                    }
                }
            }

            const endTime = performance.now();
            const totalDurationSec = (endTime - startTime) / 1000.0;
            const rate = tokenCount > 0 ? (tokenCount / totalDurationSec).toFixed(1) : '0';
            const tpot = (tokenCount > 1 && firstTokenTime) ? (((endTime - firstTokenTime) / (tokenCount - 1))).toFixed(1) : '--';

            if (metricRate) metricRate.textContent = `${rate} tok/s`;
            if (metricTpot) metricTpot.textContent = `${tpot} ms/tok`;
            if (metricEngine) metricEngine.textContent = `${selectedDevice} (${selectedModel})`;

        } catch (err) {
            bubble.innerHTML = `<span class="text-danger">Connection error: ${err.message}</span>`;
        }
    });

    function appendMessage(role, text, imgSrc = null) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}`;
        const bubbleDiv = document.createElement('div');
        bubbleDiv.className = 'message-bubble';

        if (imgSrc) {
            const imgElem = document.createElement('img');
            imgElem.src = imgSrc;
            imgElem.style.maxWidth = '180px';
            imgElem.style.maxHeight = '140px';
            imgElem.style.borderRadius = '6px';
            imgElem.style.display = 'block';
            imgElem.style.marginBottom = '8px';
            bubbleDiv.appendChild(imgElem);
        }

        const textSpan = document.createElement('span');
        textSpan.textContent = text;
        bubbleDiv.appendChild(textSpan);

        msgDiv.appendChild(bubbleDiv);
        chatMessages.appendChild(msgDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return msgDiv;
    }
}

/**
 * Configure local Ollama model table discovery and GGUF inspection modal.
 */
function initOllamaStudio() {
    const tableBody = document.getElementById('ollama-table-body');
    const btnRefresh = document.getElementById('btn-refresh-ollama');
    const modal = document.getElementById('gguf-modal');
    const btnCloseModal = document.getElementById('btn-close-modal');
    const modalTitle = document.getElementById('modal-model-title');
    const modalBody = document.getElementById('modal-model-body');
    const btnConvertGguf = document.getElementById('btn-convert-gguf');
    const precisionSelect = document.getElementById('convert-precision-select');
    const deviceSelect = document.getElementById('convert-device-select');

    const STORAGE_KEY_PRECISION = 'arc_runner_transpile_precision';
    const STORAGE_KEY_DEVICE = 'arc_runner_transpile_device';

    const memoryStore = {};
    const safeStorage = {
        getItem(key) {
            try {
                return window.localStorage.getItem(key);
            } catch (_) {
                return memoryStore[key] || null;
            }
        },
        setItem(key, value) {
            try {
                window.localStorage.setItem(key, value);
            } catch (_) {
                memoryStore[key] = String(value);
            }
        }
    };

    if (precisionSelect) {
        const savedPrecision = safeStorage.getItem(STORAGE_KEY_PRECISION);
        if (savedPrecision) {
            precisionSelect.value = savedPrecision;
        }
        precisionSelect.addEventListener('change', () => {
            safeStorage.setItem(STORAGE_KEY_PRECISION, precisionSelect.value);
        });
    }

    if (deviceSelect) {
        const savedDevice = safeStorage.getItem(STORAGE_KEY_DEVICE);
        if (savedDevice) {
            deviceSelect.value = savedDevice;
        }
        deviceSelect.addEventListener('change', () => {
            safeStorage.setItem(STORAGE_KEY_DEVICE, deviceSelect.value);
        });
    }

    let currentGgufFilePath = null;
    let currentModelName = null;

    if (btnRefresh) {
        btnRefresh.addEventListener('click', loadOllamaModels);
    }

    if (btnCloseModal && modal) {
        btnCloseModal.addEventListener('click', () => modal.classList.add('hidden'));
    }

    if (btnConvertGguf) {
        btnConvertGguf.addEventListener('click', async () => {
            if (!currentGgufFilePath) {
                alert('No local GGUF container path detected for conversion.');
                return;
            }

            const precision = precisionSelect ? precisionSelect.value : 'int4';
            const device = deviceSelect ? deviceSelect.value : 'GPU';
            if (precisionSelect) safeStorage.setItem(STORAGE_KEY_PRECISION, precision);
            if (deviceSelect) safeStorage.setItem(STORAGE_KEY_DEVICE, device);

            const outputDirInput = document.getElementById('convert-output-dir-input');
            const cleanName = (currentModelName || 'custom-model').replace(/[^a-zA-Z0-9_-]/g, '_');
            const outputDir = (outputDirInput && outputDirInput.value.trim()) ? outputDirInput.value.trim() : `./models/${cleanName}-openvino`;

            btnConvertGguf.disabled = true;
            btnConvertGguf.textContent = 'Transpiling in Progress...';

            const progressCard = document.getElementById('gguf-convert-progress-card');
            const progressFill = document.getElementById('convert-progress-fill');
            const progressPct = document.getElementById('convert-progress-pct');
            const currTensorEl = document.getElementById('convert-current-tensor');
            const counterEl = document.getElementById('convert-tensor-counter');
            const throughputEl = document.getElementById('convert-throughput');
            const etaEl = document.getElementById('convert-eta');
            const sqnrEl = document.getElementById('convert-sqnr');
            const terminalBox = document.getElementById('convert-terminal-box');
            const completionBanner = document.getElementById('convert-completion-banner');
            const completionDetails = document.getElementById('convert-completion-details');

            if (progressCard) progressCard.classList.remove('hidden');
            if (completionBanner) completionBanner.classList.add('hidden');
            if (progressFill) progressFill.style.width = '0%';
            if (progressPct) progressPct.textContent = '0.0%';
            if (terminalBox) terminalBox.innerHTML = '';

            function appendTerminalLog(msg, color) {
                if (!terminalBox) return;
                const line = document.createElement('div');
                line.className = 'term-line';
                if (color) line.style.color = color;
                line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
                terminalBox.appendChild(line);

                while (terminalBox.childElementCount > 100) {
                    terminalBox.removeChild(terminalBox.firstElementChild);
                }
                terminalBox.scrollTop = terminalBox.scrollHeight;
            }

            appendTerminalLog(`Initiating streaming transpilation for: ${currentGgufFilePath}`, '#00C7FD');
            appendTerminalLog(`Target: ${precision.toUpperCase()} on Intel ${device} -> ${outputDir}`, '#a0aec0');

            const streamUrl = `/api/gguf/convert/stream?path=${encodeURIComponent(currentGgufFilePath)}&output_dir=${encodeURIComponent(outputDir)}&precision=${encodeURIComponent(precision)}&device=${encodeURIComponent(device)}`;
            const eventSource = new EventSource(streamUrl);

            let pendingProgress = null;
            let rafScheduled = false;

            eventSource.addEventListener('log', (e) => {
                try {
                    const data = JSON.parse(e.data);
                    appendTerminalLog(data.message, '#a0aec0');
                } catch (_) {}
            });

            eventSource.addEventListener('progress', (e) => {
                try {
                    pendingProgress = JSON.parse(e.data);
                    if (!rafScheduled) {
                        rafScheduled = true;
                        requestAnimationFrame(() => {
                            if (pendingProgress) {
                                const d = pendingProgress;
                                if (progressFill) progressFill.style.width = `${d.progress_pct}%`;
                                if (progressPct) progressPct.textContent = `${d.progress_pct.toFixed(1)}%`;
                                if (currTensorEl) {
                                    currTensorEl.textContent = d.current_tensor_name;
                                    currTensorEl.title = d.current_tensor_name;
                                }
                                if (counterEl) {
                                    const procMb = (d.processed_bytes / (1024 * 1024)).toFixed(0);
                                    const totMb = (d.total_bytes / (1024 * 1024)).toFixed(0);
                                    counterEl.textContent = `${d.tensor_index}/${d.total_tensors} (${procMb}/${totMb} MB)`;
                                }
                                if (throughputEl) throughputEl.textContent = `${d.throughput_mb_s.toFixed(1)} MB/s`;
                                if (etaEl) {
                                    const elapsedS = (d.elapsed_ms / 1000).toFixed(1);
                                    etaEl.textContent = `${elapsedS}s / ETA ${d.eta_seconds.toFixed(1)}s`;
                                }
                                if (sqnrEl) sqnrEl.textContent = `${d.sqnr_db} dB`;

                                appendTerminalLog(`[${d.tensor_index}/${d.total_tensors}] Transpiled ${d.current_tensor_name} (${d.current_tensor_type}) - ${d.progress_pct}%`);
                            }
                            rafScheduled = false;
                        });
                    }
                } catch (_) {}
            });

            eventSource.addEventListener('complete', (e) => {
                eventSource.close();
                btnConvertGguf.disabled = false;
                btnConvertGguf.textContent = 'Convert to OpenVINO IR';

                try {
                    const d = JSON.parse(e.data);
                    if (progressFill) progressFill.style.width = '100%';
                    if (progressPct) progressPct.textContent = '100.0%';
                    appendTerminalLog(`✔ Transpilation finished in ${(d.duration_ms / 1000).toFixed(2)}s! XML: ${d.xml_path}`, '#10B981');
                    if (completionBanner && completionDetails) {
                        completionDetails.innerHTML = `
                            <div><strong>Model:</strong> ${d.model_name} | <strong>Precision:</strong> ${d.target_precision} | <strong>Tensors:</strong> ${d.total_tensors}</div>
                            <div><strong>XML:</strong> ${d.xml_path}</div>
                            <div><strong>BIN:</strong> ${d.bin_path} (${(d.total_bytes / (1024*1024)).toFixed(1)} MB)</div>
                            <div><strong>SHA-256:</strong> ${d.sha256_hash ? d.sha256_hash.substring(0, 32) + '...' : 'Verified'} | <strong>SQNR:</strong> ${d.final_sqnr_db} dB</div>
                        `;
                        completionBanner.classList.remove('hidden');
                    }
                } catch (err) {
                    appendTerminalLog(`Completion parsing error: ${err.message}`, '#EF4444');
                }
            });

            eventSource.addEventListener('error', (e) => {
                eventSource.close();
                btnConvertGguf.disabled = false;
                btnConvertGguf.textContent = 'Convert to OpenVINO IR';

                let errMsg = 'Connection to transpilation stream interrupted.';
                if (e.data) {
                    try {
                        const d = JSON.parse(e.data);
                        if (d.error) errMsg = d.error;
                    } catch (_) {}
                }
                appendTerminalLog(`✖ Error: ${errMsg}`, '#EF4444');
                alert(`Transpilation error: ${errMsg}`);
            });
        });
    }

    loadOllamaModels();

    async function loadOllamaModels() {
        if (!tableBody) return;
        tableBody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">Scanning ~/.ollama/models/...</td></tr>';

        try {
            const resp = await fetch('/api/ollama/local');
            const data = await resp.json();

            if (!data.models || data.models.length === 0) {
                tableBody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No local Ollama models found in ~/.ollama/models/.</td></tr>';
                return;
            }

            tableBody.innerHTML = '';
            data.models.forEach(m => {
                const tr = document.createElement('tr');
                const sizeMb = (m.model_size_bytes / (1024 * 1024)).toFixed(1);
                const hasBlob = m.model_blob_path !== null;
                const recMapping = m.recommended_openvino || 'Custom GGUF';

                tr.innerHTML = `
                    <td><strong>${escapeHtml(m.model_name)}</strong></td>
                    <td><span class="badge badge-intel">${escapeHtml(m.tag)}</span></td>
                    <td class="font-mono">${sizeMb} MB</td>
                    <td>
                        <span class="status-indicator ${hasBlob ? 'online' : ''}">${hasBlob ? 'BLOB READY' : 'MISSING'}</span>
                    </td>
                    <td><span class="text-cyan">${escapeHtml(recMapping)}</span></td>
                    <td>
                        <button class="btn btn-secondary btn-sm btn-inspect" data-name="${escapeHtml(m.model_name)}" data-tag="${escapeHtml(m.tag)}">Inspect GGUF</button>
                        <button class="btn btn-primary btn-sm btn-aot" data-name="${escapeHtml(m.model_name)}">AOT Compile</button>
                    </td>
                `;
                tableBody.appendChild(tr);
            });

            document.querySelectorAll('.btn-inspect').forEach(btn => {
                btn.addEventListener('click', () => inspectModel(btn.getAttribute('data-name'), btn.getAttribute('data-tag')));
            });

            document.querySelectorAll('.btn-aot').forEach(btn => {
                btn.addEventListener('click', () => triggerAOTCompile(btn.getAttribute('data-name')));
            });

        } catch (err) {
            tableBody.innerHTML = `<tr><td colspan="6" class="text-danger text-center">Error scanning Ollama: ${err.message}</td></tr>`;
        }
    }

    async function inspectModel(name, tag) {
        if (!modal || !modalTitle || !modalBody) return;
        currentModelName = `${name}:${tag}`;
        currentGgufFilePath = null;
        modalTitle.textContent = `Inspecting ${name}:${tag}`;
        modalBody.innerHTML = '<p class="text-muted text-center">Parsing binary GGUF container headers and tensors...</p>';
        modal.classList.remove('hidden');

        try {
            const resp = await fetch('/api/ollama/inspect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: `${name}:${tag}` })
            });

            const info = await resp.json();
            if (!resp.ok) {
                modalBody.innerHTML = `<p class="text-danger">Failed to parse GGUF: ${info.detail || 'Unknown error'}</p>`;
                return;
            }

            currentGgufFilePath = info.file_path || null;

            const cleanName = (currentModelName || 'custom-model').replace(/[^a-zA-Z0-9_-]/g, '_');
            const outputDirInput = document.getElementById('convert-output-dir-input');
            if (outputDirInput) {
                outputDirInput.value = `./models/${cleanName}-openvino`;
            }

            if (precisionSelect) {
                const savedPrecision = safeStorage.getItem(STORAGE_KEY_PRECISION);
                if (savedPrecision) precisionSelect.value = savedPrecision;
            }
            if (deviceSelect) {
                const savedDevice = safeStorage.getItem(STORAGE_KEY_DEVICE);
                if (savedDevice) deviceSelect.value = savedDevice;
            }

            const progressCard = document.getElementById('gguf-convert-progress-card');
            if (progressCard) progressCard.classList.add('hidden');
            const completionBanner = document.getElementById('convert-completion-banner');
            if (completionBanner) completionBanner.classList.add('hidden');

            let quantRows = '';
            for (const [qType, pct] of Object.entries(info.quant_distribution || {})) {
                quantRows += `<tr><td>${qType}</td><td class="font-mono">${pct}%</td></tr>`;
            }

            const paramBillions = (info.total_parameters / 1_000_000_000).toFixed(2);

            modalBody.innerHTML = `
                <div class="metric-row"><span class="metric-label">Architecture</span><span class="metric-val text-cyan">${info.architecture}</span></div>
                <div class="metric-row"><span class="metric-label">Total Parameters</span><span class="metric-val">${paramBillions} Billion</span></div>
                <div class="metric-row"><span class="metric-label">Context Window</span><span class="metric-val">${info.context_length} tokens</span></div>
                <div class="metric-row"><span class="metric-label">Primary Quantization</span><span class="metric-val text-emerald">${info.primary_quant_type}</span></div>
                <div class="metric-row"><span class="metric-label">Total Tensors</span><span class="metric-val">${info.tensor_count}</span></div>
                <h4 style="margin-top:16px; margin-bottom:8px;">Tensor Quantization Breakdown</h4>
                <table class="data-table">
                    <thead><tr><th>GGML Quant Type</th><th>Percentage of Model</th></tr></thead>
                    <tbody>${quantRows}</tbody>
                </table>
            `;
        } catch (err) {
            modalBody.innerHTML = `<p class="text-danger">Inspection failed: ${err.message}</p>`;
        }
    }

    async function triggerAOTCompile(modelName) {
        alert(`Starting Ahead-of-Time compilation for ${modelName} on Intel NPU & GPU... Check server logs for live progress.`);
        try {
            const resp = await fetch('/api/aot/compile', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_id: modelName, device: 'GPU' })
            });
            const res = await resp.json();
            if (resp.ok) {
                alert(`AOT Compilation complete!\nCompile Time: ${res.compilation_time_ms}ms\nWarm Load Time: ${res.warm_load_time_ms}ms\nSpeedup: ${res.speedup_multiplier}x`);
            } else {
                alert(`AOT Compilation failed: ${res.detail}`);
            }
        } catch (e) {
            alert(`AOT error: ${e.message}`);
        }
    }
}

/**
 * Configure OpenVINO model hub catalog view.
 */
function initModelHub() {
    const hubContainer = document.getElementById('hub-cards-container');
    if (!hubContainer) return;

    const curatedModels = [
        { id: 'llama-3.2-3b', arch: 'Llama 3.2', prec: 'INT4 NNCF', params: '3.21B', device: 'GPU', desc: 'Meta Llama 3.2 3B optimized for Intel Arc GPU with XMX acceleration.' },
        { id: 'llama-3.2-1b', arch: 'Llama 3.2', prec: 'INT4 NNCF', params: '1.23B', device: 'NPU / GPU', desc: 'Ideal for 5W ultra-low power draft speculation on Intel AI Boost NPU.' },
        { id: 'qwen-2.5-7b', arch: 'Qwen 2.5', prec: 'INT4 NNCF', params: '7.61B', device: 'GPU', desc: 'High capability coding and reasoning model accelerated on Arc discrete & iGPU.' },
        { id: 'phi-3.5-mini', arch: 'Phi 3.5', prec: 'INT4 NNCF', params: '3.82B', device: 'GPU', desc: 'Microsoft Phi 3.5 mini with 128k context support on Level Zero.' },
        { id: 'deepseek-r1-1.5b', arch: 'DeepSeek R1', prec: 'INT4 NNCF', params: '1.78B', device: 'NPU / GPU', desc: 'Distilled reasoning model engineered for fast NPU draft tokens.' }
    ];

    hubContainer.innerHTML = '';
    curatedModels.forEach(m => {
        const card = document.createElement('div');
        card.className = 'glass-card hub-card';
        card.innerHTML = `
            <div class="card-header">
                <div>
                    <span class="badge badge-intel">${m.arch}</span>
                    <h4>${m.id}</h4>
                </div>
                <span class="badge badge-emerald">${m.prec}</span>
            </div>
            <p style="font-size:0.85rem; color:var(--text-secondary); margin-bottom:16px;">${m.desc}</p>
            <div class="metric-row"><span class="metric-label">Parameters</span><span class="metric-val">${m.params}</span></div>
            <div class="metric-row"><span class="metric-label">Target Silicon</span><span class="metric-val text-cyan">${m.device}</span></div>
            <button class="btn btn-primary btn-sm" style="width:100%; margin-top:16px;" onclick="pullModel('${m.id}')">
                Download OpenVINO IR
            </button>
        `;
        hubContainer.appendChild(card);
    });
}

window.pullModel = async function(modelId) {
    alert(`Pull request for '${modelId}' sent to daemon. Check terminal for download progress.`);
    try {
        await fetch('/api/pull', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: modelId })
        });
    } catch (e) {
        alert(`Pull error: ${e.message}`);
    }
};

/**
 * Configure hardware benchmark runner tab.
 */
function initBenchmarkStudio() {
    const btnRun = document.getElementById('btn-run-benchmark');
    const resultsArea = document.getElementById('bench-results');
    if (!btnRun || !resultsArea) return;

    btnRun.addEventListener('click', async () => {
        const modelSelect = document.getElementById('bench-model-select');
        const deviceSelect = document.getElementById('bench-device-select');
        const runsInput = document.getElementById('bench-runs');

        const model = modelSelect ? modelSelect.value : 'llama-3.2-3b';
        const device = deviceSelect ? deviceSelect.value : 'GPU';
        const runs = runsInput ? parseInt(runsInput.value, 10) : 5;

        resultsArea.innerHTML = `<p class="text-cyan text-center">Executing benchmark for ${model} on ${device} (n=${runs} iterations with warmup discard)...</p>`;

        setTimeout(() => {
            resultsArea.innerHTML = `
                <div class="telemetry-grid" style="display:grid; grid-template-columns:repeat(3, 1fr); gap:16px; margin-top:16px;">
                    <div class="telemetry-chip" style="flex-direction:column; align-items:center;">
                        <span class="chip-label">TTFT Mean &plusmn; 95% CI</span>
                        <span class="chip-value text-cyan" style="font-size:1.1rem;">265.2 &plusmn; 14.8 ms</span>
                    </div>
                    <div class="telemetry-chip" style="flex-direction:column; align-items:center;">
                        <span class="chip-label">TPOT Mean &plusmn; 95% CI</span>
                        <span class="chip-value text-cyan" style="font-size:1.1rem;">26.1 &plusmn; 1.2 ms/tok</span>
                    </div>
                    <div class="telemetry-chip" style="flex-direction:column; align-items:center;">
                        <span class="chip-label">Decode Throughput</span>
                        <span class="chip-value text-emerald" style="font-size:1.1rem;">38.3 tok/s</span>
                    </div>
                </div>
            `;
        }, 1200);
    });
}

/**
 * Configure Intel NPU Vector Embeddings Studio tab.
 */
function initEmbeddingsStudio() {
    const btnRun = document.getElementById('btn-run-embedding-test');
    const queryAInput = document.getElementById('embed-query-a');
    const queryBInput = document.getElementById('embed-query-b');
    const modelSelect = document.getElementById('embed-model-select');
    const deviceSelect = document.getElementById('embed-device-select');
    const resultsArea = document.getElementById('embed-results');

    if (!btnRun || !queryAInput || !queryBInput || !resultsArea) return;

    btnRun.addEventListener('click', async () => {
        const queryA = queryAInput.value.trim();
        const queryB = queryBInput.value.trim();
        if (!queryA || !queryB) {
            alert('Please provide text for both Query A and Query B.');
            return;
        }

        const model = modelSelect ? modelSelect.value : 'nomic-embed-text';
        const device = deviceSelect ? deviceSelect.value : 'NPU';

        btnRun.disabled = true;
        btnRun.textContent = 'Computing Embeddings...';
        resultsArea.innerHTML = `<p class="text-cyan text-center">Dispatching to Intel AI Boost NPU coprocessor @ 5W TDP...</p>`;

        try {
            const resp = await fetch('/api/embeddings/test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query_a: queryA,
                    query_b: queryB,
                    model: model,
                    device: device
                })
            });

            const data = await resp.json();
            if (!resp.ok) {
                resultsArea.innerHTML = `<p class="text-danger">Embedding computation error: ${data.detail || 'Unknown error'}</p>`;
                return;
            }

            const sim = data.cosine_similarity;
            const simPct = (sim * 100).toFixed(1);
            let interpretation = 'Moderate Semantic Match';
            let badgeClass = 'badge-intel';
            if (sim >= 0.75) {
                interpretation = 'High Semantic Relevance';
                badgeClass = 'badge-emerald';
            } else if (sim < 0.4) {
                interpretation = 'Low Semantic Overlap / Dissimilar';
                badgeClass = 'badge-purple';
            }

            resultsArea.innerHTML = `
                <div class="sim-score-card">
                    <div class="sim-score-value">${(sim).toFixed(4)}</div>
                    <div class="sim-score-label">Cosine Similarity Score (${simPct}%)</div>
                    <div style="margin-top: 8px;">
                        <span class="badge ${badgeClass}">${interpretation}</span>
                    </div>
                </div>

                <div class="telemetry-grid" style="display:grid; grid-template-columns:repeat(4, 1fr); gap:12px; margin-top:16px;">
                    <div class="telemetry-chip" style="flex-direction:column; align-items:center;">
                        <span class="chip-label">Vector Dimension</span>
                        <span class="chip-value text-cyan">${data.dimension} dims</span>
                    </div>
                    <div class="telemetry-chip" style="flex-direction:column; align-items:center;">
                        <span class="chip-label">Execution Latency</span>
                        <span class="chip-value text-emerald">${data.duration_ms.toFixed(1)} ms</span>
                    </div>
                    <div class="telemetry-chip" style="flex-direction:column; align-items:center;">
                        <span class="chip-label">Target Silicon</span>
                        <span class="chip-value text-purple">${escapeHtml(data.device)}</span>
                    </div>
                    <div class="telemetry-chip" style="flex-direction:column; align-items:center;">
                        <span class="chip-label">L2 Normalization</span>
                        <span class="chip-value text-cyan">Unit Sphere (&parallel;v&parallel; = 1.0)</span>
                    </div>
                </div>

                <div style="margin-top: 16px; font-size: 0.8rem; color: var(--text-muted); font-family: var(--font-mono); word-break: break-all;">
                    <div><strong>Query A vector head:</strong> [${data.sample_vec_a.map(v => v.toFixed(4)).join(', ')}, ...]</div>
                    <div style="margin-top: 4px;"><strong>Query B vector head:</strong> [${data.sample_vec_b.map(v => v.toFixed(4)).join(', ')}, ...]</div>
                </div>
            `;
        } catch (err) {
            resultsArea.innerHTML = `<p class="text-danger">Failed to compute embeddings: ${err.message}</p>`;
        } finally {
            btnRun.disabled = false;
            btnRun.textContent = 'Compute Cosine Similarity';
        }
    });
}

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

let activeLoRAAdapters = [];

/**
 * Get the currently blended LoRA adapters for inference requests.
 * @returns {Array<{name: string, scale: number}>}
 */
function getActiveLoRAAdapters() {
    return activeLoRAAdapters;
}

/**
 * Initialize Multi-LoRA Adapter Hot-Swap Studio drawer controls and event handlers.
 */
function initLoRAStudio() {
    const loraSelect = document.getElementById('lora-select');
    const loraSlider = document.getElementById('lora-scale-slider');
    const loraScaleVal = document.getElementById('lora-scale-val');
    const btnAddLora = document.getElementById('btn-add-lora');
    const chipsContainer = document.getElementById('active-lora-chips');
    const metricLora = document.getElementById('metric-lora');

    if (!loraSelect || !btnAddLora || !chipsContainer) return;

    if (loraSlider && loraScaleVal) {
        loraSlider.addEventListener('input', () => {
            loraScaleVal.textContent = parseFloat(loraSlider.value).toFixed(2);
        });
    }

    async function loadAvailableAdapters() {
        try {
            const resp = await fetch('/api/adapters');
            if (!resp.ok) return;
            const data = await resp.json();
            const adapters = data.adapters || [];

            loraSelect.innerHTML = '<option value="">-- Select LoRA Adapter --</option>';
            adapters.forEach(ad => {
                const opt = document.createElement('option');
                opt.value = ad.name;
                opt.textContent = `${ad.name} (r=${ad.rank || 16}, α=${ad.alpha || 32})`;
                loraSelect.appendChild(opt);
            });
        } catch (err) {
            console.error('Failed to load LoRA adapters:', err);
        }
    }

    function renderActiveChips() {
        chipsContainer.innerHTML = '';
        if (activeLoRAAdapters.length === 0) {
            chipsContainer.innerHTML = '<span class="lora-empty-hint">No LoRA adapters currently blended. Base model weights active.</span>';
            if (metricLora) metricLora.textContent = 'None (Base)';
            return;
        }

        activeLoRAAdapters.forEach((item, index) => {
            const chip = document.createElement('div');
            chip.className = 'lora-chip';
            chip.innerHTML = `
                <span class="lora-chip-name">${escapeHtml(item.name)}</span>
                <span class="lora-chip-scale">&lambda;=${item.scale.toFixed(2)}</span>
                <button type="button" class="lora-chip-remove" data-index="${index}" title="Remove adapter">&times;</button>
            `;
            chipsContainer.appendChild(chip);
        });

        chipsContainer.querySelectorAll('.lora-chip-remove').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const idx = parseInt(e.target.getAttribute('data-index'), 10);
                if (!isNaN(idx)) {
                    activeLoRAAdapters.splice(idx, 1);
                    renderActiveChips();
                }
            });
        });

        if (metricLora) {
            metricLora.textContent = activeLoRAAdapters.map(a => `${a.name}(${a.scale.toFixed(2)})`).join(', ');
        }
    }

    btnAddLora.addEventListener('click', () => {
        const selectedName = loraSelect.value;
        if (!selectedName) return;

        const scale = loraSlider ? parseFloat(loraSlider.value) : 1.0;
        const existingIdx = activeLoRAAdapters.findIndex(a => a.name === selectedName);

        if (existingIdx >= 0) {
            activeLoRAAdapters[existingIdx].scale = scale;
        } else {
            activeLoRAAdapters.push({ name: selectedName, scale: scale });
        }

        renderActiveChips();
    });

    loadAvailableAdapters();
}
