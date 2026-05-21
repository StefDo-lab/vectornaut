// State management
let isMock = false;
let lossChart = null;
let velocityChart = null;
let lastRunData = null;

// Initialize elements
document.addEventListener("DOMContentLoaded", () => {
    applySampleQuery();
});

function applySampleQuery() {
    const select = document.getElementById("query-select");
    const textarea = document.getElementById("query-input");
    if (select.value) {
        textarea.value = select.value;
    }
}

function setMode(mockVal) {
    isMock = mockVal;
    const btnLive = document.getElementById("mode-live");
    const btnMock = document.getElementById("mode-mock");
    
    if (mockVal) {
        btnLive.classList.remove("active");
        btnMock.classList.add("active");
        writeLog("[SYSTEM] Switched to OFFLINE MOCK mode. Execution will be instantaneous and cost no tokens.", "info-log");
    } else {
        btnMock.classList.remove("active");
        btnLive.classList.add("active");
        writeLog("[SYSTEM] Switched to LIVE API mode. Gemini-3.5-flash will perform bionic mining and audits.", "info-log");
    }
}

function writeLog(text, className = "system-log") {
    const consoleBox = document.getElementById("console-output");
    const logSpan = document.createElement("span");
    logSpan.className = className;
    logSpan.innerText = text;
    consoleBox.appendChild(logSpan);
    consoleBox.scrollTop = consoleBox.scrollHeight;
}

function openTab(evt, tabId) {
    const contents = document.getElementsByClassName("tab-content");
    for (let content of contents) {
        content.classList.remove("active");
    }
    const links = document.getElementsByClassName("tab-link");
    for (let link of links) {
        link.classList.remove("active");
    }
    document.getElementById(tabId).classList.add("active");
    evt.currentTarget.classList.add("active");
}

async function runDiscoveryLoop(overrideParams = null) {
    const query = document.getElementById("query-input").value.trim();
    const epochs = parseInt(document.getElementById("epochs-input").value);
    const runBtn = document.getElementById("run-btn");
    
    if (!query) {
        alert("Please specify a design request!");
        return;
    }

    // UI resets
    runBtn.disabled = true;
    runBtn.querySelector(".btn-text").innerText = "Processing...";
    
    resetStepper();
    clearResults();
    
    writeLog(`[SYSTEM] Starting discovery loop for: "${query}"`, "info-log");
    if (overrideParams) {
        writeLog(`[SYSTEM] Injecting parameter overrides: ${JSON.stringify(overrideParams)}`, "info-log");
    }
    writeLog(`[SYSTEM] Requesting ${epochs} epochs for PINN-lite solver...`);

    // Step 1: Miner Active
    setStepState("step-miner", "active", "Exploring semantic spaces...");

    try {
        const bodyPayload = { query, epochs, is_mock: isMock };
        if (overrideParams) {
            bodyPayload.override_parameters = overrideParams;
        }
        
        const response = await fetch("/api/run", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(bodyPayload)
        });
        
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || "Server error running pipeline.");
        }
        
        const data = await response.json();
        
        const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

        // Orchestrate simulated delays to make the agent pipeline interactive to watch
        // Miner Complete -> Auditor Start
        await delay(1500);
        setStepState("step-miner", "completed", "Concept discovered!");
        writeLog(`[MINER] Concept discovered: "${data.miner.design_name}"`, "success-log");
        writeLog(`[MINER] Inspiration: ${data.miner.inspiration_source}`);
        writeLog(`[MINER] Domain: ${data.miner.domain}`);
        writeLog(`[MINER] Proposed parameters extracted: ${data.miner.parameters.length} variables.`);
        
        setStepState("step-auditor", "active", "Sanity checking variables...");
        
        // Auditor Complete -> Simulator Start
        await delay(1500);
        setStepState("step-auditor", "completed", "Parameters verified!");
        writeLog(`[AUDITOR] Audit completed successfully: ${data.auditor.audit_passed ? 'PASSED' : 'FAILED'}`, "success-log");
        writeLog(`[AUDITOR] Notes: "${data.auditor.audit_notes}"`);
        writeLog(`[AUDITOR] Solver selected: ${data.auditor.solver_method.toUpperCase()}`);
        writeLog(`[AUDITOR] Derived coefficient = ${data.auditor.simulation_coefficient.toFixed(6)}`);
        
        setStepState("step-simulator", "active", data.simulator.solver_method === "pinn" ? "Training PINN neural network..." : `Solving via ${data.simulator.solver_method} solver...`);
        writeLog(`[SIMULATOR] Launching physics-solver dispatcher...`);
        writeLog(`[SIMULATOR] Solving equation: ${data.miner.governing_equation}`);
        
        // Simulator Complete -> Render Results
        await delay(1800);
        setStepState("step-simulator", "completed", "Simulation complete!");
        if (data.simulator.solver_method === "pinn") {
            writeLog(`[SIMULATOR] PyTorch PINN converged. Final Loss: ${data.simulator.final_loss.toExponential(4)}`, "success-log");
        } else {
            writeLog(`[SIMULATOR] Solver evaluated successfully. Relative error: ${data.simulator.relative_error.toExponential(4)}`, "success-log");
        }
        writeLog(`[SIMULATOR] ${data.auditor.ui_metadata.performance_gain.label}: ${data.simulator.performance_gain_pct.toFixed(2)}%`, "success-log");
        writeLog(`[SYSTEM] Compilation of run report completed successfully!`, "info-log");
        
        renderResults(data);
        
        runBtn.disabled = false;
        runBtn.querySelector(".btn-text").innerText = "Run Discovery Loop";

    } catch (e) {
        writeLog(`[ERROR] Pipeline failed: ${e.message}`, "error-log");
        setStepState("step-miner", "error", "Failed");
        setStepState("step-auditor", "error", "Failed");
        setStepState("step-simulator", "error", "Failed");
        runBtn.disabled = false;
        runBtn.querySelector(".btn-text").innerText = "Run Discovery Loop";
    }
}

function resetStepper() {
    const steps = ["step-miner", "step-auditor", "step-simulator"];
    steps.forEach(id => {
        const el = document.getElementById(id);
        el.className = "step";
        el.querySelector(".step-status").innerText = "Idle";
    });
}

function setStepState(stepId, state, statusText) {
    const el = document.getElementById(stepId);
    el.className = `step ${state}`;
    el.querySelector(".step-status").innerText = statusText;
}

function clearResults() {
    document.querySelectorAll(".empty-state").forEach(el => el.classList.remove("hidden"));
    document.querySelectorAll(".overview-grid, .table-container, .simulation-view, #raw-json-report, #markdown-report, #report-actions-container").forEach(el => el.classList.add("hidden"));
    const rawReport = document.getElementById("raw-json-report");
    if (rawReport) {
        rawReport.style.display = "none";
    }
}

function renderResults(data) {
    lastRunData = data;
    
    // 1. Render Markdown report if available
    try {
        if (data.report_md && typeof marked !== 'undefined') {
            const mdEl = document.getElementById("markdown-report");
            if (mdEl) {
                mdEl.innerHTML = marked.parse(data.report_md);
                mdEl.classList.remove("hidden");
            }
        }
    } catch (mdErr) {
        console.error("Failed to render markdown report:", mdErr);
    }

    // 2. Render raw JSON report in background (remains hidden until toggled)
    try {
        const rawReportEl = document.getElementById("raw-json-report");
        if (rawReportEl) {
            rawReportEl.innerText = JSON.stringify(data, null, 4);
        }
    } catch (rawErr) {
        console.error("Failed to render raw report:", rawErr);
    }

    // Hide empty states, reveal elements
    document.querySelectorAll(".empty-state").forEach(el => el.classList.add("hidden"));
    document.querySelectorAll(".overview-grid, .table-container, .simulation-view, #markdown-report, #report-actions-container").forEach(el => el.classList.remove("hidden"));
    
    // Normalize properties dynamically to support old/new keys
    const minerObj = data.miner || data.miner_stage || {};
    const auditorObj = data.auditor || data.auditor_stage || {};
    const sim = data.simulator || data.simulator_stage || {};
    
    const uiMeta = auditorObj.ui_metadata || {
        domain_name: minerObj.domain || "Scientific Domain",
        independent_var: { label: "Coordinate", unit: "m" },
        dependent_var: { label: "Field", unit: "" },
        primary_metric: { label: "Primary Metric" },
        reference_metric: { label: "Reference Metric" },
        performance_gain: { label: "Performance Gain" }
    };

    // Helper for safe numeric conversion
    const getSafeNumber = (val) => {
        if (typeof val === 'number' && !isNaN(val)) return val;
        if (typeof val === 'string') {
            const parsed = parseFloat(val);
            if (!isNaN(parsed)) return parsed;
        }
        return null;
    };

    // Overview tab
    document.getElementById("concept-name").innerText = minerObj.design_name || "Unknown Design";
    document.getElementById("inspiration-source").innerText = minerObj.inspiration_source || "N/A";
    document.getElementById("physical-domain").innerText = minerObj.domain || "N/A";
    document.getElementById("physical-mechanism").innerText = minerObj.physical_mechanism || "N/A";
    
    const simCoeff = getSafeNumber(auditorObj.simulation_coefficient);
    document.getElementById("derived-coefficient").innerText = simCoeff !== null ? simCoeff.toFixed(6) : "-";
    document.getElementById("derived-coefficient-label").innerText = `Derived Coefficient (for ${uiMeta.domain_name})`;
    
    // Parameters tab
    const tbody = document.getElementById("params-table-body");
    tbody.innerHTML = "";
    
    // Miner parameters list
    const minerParams = minerObj.parameters || [];
    const auditedDict = auditorObj.audited_parameters_dict || {};
    
    minerParams.forEach(p => {
        const row = document.createElement("tr");
        const valNum = getSafeNumber(p.value);
        const minNum = getSafeNumber(p.min_bound);
        const maxNum = getSafeNumber(p.max_bound);
        
        const auditedValRaw = auditedDict[p.name];
        const auditedValNum = getSafeNumber(auditedValRaw);
        
        const displayVal = valNum !== null ? valNum.toExponential(4) : (p.value || "N/A");
        const displayAudited = auditedValNum !== null ? auditedValNum.toExponential(4) : (auditedValRaw !== undefined ? auditedValRaw : "N/A");
        const displayBounds = (minNum !== null && maxNum !== null) ? `[${minNum.toExponential(2)}, ${maxNum.toExponential(2)}]` : "N/A";

        row.innerHTML = `
            <td><strong>${p.name}</strong></td>
            <td>${displayVal}</td>
            <td><span class="val-highlight">${displayAudited}</span></td>
            <td>${displayBounds}</td>
            <td class="desc-text">${p.justification || ""}</td>
        `;
        tbody.appendChild(row);
    });
    
    // Dimensionless cards
    const dimContainer = document.getElementById("dimensionless-container");
    dimContainer.innerHTML = "";
    
    const dimList = auditorObj.dimensionless_numbers || [];
    dimList.forEach(item => {
        const card = document.createElement("div");
        card.className = "info-card";
        const itemVal = getSafeNumber(item.value);
        const displayItemVal = itemVal !== null ? itemVal.toLocaleString(undefined, {minimumFractionDigits: 1, maximumFractionDigits: 4}) : (item.value || "-");
        card.innerHTML = `
            <span class="label">${item.name}</span>
            <p class="val-highlight">${displayItemVal}</p>
        `;
        dimContainer.appendChild(card);
    });
    
    // Simulation Tab labels & metrics
    document.getElementById("metric-gain-label").innerText = uiMeta.performance_gain?.label || "Performance Gain";
    document.getElementById("metric-primary-label").innerText = uiMeta.primary_metric?.label || "Primary Metric";
    document.getElementById("metric-ref-label").innerText = uiMeta.reference_metric?.label || "Reference Metric";

    const performanceGain = getSafeNumber(sim.performance_gain_pct);
    const primaryMetricVal = getSafeNumber(sim.primary_metric_value ?? sim.wall_shear_stress_pinn);
    const refMetricVal = getSafeNumber(sim.reference_metric_value ?? sim.wall_shear_stress_analytical);
    const finalLossVal = getSafeNumber(sim.final_loss);

    document.getElementById("metric-drag-reduction").innerText = performanceGain !== null ? `${performanceGain.toFixed(2)}%` : "-";
    document.getElementById("metric-wss-pinn").innerText = primaryMetricVal !== null ? primaryMetricVal.toFixed(4) : "-";
    document.getElementById("metric-wss-analytical").innerText = refMetricVal !== null ? refMetricVal.toFixed(4) : "-";
    
    // Toggle PINN loss card and chart visibility
    const isPinn = sim.solver_method === "pinn";
    const lossHistory = sim.loss_history || [];
    const chartsGrid = document.querySelector(".charts-grid");
    
    if (isPinn && lossHistory.length > 0) {
        document.getElementById("card-pinn-loss").style.display = "block";
        document.getElementById("container-loss-chart").style.display = "block";
        document.getElementById("metric-loss").innerText = finalLossVal !== null ? finalLossVal.toExponential(4) : "-";
        if (chartsGrid) chartsGrid.classList.remove("single-chart");
    } else {
        document.getElementById("card-pinn-loss").style.display = "none";
        document.getElementById("container-loss-chart").style.display = "none";
        if (chartsGrid) chartsGrid.classList.add("single-chart");
    }

    // Dynamic plot title
    const depVarLabel = uiMeta.dependent_var?.label || "Field";
    document.getElementById("field-chart-title").innerText = `${depVarLabel} Field (Primary vs Reference)`;

    // Toggle 1D vs 2D visualization containers
    const is2D = sim.sample_points && sim.sample_points.length > 0 && Array.isArray(sim.sample_points[0]);
    const fieldChartContainer = document.getElementById("container-field-chart");
    const heatmapContainer = document.getElementById("container-heatmap");
    
    if (is2D) {
        if (fieldChartContainer) fieldChartContainer.style.display = "none";
        if (heatmapContainer) heatmapContainer.style.display = "block";
    } else {
        if (fieldChartContainer) fieldChartContainer.style.display = "block";
        if (heatmapContainer) heatmapContainer.style.display = "none";
    }

    // Build Charts (wrapped in try-catch so it won't break the page if Chart.js is offline)
    try {
        buildCharts(sim, uiMeta);
    } catch (chartError) {
        console.error("Error building charts:", chartError);
        writeLog(`[ERROR] Visualizations failed: ${chartError.message}`, "error-log");
    }

    // Update slider and display value to actual epochs trained
    if (sim && sim.epochs_trained) {
        const slider = document.getElementById("epochs-input");
        const label = document.getElementById("epochs-val");
        if (slider && label) {
            if (sim.epochs_trained > parseInt(slider.max)) {
                slider.max = sim.epochs_trained;
            }
            slider.value = sim.epochs_trained;
            label.innerText = sim.epochs_trained;
        }
    }

    // 3. Initialize/Update Bionic Assistant Tab
    try {
        const assistantEmpty = document.getElementById("assistant-empty-state");
        const assistantWrapper = document.getElementById("assistant-wrapper");
        if (assistantEmpty) assistantEmpty.classList.add("hidden");
        if (assistantWrapper) assistantWrapper.classList.remove("hidden");
        
        resetChat();
        updateAssistantTab(data);
    } catch (assistantErr) {
        console.error("Failed to update Bionic Assistant tab:", assistantErr);
    }
}

function buildCharts(sim, uiMeta) {
    const is2D = sim.sample_points && sim.sample_points.length > 0 && Array.isArray(sim.sample_points[0]);
    
    if (is2D) {
        renderHeatmap2D(sim, uiMeta);
    }

    if (typeof Chart === 'undefined') {
        console.warn("Chart.js is not loaded. Skipping chart rendering.");
        writeLog("[WARNING] Chart.js is not loaded. Charts will not be displayed.");
        return;
    }

    // 1. Loss Chart
    if (lossChart) {
        lossChart.destroy();
        lossChart = null;
    }
    
    const isPinn = sim.solver_method === "pinn";
    const lossHistory = sim.loss_history || [];
    
    if (isPinn && lossHistory.length > 0) {
        const lossCtx = document.getElementById("lossChart")?.getContext("2d");
        if (lossCtx) {
            const epochLabels = Array.from({length: lossHistory.length}, (_, i) => i + 1);
            
            lossChart = new Chart(lossCtx, {
                type: 'line',
                data: {
                    labels: epochLabels,
                    datasets: [{
                        label: 'Residual MSE Loss',
                        data: lossHistory,
                        borderColor: '#9d4edd',
                        backgroundColor: 'rgba(157, 78, 221, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        pointRadius: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            type: 'logarithmic',
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: { color: '#8e94a5' }
                        },
                        x: {
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: { color: '#8e94a5', maxTicksLimit: 10 }
                        }
                    },
                    plugins: {
                        legend: { display: false }
                    }
                }
            });
        }
    }
    
    // 2. Velocity/Field Chart (only if 1D)
    if (velocityChart) {
        velocityChart.destroy();
        velocityChart = null;
    }
    
    if (is2D) {
        return; // 2D visualization handled separately by Canvas
    }
    
    const velCanvas = document.getElementById("velocityChart");
    const samplePoints = sim.sample_points || sim.sample_points_y || [];
    const solutionPrimary = sim.solution_primary || sim.sample_points_u_pinn || [];
    const solutionReference = sim.solution_reference || sim.sample_points_u_analytical || [];
    
    if (velCanvas && samplePoints.length > 0) {
        const velCtx = velCanvas.getContext("2d");
        
        velocityChart = new Chart(velCtx, {
            type: 'line',
            data: {
                labels: samplePoints.map(x => (typeof x === 'number') ? x.toFixed(3) : x),
                datasets: [
                    {
                        label: `Primary Solver (${(sim.solver_method || '').toUpperCase()})`,
                        data: solutionPrimary,
                        borderColor: '#00f5d4',
                        borderWidth: 2.5,
                        pointRadius: 3,
                        pointBackgroundColor: '#00f5d4',
                        fill: false
                    },
                    {
                        label: `Reference Solver`,
                        data: solutionReference,
                        borderColor: 'rgba(255, 255, 255, 0.35)',
                        borderWidth: 1.5,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        title: { 
                            display: true, 
                            text: `${uiMeta.dependent_var?.label || 'Field'} (${uiMeta.dependent_var?.unit || ''})`, 
                            color: '#8e94a5',
                            font: { family: 'Outfit', size: 12 }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#8e94a5' }
                    },
                    x: {
                        title: { 
                            display: true, 
                            text: `${uiMeta.independent_var?.label || 'Coordinate'} (${uiMeta.independent_var?.unit || ''})`, 
                            color: '#8e94a5',
                            font: { family: 'Outfit', size: 12 }
                        },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#8e94a5' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#f1f3f9', font: { family: 'Outfit' } }
                    }
                }
            }
        });
    }
}

function getColorForValue(t) {
    t = Math.max(0, Math.min(1, t));
    
    // Thermal color map stops:
    // 0.0 -> deep blue (0, 10, 80)
    // 0.35 -> violet (115, 0, 199)
    // 0.7 -> orange/red (255, 84, 0)
    // 1.0 -> bright yellow (254, 228, 64)
    let r, g, b;
    if (t < 0.35) {
        const factor = t / 0.35;
        r = Math.round(0 + factor * 115);
        g = Math.round(10 - factor * 10);
        b = Math.round(80 + factor * 119);
    } else if (t < 0.7) {
        const factor = (t - 0.35) / 0.35;
        r = Math.round(115 + factor * 140);
        g = Math.round(0 + factor * 84);
        b = Math.round(199 - factor * 199);
    } else {
        const factor = (t - 0.7) / 0.3;
        r = Math.round(255 + factor * (-1));
        g = Math.round(84 + factor * 144);
        b = Math.round(0 + factor * 64);
    }
    return `rgb(${r}, ${g}, ${b})`;
}

function renderHeatmap2D(sim, uiMeta) {
    const canvas = document.getElementById("heatmapCanvas");
    if (!canvas) return;
    
    const ctx = canvas.getContext("2d");
    const solution = sim.solution_primary || [];
    const samplePoints = sim.sample_points || [];
    
    if (solution.length === 0) return;
    
    const minVal = Math.min(...solution);
    const maxVal = Math.max(...solution);
    const range = maxVal - minVal || 1.0;
    
    // Smooth bilinear interpolation using 20x20 offscreen canvas
    const offscreen = document.createElement("canvas");
    offscreen.width = 20;
    offscreen.height = 20;
    const oCtx = offscreen.getContext("2d");
    const imgData = oCtx.createImageData(20, 20);
    
    for (let i = 0; i < 20; i++) {
        for (let j = 0; j < 20; j++) {
            const idx = i * 20 + j;
            const val = solution[idx] !== undefined ? solution[idx] : minVal;
            const t = (val - minVal) / range;
            
            const colorStr = getColorForValue(t);
            const matches = colorStr.match(/\d+/g);
            const r = parseInt(matches[0]);
            const g = parseInt(matches[1]);
            const b = parseInt(matches[2]);
            
            const pixelX = i;
            const pixelY = 19 - j; // Flip Y for math coordinate space
            const offset = (pixelY * 20 + pixelX) * 4;
            
            imgData.data[offset] = r;
            imgData.data[offset+1] = g;
            imgData.data[offset+2] = b;
            imgData.data[offset+3] = 255;
        }
    }
    oCtx.putImageData(imgData, 0, 0);
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
    
    // Add Legend
    const legend = document.getElementById("heatmap-legend");
    if (legend) {
        legend.innerHTML = `
            <span>Min: ${minVal.toFixed(2)} ${uiMeta.dependent_var?.unit || ''}</span>
            <div style="width: 150px; height: 12px; background: linear-gradient(to right, rgb(0,10,80), rgb(115,0,199), rgb(255,84,0), rgb(254,228,64)); border-radius: 6px;"></div>
            <span>Max: ${maxVal.toFixed(2)} ${uiMeta.dependent_var?.unit || ''}</span>
        `;
    }
    
    // Interactive Tooltip logic
    const tooltip = document.getElementById("heatmap-tooltip");
    
    // Remove existing event listeners to avoid duplicates
    const newCanvas = canvas.cloneNode(true);
    canvas.parentNode.replaceChild(newCanvas, canvas);
    
    // Redraw offscreen to the new canvas
    const newCtx = newCanvas.getContext("2d");
    newCtx.imageSmoothingEnabled = true;
    newCtx.imageSmoothingQuality = 'high';
    newCtx.drawImage(offscreen, 0, 0, newCanvas.width, newCanvas.height);
    
    newCanvas.addEventListener("mousemove", (e) => {
        const rect = newCanvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        const xFraction = mouseX / rect.width;
        const yFraction = 1.0 - (mouseY / rect.height);
        
        const i = Math.max(0, Math.min(19, Math.round(xFraction * 19)));
        const j = Math.max(0, Math.min(19, Math.round(yFraction * 19)));
        
        const idx = i * 20 + j;
        if (idx >= 0 && idx < solution.length) {
            const val = solution[idx];
            tooltip.style.display = "block";
            tooltip.style.left = `${mouseX + 15}px`;
            tooltip.style.top = `${mouseY + 15}px`;
            tooltip.innerHTML = `
                <strong>Ort (x, y):</strong> (${(i/19).toFixed(2)}, ${(j/19).toFixed(2)})<br>
                <strong>Wert:</strong> ${val.toFixed(4)} ${uiMeta.dependent_var?.unit || ''}
            `;
        }
    });
    
    newCanvas.addEventListener("mouseleave", () => {
        tooltip.style.display = "none";
    });
}

// Global actions for Discovery Report download & toggle
function downloadRawReport() {
    if (!lastRunData) {
        alert("No report data available to download!");
        return;
    }
    const designName = lastRunData.miner?.design_name || "discovery_report";
    const slug = designName.toLowerCase().replace(/[^a-z0-9_]/g, "_").replace(/_+/g, "_");
    const blob = new Blob([JSON.stringify(lastRunData, null, 4)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `vectornaut_raw_report_${slug}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function toggleRawReport() {
    const rawReport = document.getElementById("raw-json-report");
    if (rawReport) {
        if (rawReport.style.display === "none" || rawReport.classList.contains("hidden")) {
            rawReport.style.display = "block";
            rawReport.classList.remove("hidden");
        } else {
            rawReport.style.display = "none";
            rawReport.classList.add("hidden");
        }
    }
}

// Bionic Assistant Chat State
let chatHistory = [];
let currentSuggestedParams = null;

function resetChat() {
    chatHistory = [];
    const chatHistoryEl = document.getElementById("chat-history");
    if (chatHistoryEl) {
        chatHistoryEl.innerHTML = `
            <div class="chat-message system-message">
                <p>Hallo! Ich bin dein Bionischer Design-Assistent. Ich habe Zugriff auf alle Parameter und Testergebnisse deines aktuellen Entwurfs. Wie kann ich dir heute helfen?</p>
            </div>
        `;
    }
    const optProposal = document.getElementById("optimization-proposal");
    if (optProposal) optProposal.classList.add("hidden");
    currentSuggestedParams = null;
}

function updateAssistantTab(data) {
    const titleEl = document.getElementById("findings-summary-title");
    const descEl = document.getElementById("findings-summary-desc");
    const questionsEl = document.getElementById("suggested-questions");
    const statsGridEl = document.getElementById("findings-stats-grid");
    
    if (!titleEl || !descEl || !questionsEl) return;
    
    const minerObj = data.miner || {};
    const auditedObj = data.auditor || {};
    const simObj = data.simulator || {};
    const designName = minerObj.design_name || "Bionischer Entwurf";
    const gainPct = simObj.performance_gain_pct || 0;
    
    let summaryText = "";
    let titleText = `Analyseergebnis: ${designName}`;
    let questions = [];
    let stats = [];
    
    if (designName.toLowerCase().includes("plastron") || designName.toLowerCase().includes("ski")) {
        // Extract slip_length and film_thickness dynamically to compute the ratio
        let slipLength = 0.00002;
        let filmThickness = 0.00001;
        
        const minerParams = minerObj.parameters || [];
        const auditedDict = auditedObj.audited_parameters_dict || {};
        
        if (auditedDict["slip_length"] !== undefined) {
            slipLength = auditedDict["slip_length"];
        } else {
            const found = minerParams.find(p => p.name === "slip_length");
            if (found) slipLength = found.value;
        }
        
        if (auditedDict["film_thickness"] !== undefined) {
            filmThickness = auditedDict["film_thickness"];
        } else {
            const found = minerParams.find(p => p.name === "film_thickness");
            if (found) filmThickness = found.value;
        }
        
        const frictionFraction = filmThickness / (filmThickness + slipLength);
        let fractionText = "";
        let fractionVal = "";
        if (Math.abs(frictionFraction - 1/3) < 0.01) {
            fractionText = "nur noch 1/3";
            fractionVal = "1/3";
        } else if (Math.abs(frictionFraction - 1/9) < 0.01) {
            fractionText = "nur noch 1/9";
            fractionVal = "1/9";
        } else {
            fractionText = `nur noch ${(frictionFraction * 100).toFixed(1)}%`;
            fractionVal = `${frictionFraction.toFixed(3)}`;
        }
        
        const slipUm = (slipLength * 1e6).toFixed(1);
        const filmUm = (filmThickness * 1e6).toFixed(1);
        
        titleText = `Analyseergebnis: ${designName} (Reibungsreduktion auf ${fractionVal})`;
        summaryText = `Der bionische Belag verringert die Wandreibung auf ${fractionText} des Werts herkömmlicher Ski-Beläge (dies entspricht einer Reibungsreduktion von exakt ${gainPct.toFixed(2)}%). Der physikalische Mechanismus basiert auf der Oberflächenstruktur des Springschwanzes (Collembola), die ein mikroskopisches Luftpolster (Plastron) stabilisiert.`;
        
        stats = [
            { val: fractionVal, lbl: "Reibungsverhältnis" },
            { val: `${gainPct.toFixed(2)}%`, lbl: "Reibungsreduktion" },
            { val: `${slipUm} µm`, lbl: "Slip-Länge (λ)" },
            { val: `${filmUm} µm`, lbl: "Filmdicke (h)" }
        ];
        
        questions = [
            `Erkläre mir die physikalische Ursache der Reibungsreduktion von ${gainPct.toFixed(2)}%.`,
            "Wie lässt sich diese Struktur herstellen?",
            "Wie kann das Design optimiert werden, um die Reibung weiter zu reduzieren?",
            "Zeige mir die Testergebnisse dieses Simulationslaufs."
        ];
    } else if (designName.toLowerCase().includes("shark") || designName.toLowerCase().includes("riblet") || designName.toLowerCase().includes("spheniscidae") || designName.toLowerCase().includes("penguin")) {
        titleText = `Analyseergebnis: ${designName}`;
        if (designName.toLowerCase().includes("insulated") || designName.toLowerCase().includes("penguin")) {
            summaryText = `Das bionische Isolationssystem erzielt eine hervorragende thermische Effizienz von ${gainPct.toFixed(2)}% im Vergleich zu herkömmlichen Systemen, indem es konvektiven Wärmetransfer durch gefangene Luftschichten unterdrückt.`;
            
            stats = [
                { val: `${gainPct.toFixed(2)}%`, lbl: "Wärme-Effizienz" },
                { val: simObj.primary_metric_value !== undefined ? simObj.primary_metric_value.toFixed(4) : "N/A", lbl: "Bionisch (k_eff)" },
                { val: simObj.reference_metric_value !== undefined ? simObj.reference_metric_value.toFixed(4) : "N/A", lbl: "Referenz (k_eff)" },
                { val: simObj.solver_method ? simObj.solver_method.toUpperCase() : "N/A", lbl: "Solver-Methode" }
            ];
            
            questions = [
                "Wie lässt sich diese Isolationsstruktur herstellen?",
                "Welcher physikalische Effekt verhindert hier den Wärmeverlust?",
                "Wie können wir die Geometrie der Luftspalten weiter optimieren?",
                "Zeige mir die Testergebnisse dieses Simulationslaufs."
            ];
        } else {
            summaryText = `Das bionische Design erzielt eine Reibungsreduktion von ${gainPct.toFixed(2)}% durch die Optimierung des viskosen Widerstands an der Wand und die Stabilisierung der laminaren Grenzschicht.`;
            
            stats = [
                { val: `${gainPct.toFixed(2)}%`, lbl: "Drag-Reduction" },
                { val: simObj.primary_metric_value !== undefined ? simObj.primary_metric_value.toFixed(4) : "N/A", lbl: "Bionische Reibung" },
                { val: simObj.reference_metric_value !== undefined ? simObj.reference_metric_value.toFixed(4) : "N/A", lbl: "Referenz Reibung" },
                { val: simObj.solver_method ? simObj.solver_method.toUpperCase() : "N/A", lbl: "Solver-Methode" }
            ];
            
            questions = [
                "Erkläre mir die physikalische Ursache für diese Effizienz.",
                "Wie lässt sich diese Oberflächenstruktur fertigen?",
                "Welche Parameter können wir anpassen, um das Ergebnis zu verbessern?",
                "Zeige mir die Testergebnisse dieses Simulationslaufs."
            ];
        }
    } else {
        summaryText = `Das bionische Design erzielt eine Effizienzsteigerung von ${gainPct.toFixed(2)}% im Vergleich zur konventionellen Referenz durch gezielte Anpassung der geometrischen Parameter.`;
        
        const uiMeta = auditedObj.ui_metadata || {};
        const primaryLabel = uiMeta.primary_metric?.label || "Bionisch";
        const referenceLabel = uiMeta.reference_metric?.label || "Referenz";
        const gainLabel = uiMeta.performance_gain?.label || "Steigerung";
        
        stats = [
            { val: `${gainPct.toFixed(2)}%`, lbl: gainLabel },
            { val: simObj.primary_metric_value !== undefined ? simObj.primary_metric_value.toFixed(4) : "N/A", lbl: primaryLabel },
            { val: simObj.reference_metric_value !== undefined ? simObj.reference_metric_value.toFixed(4) : "N/A", lbl: referenceLabel },
            { val: simObj.solver_method ? simObj.solver_method.toUpperCase() : "N/A", lbl: "Solver-Methode" }
        ];
        
        questions = [
            "Wie funktioniert der bionische Mechanismus im Detail?",
            "Welche Fertigungsmethoden sind für dieses Design geeignet?",
            "Schlage mir eine Optimierung der Parameter vor.",
            "Zeige mir die Testergebnisse dieses Simulationslaufs."
        ];
    }
    
    titleEl.innerHTML = `<span class="indicator-dot green" style="box-shadow: 0 0 6px var(--success-color);"></span> ${titleText}`;
    descEl.innerText = summaryText;
    
    // Populate Stats Grid
    if (statsGridEl) {
        statsGridEl.innerHTML = "";
        stats.forEach(st => {
            const box = document.createElement("div");
            box.className = "finding-stat-box";
            box.innerHTML = `
                <div class="stat-val">${st.val}</div>
                <div class="stat-lbl">${st.lbl}</div>
            `;
            statsGridEl.appendChild(box);
        });
    }
    
    // Populate Suggested Questions
    questionsEl.innerHTML = "";
    questions.forEach(q => {
        const btn = document.createElement("button");
        btn.className = "question-btn";
        btn.innerText = q;
        btn.onclick = () => {
            sendChatMessage(q);
        };
        questionsEl.appendChild(btn);
    });

    const optProposal = document.getElementById("optimization-proposal");
    if (optProposal) optProposal.classList.add("hidden");
    currentSuggestedParams = null;
}

async function sendChatMessage(messageText = null) {
    const inputEl = document.getElementById("chat-input");
    const sendBtn = document.getElementById("chat-send-btn");
    const chatHistoryEl = document.getElementById("chat-history");
    
    if (!inputEl || !chatHistoryEl) return;
    
    let text = messageText;
    if (text === null) {
        text = inputEl.value.trim();
        inputEl.value = "";
    }
    
    if (!text) return;
    
    // 1. Add user message to UI
    const userMsgDiv = document.createElement("div");
    userMsgDiv.className = "chat-message user-message";
    userMsgDiv.innerHTML = `<p>${escapeHtml(text)}</p>`;
    chatHistoryEl.appendChild(userMsgDiv);
    chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
    
    // 2. Push to local history state
    chatHistory.push({ role: "user", content: text });
    
    // 3. Disable input/button during request
    if (inputEl) inputEl.disabled = true;
    if (sendBtn) sendBtn.disabled = true;
    
    // Show typing / loading message
    const loadingDiv = document.createElement("div");
    loadingDiv.className = "chat-message assistant-message loading-message";
    loadingDiv.innerHTML = `<p><em>Analysiere physikalische Randbedingungen...</em></p>`;
    chatHistoryEl.appendChild(loadingDiv);
    chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
    
    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: text,
                history: chatHistory,
                current_run: lastRunData,
                is_mock: isMock
            })
        });
        
        if (!response.ok) {
            throw new Error("Fehler bei der Kommunikation mit dem Assistenten.");
        }
        
        const data = await response.json();
        
        // Remove loading message
        loadingDiv.remove();
        
        // 4. Add assistant message to UI (using marked.js if available)
        const assistantMsgDiv = document.createElement("div");
        assistantMsgDiv.className = "chat-message assistant-message";
        
        // Clean double-escaped newlines
        const cleanReply = data.reply.replace(/\\n/g, "\n");
        let parsedReply = cleanReply;
        if (typeof marked !== 'undefined') {
            parsedReply = marked.parse(cleanReply);
        } else {
            parsedReply = `<p>${escapeHtml(cleanReply).replace(/\n/g, "<br>")}</p>`;
        }
        
        assistantMsgDiv.innerHTML = parsedReply;
        chatHistoryEl.appendChild(assistantMsgDiv);
        
        // Push to local history state
        chatHistory.push({ role: "assistant", content: cleanReply });
        
        // 5. Handle optimization proposal
        const optProposal = document.getElementById("optimization-proposal");
        const paramsList = document.getElementById("proposal-params-list");
        
        if (data.suggested_params && Object.keys(data.suggested_params).length > 0) {
            currentSuggestedParams = data.suggested_params;
            if (paramsList) {
                paramsList.innerHTML = "";
                for (const [name, val] of Object.entries(data.suggested_params)) {
                    const badge = document.createElement("div");
                    badge.className = "proposal-param-badge";
                    badge.innerHTML = `<span class="p-name">${name}</span><span class="p-val">${val.toExponential(4)}</span>`;
                    paramsList.appendChild(badge);
                }
            }
            if (optProposal) optProposal.classList.remove("hidden");
        } else {
            if (optProposal) optProposal.classList.add("hidden");
            currentSuggestedParams = null;
        }
        
    } catch (e) {
        console.error("Chat error:", e);
        if (loadingDiv) loadingDiv.remove();
        
        const errorDiv = document.createElement("div");
        errorDiv.className = "chat-message system-message";
        errorDiv.innerHTML = `<p>Error: ${escapeHtml(e.message)}</p>`;
        chatHistoryEl.appendChild(errorDiv);
    } finally {
        if (inputEl) {
            inputEl.disabled = false;
            inputEl.focus();
        }
        if (sendBtn) sendBtn.disabled = false;
        chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
    }
}

function applySuggestedParameters() {
    if (!currentSuggestedParams) return;
    
    writeLog(`[SYSTEM] Optimierungslauf gestartet mit Parametern: ${JSON.stringify(currentSuggestedParams)}`, "info-log");
    
    runDiscoveryLoop(currentSuggestedParams);
    
    const optProposal = document.getElementById("optimization-proposal");
    if (optProposal) optProposal.classList.add("hidden");
    currentSuggestedParams = null;
}

function escapeHtml(unsafe) {
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}
