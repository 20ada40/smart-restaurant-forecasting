/* ── Helpers ── */
const $ = id => document.getElementById(id);
const show = id => $(id).style.display = "";
const hide = id => $(id).style.display = "none";
const fmt = n => typeof n === "number" ? n.toFixed(1) : "—";
const fmtInt = n => typeof n === "number" ? Math.round(n).toLocaleString() : "—";
const fmtGBP = n => typeof n === "number" ? "£" + n.toFixed(2) : "—";

function showSpinner() { show("spinner"); }
function hideSpinner() { hide("spinner"); }

async function api(endpoint, body) {
  showSpinner();
  try {
    const res = await fetch("/api/" + endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    return await res.json();
  } finally {
    hideSpinner();
  }
}

async function apiGet(endpoint) {
  showSpinner();
  try {
    const res = await fetch("/api/" + endpoint);
    return await res.json();
  } finally {
    hideSpinner();
  }
}

/* ── Set default dates ── */
function tomorrow() {
  const d = new Date(); d.setDate(d.getDate() + 1);
  return d.toISOString().slice(0, 10);
}
function today() {
  return new Date().toISOString().slice(0, 10);
}

window.addEventListener("DOMContentLoaded", () => {
  $("fc-date").value = tomorrow();
  $("sf-date").value = tomorrow();
  $("ing-date").value = today();
  $("fb-date").value = today();

  loadModelStatus();
  loadFeedbackHistory();
});

/* ── Tab navigation ── */
document.querySelectorAll(".nav-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    btn.classList.add("active");
    $("tab-" + btn.dataset.tab).classList.add("active");
    if (btn.dataset.tab === "model") loadModelStatus();
    if (btn.dataset.tab === "feedback") loadFeedbackHistory();
  });
});

/* ── Chart instance ── */
let coversChart = null;

/* ══════════════════════════════════
   FORECAST
══════════════════════════════════ */
$("btn-forecast").addEventListener("click", async () => {
  const data = await api("predict/covers", {
    date: $("fc-date").value,
    weather: $("fc-weather").value,
    is_holiday: $("fc-holiday").checked,
    is_special_event: $("fc-event").checked
  });
  if (data.error) { alert(data.error); return; }

  // Summary cards
  $("stat-total").textContent = fmtInt(data.daily_total);
  $("stat-range").textContent = `${fmtInt(data.daily_lower)} – ${fmtInt(data.daily_upper)}`;
  const peak = data.hourly.reduce((a, b) => a.predicted_covers > b.predicted_covers ? a : b);
  $("stat-peak").textContent = `${peak.hour}:00 (${fmtInt(peak.predicted_covers)})`;
  $("stat-conf").textContent = (data.confidence * 100).toFixed(0) + "%";
  const stats = data.feedback_stats;
  $("stat-corrections").textContent = stats.count || "0";

  show("fc-summary");
  show("fc-chart-wrap");
  show("fc-table-wrap");

  // Chart
  const hours = data.hourly.map(h => h.hour + ":00");
  const values = data.hourly.map(h => h.predicted_covers);
  const lower = data.hourly.map(h => h.lower_bound);
  const upper = data.hourly.map(h => h.upper_bound);

  if (coversChart) coversChart.destroy();
  const ctx = $("chart-covers").getContext("2d");
  coversChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: hours,
      datasets: [
        {
          label: "Predicted Covers",
          data: values,
          backgroundColor: "rgba(200,168,107,0.6)",
          borderColor: "rgba(200,168,107,1)",
          borderWidth: 1,
          borderRadius: 3,
          order: 2
        },
        {
          label: "Upper Bound",
          data: upper,
          type: "line",
          borderColor: "rgba(107,158,200,0.5)",
          backgroundColor: "rgba(107,158,200,0.05)",
          pointRadius: 0,
          fill: "+1",
          tension: 0.4,
          order: 1
        },
        {
          label: "Lower Bound",
          data: lower,
          type: "line",
          borderColor: "rgba(107,158,200,0.5)",
          backgroundColor: "rgba(107,158,200,0.05)",
          pointRadius: 0,
          fill: false,
          tension: 0.4,
          order: 1
        }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: "#888", font: { family: "DM Mono", size: 11 } } },
        tooltip: { backgroundColor: "#1e1e1e", borderColor: "#2a2a2a", borderWidth: 1 }
      },
      scales: {
        x: { grid: { color: "#1e1e1e" }, ticks: { color: "#888" } },
        y: { grid: { color: "#1e1e1e" }, ticks: { color: "#888" }, beginAtZero: true }
      }
    }
  });

  // Table
  const tbody = $("fc-table").querySelector("tbody");
  tbody.innerHTML = data.hourly.map(h => `
    <tr>
      <td class="mono">${h.hour}:00</td>
      <td class="mono">${fmtInt(h.predicted_covers)}</td>
      <td class="mono" style="color:var(--text-muted)">${fmtInt(h.lower_bound)} – ${fmtInt(h.upper_bound)}</td>
      <td class="mono">${(h.confidence * 100).toFixed(0)}%</td>
      <td>${h.factors.weather} <span style="color:var(--text-dim)">(×${h.factors.weather_score ?? "—"})</span></td>
    </tr>
  `).join("");
});

/* ══════════════════════════════════
   STAFF
══════════════════════════════════ */
$("btn-staff").addEventListener("click", async () => {
  const data = await api("predict/staff", {
    date: $("sf-date").value,
    weather: $("sf-weather").value
  });
  if (data.error) { alert(data.error); return; }

  $("sf-cost").textContent = fmtGBP(data.daily_summary.total_labor_cost);
  $("sf-hours").textContent = data.daily_summary.total_staff_hours + " hrs";
  show("sf-summary");

  const tbody = $("sf-table").querySelector("tbody");
  tbody.innerHTML = data.daily_summary.by_role.map(r => `
    <tr>
      <td>${r.role.replace(/_/g, " ")}</td>
      <td><span class="pill ${r.station === 'kitchen' ? 'warn' : r.station === 'bar' ? 'normal' : 'buffer'}">${r.station}</span></td>
      <td class="mono">${r.peak_count}</td>
      <td class="mono">${r.total_hours}</td>
      <td class="mono">£${r.hourly_rate.toFixed(2)}</td>
      <td class="mono">${fmtGBP(r.total_cost)}</td>
    </tr>
  `).join("");
  show("sf-table-wrap");
});

/* ══════════════════════════════════
   INGREDIENTS
══════════════════════════════════ */
$("btn-ingredients").addEventListener("click", async () => {
  const data = await api("predict/ingredients", {
    start_date: $("ing-date").value,
    days_ahead: parseInt($("ing-days").value)
  });
  if (data.error) { alert(data.error); return; }

  $("ing-total").textContent = fmtGBP(data.cost_summary.total_food_cost);
  const critical = data.orders.filter(o => o.urgency === "critical").length;
  $("ing-critical").textContent = critical + " items";
  show("ing-summary");

  // Category chips
  const catsEl = $("ing-cats");
  catsEl.innerHTML = Object.entries(data.cost_summary.by_category).map(([cat, cost]) => `
    <div class="cat-chip">
      ${cat.replace(/_/g, " ")}
      <span>${fmtGBP(cost)}</span>
    </div>
  `).join("");
  show("ing-cats");

  const tbody = $("ing-table").querySelector("tbody");
  tbody.innerHTML = data.orders.map(o => `
    <tr>
      <td>${o.ingredient.replace(/_/g, " ")}</td>
      <td class="mono">${o.quantity_to_order}</td>
      <td class="mono">${o.unit}</td>
      <td class="mono">${o.shelf_life_days}d</td>
      <td class="mono">${o.lead_days}d</td>
      <td class="mono">${fmtGBP(o.cost)}</td>
      <td><span class="pill ${o.urgency}">${o.urgency}</span></td>
    </tr>
  `).join("");
  show("ing-table-wrap");
});

/* ══════════════════════════════════
   FEEDBACK
══════════════════════════════════ */
$("btn-feedback").addEventListener("click", async () => {
  const predicted = parseFloat($("fb-predicted").value);
  const actual = parseFloat($("fb-actual").value);
  if (isNaN(predicted) || isNaN(actual)) {
    alert("Please enter both predicted and actual covers."); return;
  }

  const data = await api("feedback", {
    date: $("fb-date").value,
    predicted_covers: predicted,
    actual_covers: actual,
    weather: $("fb-weather").value,
    note: $("fb-note").value
  });

  const el = $("fb-result");
  el.className = "feedback-result" + (data.status === "success" ? "" : " error");
  const diff = actual - predicted;
  const diffStr = diff >= 0 ? `+${diff.toFixed(0)}` : diff.toFixed(0);
  el.innerHTML = data.status === "success"
    ? `✓ <strong>Model updated.</strong> ${data.message}<br>
       Residual: <strong class="${diff >= 0 ? 'pos' : 'neg'}">${diffStr} covers</strong> &nbsp;|&nbsp;
       Factor: <strong>${data.adjustment_factor}</strong>`
    : `✗ ${data.error}`;
  show("fb-result");

  // Clear inputs
  $("fb-predicted").value = "";
  $("fb-actual").value = "";
  $("fb-note").value = "";

  loadFeedbackHistory();
});

async function loadFeedbackHistory() {
  const data = await apiGet("feedback/history");
  const history = data.history || [];
  $("fb-count").textContent = (data.stats?.count || 0) + " events";

  const tbody = $("fb-history-table").querySelector("tbody");
  if (!history.length) {
    tbody.innerHTML = `<tr><td colspan="5" style="color:var(--text-dim);padding:1.5rem;text-align:center">No corrections submitted yet</td></tr>`;
    return;
  }
  tbody.innerHTML = [...history].reverse().slice(0, 20).map(h => {
    const res = (h.residual || (h.actual - h.predicted));
    const cls = res >= 0 ? "pos" : "neg";
    return `
      <tr>
        <td class="mono">${h.date}</td>
        <td class="mono">${fmtInt(h.predicted)}</td>
        <td class="mono">${fmtInt(h.actual)}</td>
        <td class="mono ${cls}">${res >= 0 ? "+" : ""}${fmt(res)}</td>
        <td style="color:var(--text-muted);font-size:.82rem">${h.note || "—"}</td>
      </tr>
    `;
  }).join("");
}

/* ══════════════════════════════════
   MODEL HEALTH
══════════════════════════════════ */
async function loadModelStatus() {
  const data = await apiGet("model/status");

  $("m-cvmae").textContent = data.cv_mae ? fmt(data.cv_mae) : "—";
  const stats = data.feedback_stats || {};
  $("m-fb-count").textContent = stats.count ?? "—";
  $("m-fb-mae").textContent = stats.mae != null ? fmt(stats.mae) : "—";

  const bias = stats.recent_bias;
  const biasEl = $("m-bias");
  if (bias != null) {
    biasEl.textContent = (bias >= 0 ? "+" : "") + fmt(bias);
    biasEl.className = "stat-value " + (Math.abs(bias) < 5 ? "pos" : bias > 0 ? "pos" : "neg");
  } else {
    biasEl.textContent = "—";
  }

  // Update model badge
  const badge = $("model-badge");
  if (data.is_trained) {
    badge.textContent = "Model Active";
    badge.className = "badge good";
  } else {
    badge.textContent = "Not Trained";
    badge.className = "badge warn";
  }

  // Feature importance bars
  if (data.top_features && data.top_features.length) {
    const max = data.top_features[0][1];
    $("feature-bars").innerHTML = data.top_features.map(([feat, imp]) => `
      <div class="feature-bar-row">
        <div class="feature-name">${feat.replace(/_/g, " ")}</div>
        <div class="feature-bar-bg">
          <div class="feature-bar-fill" style="width:${(imp / max * 100).toFixed(1)}%"></div>
        </div>
        <div class="feature-imp">${(imp * 100).toFixed(1)}%</div>
      </div>
    `).join("");
  }
}
