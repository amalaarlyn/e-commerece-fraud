/* ═══════════════════════════════════════════════════════
   ReturnShield — Application Logic
   Client-side mock scoring that mirrors the real pipeline.
   ═══════════════════════════════════════════════════════ */

// --- Constants (mirror features.py) ---
const SEQ_LEN = 8;
const N_FEATURES = 11;
const REASON_VOCAB = ["none", "wrong_size", "changed_mind", "wrong_item", "damaged", "item_not_received"];
const FEATURE_NAMES = [
  "recency", "order_value", "high_value_category", "returned_flag", "days_to_return",
  "delivery_confirmed", "return_reason", "rolling_return_rate",
  "rolling_refund_to_spend", "category_concentration", "days_since_last_return"
];
const HIGH_VALUE_CATS = new Set(["apparel_event", "electronics"]);

const CONFIG = { low_thresh: 0.40, high_thresh: 0.65 };

// ═══════════════════ SCENARIOS ═══════════════════
const SCENARIOS = {
  genuine: {
    label: "Genuine Customer",
    description: "Normal purchase/return behavior — low risk.",
    orders: [
      { order_date: "2026-01-10T10:00:00", category: "home",          order_value: 45.00,  returned: 0, return_date: null, return_reason: null, delivery_confirmed: 1 },
      { order_date: "2026-02-14T12:00:00", category: "apparel_casual", order_value: 89.00,  returned: 0, return_date: null, return_reason: null, delivery_confirmed: 1 },
      { order_date: "2026-03-20T09:30:00", category: "beauty",         order_value: 32.00,  returned: 0, return_date: null, return_reason: null, delivery_confirmed: 1 },
      { order_date: "2026-04-15T14:00:00", category: "home",           order_value: 120.00, returned: 0, return_date: null, return_reason: null, delivery_confirmed: 1 },
      { order_date: "2026-05-08T11:00:00", category: "apparel_casual", order_value: 65.00,  returned: 1, return_date: "2026-05-22T11:00:00", return_reason: "wrong_size", delivery_confirmed: 1 },
    ]
  },
  wardrobing: {
    label: "Wardrobing Suspect",
    description: "High-value event wear, returned quickly after use — classic wardrobing pattern.",
    orders: [
      { order_date: "2026-03-01T10:00:00", category: "apparel_casual",  order_value: 55.00,  returned: 0, return_date: null, return_reason: null, delivery_confirmed: 1 },
      { order_date: "2026-03-10T10:00:00", category: "apparel_event",   order_value: 320.00, returned: 1, return_date: "2026-03-13T10:00:00", return_reason: "changed_mind", delivery_confirmed: 1 },
      { order_date: "2026-04-02T10:00:00", category: "apparel_event",   order_value: 480.00, returned: 1, return_date: "2026-04-05T10:00:00", return_reason: "changed_mind", delivery_confirmed: 1 },
      { order_date: "2026-04-28T10:00:00", category: "electronics",     order_value: 899.00, returned: 0, return_date: null, return_reason: null, delivery_confirmed: 1 },
      { order_date: "2026-05-20T10:00:00", category: "apparel_event",   order_value: 550.00, returned: 1, return_date: "2026-05-22T10:00:00", return_reason: "changed_mind", delivery_confirmed: 1 },
    ]
  },
  inr: {
    label: "INR Abuser",
    description: "'Item not received' claims despite confirmed delivery — classic INR fraud pattern.",
    orders: [
      { order_date: "2026-01-05T10:00:00", category: "electronics",     order_value: 450.00, returned: 0, return_date: null, return_reason: null, delivery_confirmed: 1 },
      { order_date: "2026-01-20T10:00:00", category: "electronics",     order_value: 680.00, returned: 1, return_date: "2026-01-21T10:00:00", return_reason: "item_not_received", delivery_confirmed: 1 },
      { order_date: "2026-02-08T10:00:00", category: "electronics",     order_value: 1200.00,returned: 1, return_date: "2026-02-09T10:00:00", return_reason: "item_not_received", delivery_confirmed: 1 },
      { order_date: "2026-02-28T10:00:00", category: "apparel_casual",  order_value: 90.00,  returned: 0, return_date: null, return_reason: null, delivery_confirmed: 1 },
      { order_date: "2026-03-15T10:00:00", category: "electronics",     order_value: 950.00, returned: 1, return_date: "2026-03-16T10:00:00", return_reason: "item_not_received", delivery_confirmed: 1 },
    ]
  },
  serial: {
    label: "Serial Fraud Ring",
    description: "High velocity, varied reasons, shared device/address patterns (not visible in sequence features — handled by GNN module).",
    orders: [
      { order_date: "2026-04-01T10:00:00", category: "electronics",     order_value: 340.00, returned: 1, return_date: "2026-04-03T10:00:00", return_reason: "changed_mind", delivery_confirmed: 1 },
      { order_date: "2026-04-04T10:00:00", category: "apparel_event",   order_value: 220.00, returned: 1, return_date: "2026-04-06T10:00:00", return_reason: "damaged",     delivery_confirmed: 0 },
      { order_date: "2026-04-07T10:00:00", category: "electronics",     order_value: 780.00, returned: 1, return_date: "2026-04-08T10:00:00", return_reason: "item_not_received", delivery_confirmed: 1 },
      { order_date: "2026-04-09T10:00:00", category: "beauty",          order_value: 65.00,  returned: 0, return_date: null, return_reason: null, delivery_confirmed: 1 },
      { order_date: "2026-04-11T10:00:00", category: "apparel_event",   order_value: 410.00, returned: 1, return_date: "2026-04-12T10:00:00", return_reason: "changed_mind",   delivery_confirmed: 1 },
      { order_date: "2026-04-13T10:00:00", category: "electronics",     order_value: 1100.00,returned: 1, return_date: "2026-04-14T10:00:00", return_reason: "damaged",       delivery_confirmed: 1 },
    ]
  }
};


// ═══════════════════ FEATURE ENGINEERING (mirrors features.py) ═══════════════════

function encodeReason(reason) {
  const r = (reason || "none").toLowerCase().trim();
  let idx = REASON_VOCAB.indexOf(r);
  if (idx === -1) idx = 0;
  return idx;
}

function buildLiveSequence(orders) {
  const sorted = [...orders].sort((a, b) => new Date(a.order_date) - new Date(b.order_date));
  const stepFeats = [];
  let prevDate = null;

  let totalOrders = 0;
  let totalReturns = 0;
  let totalSpend = 0.0;
  let totalRefunded = 0.0;
  const categoryReturnCounts = {};
  let lastReturnDate = null;

  for (const row of sorted) {
    const orderDate = new Date(row.order_date);
    let daysSincePrev = 0;
    if (prevDate) daysSincePrev = Math.round((orderDate - prevDate) / 86400000);
    prevDate = orderDate;

    const returned = row.returned ? 1 : 0;
    let daysToReturn = 0;
    if (returned && row.return_date) {
      daysToReturn = Math.round((new Date(row.return_date) - orderDate) / 86400000);
    }

    const logOrderValue = Math.log1p(row.order_value);
    const highValueCat = HIGH_VALUE_CATS.has(row.category) ? 1 : 0;
    const reason = returned ? row.return_reason : "none";
    const reasonIdx = encodeReason(reason);

    const rollingReturnRate = totalOrders > 0 ? (totalReturns / totalOrders) : 0.0;
    const rollingRefundToSpend = totalSpend > 0 ? (totalRefunded / totalSpend) : 0.0;
    let categoryConcentration = 0.0;
    if (totalReturns > 0) {
      const maxCatReturns = Math.max(...Object.values(categoryReturnCounts));
      categoryConcentration = maxCatReturns / totalReturns;
    }
    let daysSinceLastReturnNorm = 1.0;
    if (lastReturnDate) {
      const daysSinceLastReturn = Math.round((orderDate - lastReturnDate) / 86400000);
      daysSinceLastReturnNorm = Math.min(daysSinceLastReturn, 180) / 180.0;
    }

    const vec = [
      Math.min(daysSincePrev, 90) / 90.0,
      logOrderValue / 8.0,
      highValueCat,
      returned,
      Math.min(daysToReturn, 30) / 30.0,
      row.delivery_confirmed || 1,
      reasonIdx / REASON_VOCAB.length,
      rollingReturnRate,
      rollingRefundToSpend,
      categoryConcentration,
      daysSinceLastReturnNorm
    ];
    stepFeats.push(vec);

    // update state
    totalOrders += 1;
    totalSpend += row.order_value;
    if (returned) {
      totalReturns += 1;
      totalRefunded += row.order_value;
      categoryReturnCounts[row.category] = (categoryReturnCounts[row.category] || 0) + 1;
      lastReturnDate = row.return_date ? new Date(row.return_date) : orderDate;
    }
  }

  let window = stepFeats.slice(-SEQ_LEN);
  const padLen = SEQ_LEN - window.length;
  if (padLen > 0) {
    const padRow = new Array(N_FEATURES).fill(0);
    window = [...Array(padLen).fill(null).map(() => [...padRow]), ...window];
  }

  return window;
}


// ═══════════════════ MOCK SCORING (simulates LSTM + calibrator) ═══════════════════

function mockScore(sequence) {
  let score = 0;
  const lastRow = sequence[sequence.length - 1];

  // Fast return signal
  if (lastRow[4] > 0 && lastRow[4] < 0.15) score += 0.18;

  // High-value category on return
  if (lastRow[2] === 1 && lastRow[3] === 1) score += 0.12;

  // High order value
  if (lastRow[1] > 0.6) score += 0.08;

  // Return reason signals
  const reasonIdx = Math.round(lastRow[6] * REASON_VOCAB.length);
  const reason = REASON_VOCAB[Math.min(reasonIdx, REASON_VOCAB.length - 1)];
  if (reason === "changed_mind") score += 0.06;
  if (reason === "item_not_received") score += 0.10;
  if (reason === "damaged" && lastRow[5] < 0.5) score += 0.12;

  // Aggregate signals
  if (lastRow[7] > 0.4) score += 0.20; // high rolling return rate
  if (lastRow[8] > 0.4) score += 0.10; // high refund to spend
  if (lastRow[9] > 0.6) score += 0.08; // category concentration
  if (lastRow[10] < 0.15) score += 0.12; // short days since last return

  // Historical pattern (repeated fast returns)
  let fastReturnCount = 0;
  for (const row of sequence) {
    if (row[3] === 1 && row[4] > 0 && row[4] < 0.15) fastReturnCount++;
  }
  if (fastReturnCount >= 2) score += 0.12;

  // Clamp and add noise
  score = Math.min(Math.max(score, 0.02), 0.96);
  score += (Math.random() - 0.5) * 0.04;
  return Math.min(Math.max(score, 0.01), 0.99);
}

function threeTier(prob) {
  if (prob < CONFIG.low_thresh) return "auto_approve";
  if (prob < CONFIG.high_thresh) return "soft_friction";
  return "deny";
}


// ═══════════════════ MOCK EXPLAINABILITY ═══════════════════

function mockExplain(sequence, score) {
  const lastRow = sequence[sequence.length - 1];
  const importances = FEATURE_NAMES.map((name, i) => {
    let imp = 0;
    switch (name) {
      case "days_to_return": imp = lastRow[3] === 1 && lastRow[4] < 0.15 ? 0.35 : 0.05; break;
      case "high_value_category": imp = lastRow[2] === 1 ? 0.25 : 0.03; break;
      case "order_value": imp = lastRow[1] > 0.6 ? 0.20 : 0.08; break;
      case "return_reason":
        const rIdx = Math.round(lastRow[6] * REASON_VOCAB.length);
        const r = REASON_VOCAB[Math.min(rIdx, REASON_VOCAB.length - 1)];
        imp = (r === "changed_mind" || r === "item_not_received") ? 0.22 : 0.05;
        break;
      case "returned_flag": imp = lastRow[3] === 1 ? 0.15 : 0.02; break;
      case "recency": imp = lastRow[0] < 0.05 ? 0.18 : 0.04; break;
      case "delivery_confirmed": imp = lastRow[5] < 0.5 ? 0.20 : 0.03; break;
      case "rolling_return_rate": imp = lastRow[7] > 0.4 ? 0.30 : 0.05; break;
      case "rolling_refund_to_spend": imp = lastRow[8] > 0.4 ? 0.25 : 0.04; break;
      case "category_concentration": imp = lastRow[9] > 0.6 ? 0.20 : 0.03; break;
      case "days_since_last_return": imp = lastRow[10] < 0.15 && lastRow[10] > 0 ? 0.25 : 0.02; break;
    }
    // Add small random perturbation for realism
    imp += Math.random() * 0.03;
    return { name, importance: imp, value: lastRow[i] };
  });

  importances.sort((a, b) => b.importance - a.importance);

  // Generate reason phrases
  const reasons = [];
  for (const feat of importances.slice(0, 3)) {
    const risky = score > 0.3;
    reasons.push(generatePhrase(feat.name, feat.value, risky, lastRow));
  }

  return {
    top_features: importances.slice(0, 5).map(f => [f.name, Math.round(f.importance * 10000) / 10000]),
    reasons,
    importances
  };
}

function generatePhrase(featureName, cellValue, risky, lastRow) {
  if (featureName === "days_to_return" && risky && cellValue < 0.1) return "Returned unusually fast after purchase.";
  if (featureName === "high_value_category" && risky && cellValue > 0.5) return "Return involves a high-value / high-risk category (event wear or electronics).";
  if (featureName === "recency" && risky && cellValue < 0.05 && cellValue > 0) return "Ordered again almost immediately after the previous order.";
  if (featureName === "order_value" && risky && cellValue > 0.6) return "Unusually high order value relative to this customer's history.";
  if (featureName === "delivery_confirmed" && risky && cellValue < 0.5) return "Delivery was not confirmed by the carrier.";
  if (featureName === "return_reason") {
    const rIdx = Math.round(cellValue * REASON_VOCAB.length);
    const reason = REASON_VOCAB[Math.min(rIdx, REASON_VOCAB.length - 1)];
    if (risky && reason === "item_not_received") return "Reason given was 'item not received', a common INR-abuse pattern.";
    if (risky && reason === "changed_mind") return "Reason given was 'changed mind', consistent with wardrobing.";
    if (risky && reason === "damaged") return "Reason given was 'damaged' -- flag for image forensics cross-check.";
    return `Return reason '${reason}' noted.`;
  }
  if (featureName === "rolling_return_rate" && risky && cellValue > 0.4) return "This customer already returns a high share of what they order.";
  if (featureName === "rolling_refund_to_spend" && risky && cellValue > 0.4) return "A large fraction of this customer's total spend has come back as refunds.";
  if (featureName === "category_concentration" && risky && cellValue > 0.6) return "This customer's past returns are heavily concentrated in one category.";
  if (featureName === "days_since_last_return" && risky && cellValue < 0.15) return "Returned again very soon after their last return -- unusual rhythm.";
  return `${featureName} contributed ${risky ? 'toward' : 'away from'} the fraud signal.`;
}


// ═══════════════════ UI RENDERING ═══════════════════

let currentOrders = [];

function renderOrders(orders) {
  const container = document.getElementById("orders-container");
  container.innerHTML = "";

  orders.forEach((order, idx) => {
    const isLast = idx === orders.length - 1;
    const card = document.createElement("div");
    card.className = "order-card";

    const returnBadge = order.returned
      ? `<span class="order-return-badge returned">⟲ Returned</span>`
      : `<span class="order-return-badge kept">✓ Kept</span>`;

    const dateStr = new Date(order.order_date).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
    const returnDateStr = order.return_date ? new Date(order.return_date).toLocaleDateString("en-US", { month: "short", day: "numeric" }) : "—";

    card.innerHTML = `
      <div class="order-card-header">
        <span class="order-num">Order ${idx + 1}${isLast ? " · Being Scored" : ""}</span>
        ${returnBadge}
      </div>
      <div class="order-fields">
        <div class="order-field"><span class="order-field-label">Date</span><span class="order-field-value">${dateStr}</span></div>
        <div class="order-field"><span class="order-field-label">Category</span><span class="order-field-value">${order.category}</span></div>
        <div class="order-field"><span class="order-field-label">Value</span><span class="order-field-value">$${order.order_value.toFixed(2)}</span></div>
        <div class="order-field"><span class="order-field-label">Delivery</span><span class="order-field-value">${order.delivery_confirmed ? "Confirmed" : "Unconfirmed"}</span></div>
        ${order.returned ? `
          <div class="order-field"><span class="order-field-label">Return Date</span><span class="order-field-value">${returnDateStr}</span></div>
          <div class="order-field"><span class="order-field-label">Reason</span><span class="order-field-value">${order.return_reason || "—"}</span></div>
        ` : ""}
      </div>
    `;
    container.appendChild(card);
  });
}

function renderHeatmap(sequence) {
  const container = document.getElementById("sequence-heatmap");
  container.innerHTML = "";

  // Header row
  FEATURE_NAMES.forEach(name => {
    const cell = document.createElement("div");
    cell.className = "heatmap-cell";
    cell.style.background = "rgba(255,255,255,0.04)";
    cell.style.fontSize = "0.55rem";
    cell.style.color = "rgba(255,255,255,0.5)";
    cell.style.aspectRatio = "auto";
    cell.style.padding = "4px 2px";
    
    // Create short name for column header
    let shortName = name.slice(0, 6);
    if (name === "high_value_category") shortName = "hi_cat";
    if (name === "rolling_return_rate") shortName = "rt_rat";
    if (name === "rolling_refund_to_spend") shortName = "rf_spd";
    if (name === "category_concentration") shortName = "cat_co";
    if (name === "days_since_last_return") shortName = "ds_rtn";
    if (name === "delivery_confirmed") shortName = "del_co";
    if (name === "returned_flag") shortName = "rtn_fl";
    
    cell.textContent = shortName;
    cell.title = name;
    container.appendChild(cell);
  });

  // Data rows
  for (let t = 0; t < sequence.length; t++) {
    for (let f = 0; f < N_FEATURES; f++) {
      const val = sequence[t][f];
      const cell = document.createElement("div");
      cell.className = "heatmap-cell";

      // Color: dark -> cyan -> violet -> pink
      const intensity = Math.min(val, 1);
      let r, g, b;
      if (intensity < 0.33) {
        const p = intensity / 0.33;
        r = Math.round(15 + p * (34 - 15));
        g = Math.round(20 + p * (211 - 20));
        b = Math.round(36 + p * (238 - 36));
      } else if (intensity < 0.66) {
        const p = (intensity - 0.33) / 0.33;
        r = Math.round(34 + p * (139 - 34));
        g = Math.round(211 + p * (92 - 211));
        b = Math.round(238 + p * (246 - 238));
      } else {
        const p = (intensity - 0.66) / 0.34;
        r = Math.round(139 + p * (236 - 139));
        g = Math.round(92 + p * (72 - 92));
        b = Math.round(246 + p * (153 - 246));
      }

      cell.style.background = `rgba(${r},${g},${b},${0.15 + intensity * 0.7})`;
      cell.textContent = val.toFixed(2);
      cell.title = `${FEATURE_NAMES[f]} @ t-${sequence.length - 1 - t}: ${val.toFixed(4)}`;
      container.appendChild(cell);
    }
  }
}

function renderGauge(score) {
  const totalArc = 251.2; // arc length for semicircle
  const offset = totalArc * (1 - score);

  const gaugeFill = document.getElementById("gauge-fill");
  const gaugeScore = document.getElementById("gauge-score");
  const gaugeDot = document.getElementById("gauge-dot");

  // Animate
  gaugeFill.style.transition = "stroke-dashoffset 1.2s cubic-bezier(0.16, 1, 0.3, 1)";
  gaugeFill.setAttribute("stroke-dashoffset", offset);
  gaugeScore.textContent = score.toFixed(3);

  // Position the dot along the arc
  const angle = Math.PI * (1 - score); // π (left) to 0 (right)
  const cx = 100 - 80 * Math.cos(angle);
  const cy = 100 - 80 * Math.sin(angle);
  gaugeDot.style.transition = "all 1.2s cubic-bezier(0.16, 1, 0.3, 1)";
  gaugeDot.setAttribute("cx", cx);
  gaugeDot.setAttribute("cy", cy);
}

function renderTier(tier, score) {
  const badge = document.getElementById("tier-badge");
  badge.className = "tier-badge";
  badge.textContent = tier.replace("_", " ");

  if (tier === "auto_approve") badge.classList.add("approve");
  else if (tier === "soft_friction") badge.classList.add("friction");
  else badge.classList.add("deny");

  // Position indicator on tier bar
  const indicator = document.getElementById("tier-indicator");
  indicator.style.display = "block";
  indicator.style.left = `${score * 100}%`;
}

function renderExplanation(explanation) {
  const container = document.getElementById("explanation-content");
  container.innerHTML = "";

  // Reason phrases
  explanation.reasons.forEach((reason, i) => {
    const item = document.createElement("div");
    item.className = "explanation-item";
    item.style.animationDelay = `${i * 0.1}s`;
    item.innerHTML = `
      <span class="explanation-rank">#${i + 1}</span>
      <div>
        <span class="explanation-feature">${explanation.top_features[i][0]}</span>
        <div class="explanation-text">${reason}</div>
      </div>
    `;
    container.appendChild(item);
  });

  // Feature importance bars
  const barContainer = document.createElement("div");
  barContainer.className = "feature-bar-container";
  const maxImp = Math.max(...explanation.importances.map(f => f.importance));

  explanation.importances.slice(0, 7).forEach((feat, i) => {
    const pct = (feat.importance / maxImp) * 100;
    const row = document.createElement("div");
    row.className = "feature-bar-row";
    row.innerHTML = `
      <span class="feature-bar-name">${feat.name.length > 20 ? feat.name.substring(0, 18) + '..' : feat.name}</span>
      <div class="feature-bar-track">
        <div class="feature-bar-fill" style="width: 0%"><span>${feat.importance.toFixed(3)}</span></div>
      </div>
    `;
    barContainer.appendChild(row);

    // Animate bars after render
    requestAnimationFrame(() => {
      setTimeout(() => {
        row.querySelector(".feature-bar-fill").style.width = `${pct}%`;
      }, 100 + i * 80);
    });
  });

  container.appendChild(barContainer);
}


// ═══════════════════ EVENT HANDLERS ═══════════════════

function loadScenario(scenarioKey) {
  const scenario = SCENARIOS[scenarioKey];
  currentOrders = JSON.parse(JSON.stringify(scenario.orders));
  renderOrders(currentOrders);

  // Reset output
  document.getElementById("output-placeholder").classList.remove("hidden");
  document.getElementById("output-results").classList.add("hidden");

  // Update active button
  document.querySelectorAll(".scenario-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.scenario === scenarioKey);
  });
  
  // Set current scenario and update graph intelligence automatically
  window.currentScenario = scenarioKey;
  if (typeof calculateGraphScore === 'function') calculateGraphScore();
}

function runScoring() {
  if (!currentOrders.length) return;

  // Validate: last order must be a return
  const lastOrder = currentOrders[currentOrders.length - 1];
  if (!lastOrder.returned) {
    alert("The last order must be a return event (returned = 1) to score.");
    return;
  }

  // Build sequence
  const sequence = buildLiveSequence(currentOrders);

  // Score
  const rawScore = mockScore(sequence);
  const calibratedScore = rawScore; // In real system, Platt scaling applied here
  const tier = threeTier(calibratedScore);

  // Explain
  const explanation = mockExplain(sequence, calibratedScore);

  // Show results
  document.getElementById("output-placeholder").classList.add("hidden");
  document.getElementById("output-results").classList.remove("hidden");

  // Render all panels
  renderHeatmap(sequence);
  renderGauge(calibratedScore);
  renderTier(tier, calibratedScore);
  renderExplanation(explanation);
  
  // Sync to Risk Fusion Engine
  const bSlider = document.getElementById("behavior-score-slider");
  if (bSlider) {
    bSlider.value = calibratedScore * 100;
    const bValEl = document.getElementById("behavior-score-val");
    if(bValEl) bValEl.textContent = calibratedScore.toFixed(2);
    if (typeof runRiskFusion === 'function') runRiskFusion();
  }
}

// --- Init ---
document.addEventListener("DOMContentLoaded", () => {
  // Scenario buttons
  document.querySelectorAll(".scenario-btn").forEach(btn => {
    btn.addEventListener("click", () => loadScenario(btn.dataset.scenario));
  });

  // Score button
  document.getElementById("score-btn").addEventListener("click", runScoring);

  // Add order button
  document.getElementById("add-order-btn").addEventListener("click", () => {
    const lastDate = currentOrders.length
      ? new Date(currentOrders[currentOrders.length - 1].order_date)
      : new Date("2026-06-01T10:00:00");
    const newDate = new Date(lastDate.getTime() + 7 * 86400000);

    currentOrders.push({
      order_date: newDate.toISOString().slice(0, 19),
      category: "apparel_casual",
      order_value: 75.00,
      returned: 1,
      return_date: new Date(newDate.getTime() + 3 * 86400000).toISOString().slice(0, 19),
      return_reason: "changed_mind",
      delivery_confirmed: 1,
    });
    renderOrders(currentOrders);
  });

  // Load default scenario
  loadScenario("genuine");

  // Init Risk Fusion
  initRiskFusion();

  // Animate elements on scroll
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = "1";
        entry.target.style.transform = "translateY(0)";
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll(".pipeline-step, .result-card, .cost-comparison").forEach(el => {
    el.style.opacity = "0";
    el.style.transform = "translateY(20px)";
    el.style.transition = "opacity 0.6s ease, transform 0.6s ease";
    observer.observe(el);
  });
});

// ═══════════════════ RISK FUSION ENGINE ═══════════════════

const FUSION_WEIGHTS = {
  graph: 0.35,
  image: 0.40,
  behavior: 0.25,
};

function runRiskFusion() {
  // 1. Get input values
  const graphScore = parseInt(document.getElementById("graph-score-slider").value, 10) / 100.0;
  const imageScore = parseInt(document.getElementById("image-score-slider").value, 10) / 100.0;
  const behaviorScore = parseInt(document.getElementById("behavior-score-slider").value, 10) / 100.0;

  const costFalseReject = parseFloat(document.getElementById("cost-false-reject").value);
  const costMissedFraud = parseFloat(document.getElementById("cost-missed-fraud").value);
  const costManualReview = parseFloat(document.getElementById("cost-manual-review").value);

  // 2. Fuse score
  const finalScore = 
    (FUSION_WEIGHTS.graph * graphScore) +
    (FUSION_WEIGHTS.image * imageScore) +
    (FUSION_WEIGHTS.behavior * behaviorScore);

  // 3. Expected costs
  const expectedCosts = {
    approve: finalScore * costMissedFraud,
    reject: (1.0 - finalScore) * costFalseReject,
    review: costManualReview
  };

  // 4. Decision
  let decisionKey = "approve";
  let minCost = expectedCosts.approve;
  
  if (expectedCosts.reject < minCost) {
    minCost = expectedCosts.reject;
    decisionKey = "reject";
  }
  if (expectedCosts.review < minCost) {
    minCost = expectedCosts.review;
    decisionKey = "review";
  }

  const decisionMap = {
    "approve": "APPROVE",
    "reject": "REJECT",
    "review": "MANUAL_REVIEW",
  };

  const decision = decisionMap[decisionKey];

  // Reasoning
  const sortedCosts = Object.entries(expectedCosts).sort((a, b) => a[1] - b[1]);
  const cheapest = sortedCosts[0];
  const second = sortedCosts[1];
  const margin = second[1] - cheapest[1];

  const reasoning = `At final_score=${finalScore.toFixed(3)}, expected cost of ${decisionMap[cheapest[0]]} ($${cheapest[1].toFixed(1)}) is lowest, $${margin.toFixed(1)} cheaper than next best option (${decisionMap[second[0]]}, $${second[1].toFixed(1)}).`;

  // 5. Update UI
  document.getElementById("fusion-placeholder").classList.add("hidden");
  const resultsPanel = document.getElementById("fusion-results");
  resultsPanel.classList.remove("hidden");

  // Banner
  const banner = document.getElementById("fusion-decision-banner");
  const bannerText = document.getElementById("fusion-decision-text");
  
  banner.className = "fusion-decision-banner"; // reset classes
  if (decision === "APPROVE") {
    banner.classList.add("approve");
  } else if (decision === "REJECT") {
    banner.classList.add("deny");
  } else {
    banner.classList.add("friction");
  }
  bannerText.textContent = decision;

  // Final score
  document.getElementById("fusion-final-val").textContent = finalScore.toFixed(3);
  
  // Breakdown bars
  const breakdownContainer = document.getElementById("fusion-breakdown-bars");
  breakdownContainer.innerHTML = `
    <div class="breakdown-bar">
      <div class="breakdown-label">Graph <span>(${(FUSION_WEIGHTS.graph * 100).toFixed(0)}%)</span></div>
      <div class="breakdown-track">
        <div class="breakdown-fill graph" style="width: ${graphScore * 100}%"></div>
      </div>
      <div class="breakdown-val">${graphScore.toFixed(2)}</div>
    </div>
    <div class="breakdown-bar">
      <div class="breakdown-label">Image <span>(${(FUSION_WEIGHTS.image * 100).toFixed(0)}%)</span></div>
      <div class="breakdown-track">
        <div class="breakdown-fill image" style="width: ${imageScore * 100}%"></div>
      </div>
      <div class="breakdown-val">${imageScore.toFixed(2)}</div>
    </div>
    <div class="breakdown-bar">
      <div class="breakdown-label">Behavior <span>(${(FUSION_WEIGHTS.behavior * 100).toFixed(0)}%)</span></div>
      <div class="breakdown-track">
        <div class="breakdown-fill behavior" style="width: ${behaviorScore * 100}%"></div>
      </div>
      <div class="breakdown-val">${behaviorScore.toFixed(2)}</div>
    </div>
  `;

  // Expected cost bars
  const costMax = Math.max(...Object.values(expectedCosts));
  const expectedCostBars = document.getElementById("expected-cost-bars");
  expectedCostBars.innerHTML = `
    <div class="cost-bar-row">
      <div class="cost-label">Approve</div>
      <div class="cost-track">
        <div class="cost-fill approve ${decision === 'APPROVE' ? 'winner' : ''}" style="width: ${(expectedCosts.approve / costMax * 100)}%"></div>
      </div>
      <div class="cost-val">$${expectedCosts.approve.toFixed(1)}</div>
    </div>
    <div class="cost-bar-row">
      <div class="cost-label">Review</div>
      <div class="cost-track">
        <div class="cost-fill review ${decision === 'MANUAL_REVIEW' ? 'winner' : ''}" style="width: ${(expectedCosts.review / costMax * 100)}%"></div>
      </div>
      <div class="cost-val">$${expectedCosts.review.toFixed(1)}</div>
    </div>
    <div class="cost-bar-row">
      <div class="cost-label">Reject</div>
      <div class="cost-track">
        <div class="cost-fill reject ${decision === 'REJECT' ? 'winner' : ''}" style="width: ${(expectedCosts.reject / costMax * 100)}%"></div>
      </div>
      <div class="cost-val">$${expectedCosts.reject.toFixed(1)}</div>
    </div>
  `;

  document.getElementById("fusion-reasoning").textContent = reasoning;
  
  // Render sensitivity
  renderSensitivityChart(costFalseReject, costMissedFraud, costManualReview);
}

function findEffectiveThresholds(cfr, cmf, cmr, resolution = 100) {
  const bands = [];
  let currentDecision = null;
  let bandStart = 0.0;

  for (let i = 0; i <= resolution; i++) {
    const score = i / resolution;
    const costs = {
      approve: score * cmf,
      reject: (1.0 - score) * cfr,
      review: cmr
    };
    
    let decision = "approve";
    let minCost = costs.approve;
    if (costs.reject < minCost) { minCost = costs.reject; decision = "reject"; }
    if (costs.review < minCost) { minCost = costs.review; decision = "review"; }
    
    if (decision !== currentDecision) {
      if (currentDecision !== null) {
        bands.push({ decision: currentDecision, start: bandStart, end: score });
      }
      currentDecision = decision;
      bandStart = score;
    }
  }
  bands.push({ decision: currentDecision, start: bandStart, end: 1.0 });
  return bands;
}

function renderSensitivityChart(cfr, cmf, cmr) {
  const bands = findEffectiveThresholds(cfr, cmf, cmr, 500);
  const container = document.getElementById("sensitivity-chart");
  
  let html = `<div class="sensitivity-track">`;
  bands.forEach(b => {
    const width = (b.end - b.start) * 100;
    const decClass = b.decision === 'approve' ? 'approve' : (b.decision === 'reject' ? 'deny' : 'friction');
    const label = b.decision === 'approve' ? 'APPROVE' : (b.decision === 'reject' ? 'REJECT' : 'REVIEW');
    html += `<div class="sensitivity-band ${decClass}" style="width: ${width}%">
               <span class="band-label">${label}</span>
               <span class="band-range">[${b.start.toFixed(2)}, ${b.end.toFixed(2)})</span>
             </div>`;
  });
  html += `</div>`;
  
  container.innerHTML = html;
}

function initRiskFusion() {
  const ids = ["graph-score-slider", "image-score-slider", "behavior-score-slider"];
  ids.forEach(id => {
    const slider = document.getElementById(id);
    const val = document.getElementById(id.replace("-slider", "-val"));
    if (slider && val) {
      slider.addEventListener("input", (e) => {
        val.textContent = (parseInt(e.target.value, 10) / 100.0).toFixed(2);
        // Auto-run fusion on input
        runRiskFusion();
      });
    }
  });

  const fuseBtn = document.getElementById("fuse-btn");
  if (fuseBtn) {
    fuseBtn.addEventListener("click", runRiskFusion);
  }

  // Add listeners to cost inputs
  const costIds = ["cost-false-reject", "cost-missed-fraud", "cost-manual-review"];
  costIds.forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener("input", runRiskFusion);
    }
  });
}

// ═══════════════════ GRAPH & IMAGE SIMULATORS ═══════════════════

const GRAPH_SCENARIOS = {
  serial: {
    graph_score: 0.88,
    flags: [
      { code: "SHARED_ATTRIBUTES", message: "Shares address, payment with 5 other account(s)", severity: 1.0 },
      { code: "LARGE_CLUSTER", message: "Belongs to a connected cluster of 6 accounts", severity: 0.6 },
      { code: "DENSE_COMMUNITY", message: "Part of a tightly-knit group where most accounts are interlinked", severity: 1.0 }
    ],
    nodes: [
      { id: "cust_dc7da6f5", type: "customer", risk: 0.88, isTarget: true },
      { id: "cust_ecdd20be", type: "customer", risk: 0.88 },
      { id: "cust_a58875e6", type: "customer", risk: 0.88 },
      { id: "cust_43878b52", type: "customer", risk: 0.88 },
      { id: "cust_292d3163", type: "customer", risk: 0.88 },
      { id: "cust_be51f80b", type: "customer", risk: 0.88 }
    ],
    edges: [
      { source: "cust_ecdd20be", target: "cust_292d3163", weight: 5.0 },
      { source: "cust_ecdd20be", target: "cust_43878b52", weight: 5.0 },
      { source: "cust_ecdd20be", target: "cust_a58875e6", weight: 5.0 },
      { source: "cust_ecdd20be", target: "cust_be51f80b", weight: 5.0 },
      { source: "cust_ecdd20be", target: "cust_dc7da6f5", weight: 5.0 },
      { source: "cust_a58875e6", target: "cust_292d3163", weight: 5.0 },
      { source: "cust_a58875e6", target: "cust_43878b52", weight: 5.0 },
      { source: "cust_a58875e6", target: "cust_be51f80b", weight: 5.0 },
      { source: "cust_a58875e6", target: "cust_dc7da6f5", weight: 5.0 },
      { source: "cust_43878b52", target: "cust_292d3163", weight: 5.0 },
      { source: "cust_43878b52", target: "cust_be51f80b", weight: 5.0 },
      { source: "cust_43878b52", target: "cust_dc7da6f5", weight: 5.0 },
      { source: "cust_292d3163", target: "cust_be51f80b", weight: 5.0 },
      { source: "cust_292d3163", target: "cust_dc7da6f5", weight: 5.0 },
      { source: "cust_dc7da6f5", target: "cust_be51f80b", weight: 5.0 }
    ]
  },
  wardrobing: {
    graph_score: 0.25,
    flags: [
      { code: "SHARED_DEVICE", message: "Shares device with 1 other account", severity: 0.3 }
    ],
    nodes: [
      { id: "cust_ward_1", type: "customer", risk: 0.25, isTarget: true },
      { id: "cust_ward_2", type: "customer", risk: 0.1 }
    ],
    edges: [
      { source: "cust_ward_1", target: "cust_ward_2", weight: 1.0 }
    ]
  },
  genuine: {
    graph_score: 0.05,
    flags: [
      { code: "ISOLATED_NODE", message: "Customer has no suspicious shared links.", severity: 0.0 }
    ],
    nodes: [
      { id: "cust_gen_1", type: "customer", risk: 0.05, isTarget: true }
    ],
    edges: []
  },
  inr: {
    graph_score: 0.40,
    flags: [
      { code: "SHARED_ADDRESS", message: "Multiple INR claims at same address", severity: 0.5 }
    ],
    nodes: [
      { id: "cust_inr_1", type: "customer", risk: 0.40, isTarget: true },
      { id: "cust_inr_2", type: "customer", risk: 0.50 }
    ],
    edges: [
      { source: "cust_inr_1", target: "cust_inr_2", weight: 2.0 }
    ]
  }
};

function renderGraphSVG(nodes, edges) {
  const width = 300;
  const height = 300;
  const cx = width / 2;
  const cy = height / 2;
  const radius = 90;
  
  const nodePositions = {};
  nodes.forEach((node, i) => {
    if (nodes.length === 1) {
      nodePositions[node.id] = { x: cx, y: cy };
    } else {
      const angle = (i / nodes.length) * 2 * Math.PI - Math.PI / 2;
      nodePositions[node.id] = {
        x: cx + radius * Math.cos(angle),
        y: cy + radius * Math.sin(angle)
      };
    }
  });

  let svg = `<svg viewBox="0 0 ${width} ${height}" style="width:100%; height:100%;">`;
  
  edges.forEach(edge => {
    const p1 = nodePositions[edge.source];
    const p2 = nodePositions[edge.target];
    if (p1 && p2) {
      svg += `<line x1="${p1.x}" y1="${p1.y}" x2="${p2.x}" y2="${p2.y}" stroke="var(--border-subtle)" stroke-width="1.5" opacity="0.6"/>`;
    }
  });

  nodes.forEach(node => {
    const p = nodePositions[node.id];
    const fill = node.isTarget ? 'var(--accent-purple)' : (node.risk > 0.5 ? 'var(--accent-red)' : 'var(--accent-amber)');
    const r = node.isTarget ? 14 : 10;
    const stroke = node.isTarget ? '#fff' : 'rgba(255,255,255,0.2)';
    svg += `<circle class="graph-node" cx="${p.x}" cy="${p.y}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="2" style="filter: drop-shadow(0 0 8px ${fill}66); transition: all 0.3s ease;" />`;
    svg += `<text x="${p.x}" y="${p.y + r + 14}" fill="var(--text-secondary)" font-size="10" font-family="var(--font-mono)" text-anchor="middle">${node.id.slice(0, 8)}</text>`;
  });
  
  svg += `</svg>`;
  return svg;
}

function calculateGraphScore() {
    const scenarioKey = window.currentScenario || "genuine";
    const data = GRAPH_SCENARIOS[scenarioKey];
    
    if (!data) return;

    const score = data.graph_score;
    let flagsHTML = "";
    
    data.flags.forEach(flag => {
        const severityClass = flag.severity >= 0.8 ? "critical" : (flag.severity >= 0.4 ? "high" : (flag.severity > 0 ? "medium" : "low"));
        flagsHTML += `<div class="flag-item ${severityClass}"><span class="flag-code">${flag.code}</span><span class="flag-msg">${flag.message}</span></div>`;
    });

    const scoreTextEl = document.getElementById("graph-score-text");
    if (scoreTextEl) scoreTextEl.textContent = score.toFixed(2);
    
    const offset = 326.7 * (1 - score);
    const fill = document.getElementById("graph-score-fill");
    if (fill) fill.style.strokeDashoffset = offset;
    
    const flagsList = document.getElementById("graph-flags-list");
    if (flagsList) flagsList.innerHTML = flagsHTML;
    
    const svgContainer = document.getElementById("graph-svg-container");
    if (svgContainer) {
        svgContainer.style.background = "transparent";
        svgContainer.innerHTML = renderGraphSVG(data.nodes, data.edges);
    }

    // Update Fusion Engine
    const slider = document.getElementById("graph-score-slider");
    if (slider) {
       slider.value = score * 100;
       const valEl = document.getElementById("graph-score-val");
       if(valEl) valEl.textContent = score.toFixed(2);
       if (typeof runRiskFusion === 'function') runRiskFusion();
    }
}

function calculateImageScore() {
    const hsv = parseInt(document.getElementById('image-input-hsv').value, 10) / 100.0 || 0;
    const orb = parseInt(document.getElementById('image-input-orb').value, 10) || 0;
    const edge = parseInt(document.getElementById('image-input-edge').value, 10) / 100.0 || 0;
    const noise = parseInt(document.getElementById('image-input-noise').value, 10) / 100.0 || 0;
    const exact = document.getElementById('image-input-exact').checked;

    let manipulation = Math.min((edge + noise) * 2, 1.0);
    let score = (0.7 * hsv) + (0.3 * manipulation);
    let flagsHTML = "";

    if (exact) {
        score = Math.max(score, 0.95);
        flagsHTML += `<div class="flag-item critical"><span class="flag-code">EXACT_DUPLICATE_IMAGE</span><span class="flag-msg">Identical image previously submitted.</span></div>`;
    } else if (hsv >= 0.8 || orb >= 25) {
        score = Math.max(score, 0.80);
        score = Math.min(score, 0.94);
    } else {
        score = Math.min(manipulation, 0.75);
    }

    if (!exact && hsv >= 0.85) {
        flagsHTML += `<div class="flag-item high"><span class="flag-code">NEAR_DUPLICATE_IMAGE</span><span class="flag-msg">High similarity (${Math.round(hsv*100)}%) with prior claim.</span></div>`;
    }
    if (!exact && orb >= 25) {
        flagsHTML += `<div class="flag-item high"><span class="flag-code">FEATURE_MATCH_REUSE</span><span class="flag-msg">${orb} matching ORB keypoints found.</span></div>`;
    }
    if (edge > 0.3) {
        flagsHTML += `<div class="flag-item medium"><span class="flag-code">SPLICING_EDGE_ANOMALY</span><span class="flag-msg">Unnatural spatial edge disparity (${Math.round(edge*100)}%).</span></div>`;
    }
    if (noise > 0.3) {
        flagsHTML += `<div class="flag-item medium"><span class="flag-code">UNNATURAL_NOISE_VARIANCE</span><span class="flag-msg">Non-uniform spatial noise variance (${Math.round(noise*100)}%).</span></div>`;
    }
    
    if (flagsHTML === "") {
        flagsHTML = `<div class="flag-item low"><span class="flag-code">CLEAN_IMAGE</span><span class="flag-msg">No duplicates or manipulation artifacts found.</span></div>`;
        score = Math.min(score, 0.20);
    }

    document.getElementById("image-score-text").textContent = score.toFixed(2);
    const offset = 326.7 * (1 - score);
    const fill = document.getElementById("image-score-fill");
    if (fill) fill.style.strokeDashoffset = offset;
    
    const flagsList = document.getElementById("image-flags-list");
    if (flagsList) flagsList.innerHTML = flagsHTML;

    // Update Fusion Engine
    const slider = document.getElementById("image-score-slider");
    if (slider) {
       slider.value = score * 100;
       const valEl = document.getElementById("image-score-val");
       if (valEl) valEl.textContent = score.toFixed(2);
       runRiskFusion();
    }
}

// Initial calculation
document.addEventListener("DOMContentLoaded", () => {
  setTimeout(() => {
    calculateGraphScore();
    calculateImageScore();
  }, 100);
});


