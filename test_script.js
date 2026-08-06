// Mock DOM
const document = {
  addEventListener: () => {},
  querySelectorAll: () => [],
  getElementById: () => ({ innerHTML: '', classList: { add: ()=>{}, remove: ()=>{} }, style: {} })
};
const window = { GRAPH_SCENARIOS: {} };

// Case Queue Data
const CQ_CASES = [
  {
    id: "CLM-2026-001", scenario: "genuine",
    customer: "Priya Sharma", account: "3-year account", location: "Chennai, TN",
    category: "apparel_casual", value: 5399, reason: "wrong_size",
    behaviorScore: 0.08, imageScore: null, graphScore: 0.05,
    orderHistory: [0,0,0,0,1],
    explanation: ["reason 1"]
  }
];
const FUSION_W = { behavior: 0.25, image: 0.40, graph: 0.35 };

function cqComposite(c) {
  let usedW = FUSION_W.behavior;
  let score = c.behaviorScore * FUSION_W.behavior;
  if (c.imageScore !== null) { score += c.imageScore * FUSION_W.image; usedW += FUSION_W.image; }
  if (c.graphScore !== null) { score += c.graphScore * FUSION_W.graph; usedW += FUSION_W.graph; }
  return score / usedW;
}
console.log("Composite: " + cqComposite(CQ_CASES[0]));
