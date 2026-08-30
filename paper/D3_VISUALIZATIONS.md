# GCL D3.js Visualization Specifications

This document provides detailed specifications for interactive D3.js visualizations to accompany the GCL paper on a web presentation.

---

## Hero Section Recommendations

### Primary Hero: The Punishment Paradox

**Why this visualization?**
- Most counterintuitive finding in the paper
- Immediately captures attention ("wait, more punishment = less cooperation?")
- Clean visual narrative that's easy to understand
- Invites exploration ("why does this happen?")

**Visual Design:**
```
┌─────────────────────────────────────────────────────────────┐
│                    THE PUNISHMENT PARADOX                    │
│                                                              │
│  Cooperation                                                 │
│  Rate (%)                                                    │
│     80 ┤ ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│        │  ╲                                                  │
│     60 ┤   ╲●                                                │
│        │     ╲                                               │
│     40 ┤       ╲●                                            │
│        │         ╲                                           │
│     20 ┤           ╲●                                        │
│        │             ╲●                                      │
│      0 ┼──────┬──────┬──────┬──────┬──────                   │
│        0%    25%    50%    75%   100%                        │
│              Consequence Severity →                          │
│                                                              │
│  "Increasing punishment DECREASES cooperation"               │
│  r = -0.951, p < 0.001                                       │
└─────────────────────────────────────────────────────────────┘
```

### Secondary Hero: Dunbar Scaling

**Why this visualization?**
- Connects to familiar concept (Dunbar's number)
- Shows interesting trade-off (efficiency vs specialization)
- Interactive exploration potential
- Implications for AI system design

**Visual Design:**
```
┌─────────────────────────────────────────────────────────────┐
│                    DUNBAR-LIKE SCALING                       │
│                                                              │
│  Efficiency                              Specialization      │
│     30% ┤                                           ┤ 100%   │
│         │ ●                                    ━━━━━●        │
│     20% ┤  ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┤ 80%   │
│         │    ╲                           ╱          │        │
│     10% ┤      ╲●━━━━━━━━━━━━━━━━━━━━━━●╱           ┤ 60%   │
│         │        ╲                   ╱              │        │
│      0% ┼─────────●━━━━━━━━━━━━━━━●─────────────────┼ 40%   │
│         5    10    50   100   150   200                      │
│                   Population Size →                          │
│                        ↑                                     │
│                   ~100 agents                                │
│               "Dunbar-like limit"                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Detailed Visualization Specifications

### Figure 1: Hart-Moore Validation

**Type:** Grouped Bar Chart with Error Bars

**Data:**
```javascript
const hartMooreData = {
  investment: [
    {type: 'Complete', value: 0.527, ci: 0.038, color: '#16a34a'},
    {type: 'GCL', value: 0.409, ci: 0.053, color: '#2563eb'},
    {type: 'Incomplete', value: 0.197, ci: 0.050, color: '#dc2626'}
  ],
  holdups: [
    {type: 'Complete', value: 0.100, ci: 0.019, color: '#16a34a'},
    {type: 'GCL', value: 0.242, ci: 0.045, color: '#2563eb'},
    {type: 'Incomplete', value: 0.382, ci: 0.057, color: '#dc2626'}
  ]
};
```

**Interactions:**
- Hover: Show exact values and statistical significance
- Click: Toggle between investment and hold-up views
- Animation: Bars grow from zero on load

**D3 Implementation Notes:**
```javascript
// Key elements
const margin = {top: 40, right: 30, bottom: 60, left: 60};
const width = 600 - margin.left - margin.right;
const height = 400 - margin.top - margin.bottom;

// Scales
const x0 = d3.scaleBand().domain(['Investment', 'Hold-ups']).range([0, width]).padding(0.2);
const x1 = d3.scaleBand().domain(['Complete', 'GCL', 'Incomplete']).range([0, x0.bandwidth()]).padding(0.05);
const y = d3.scaleLinear().domain([0, 0.6]).range([height, 0]);

// Error bars using line elements
// Significance stars above bars
```

---

### Figure 2: Punishment Paradox (HERO)

**Type:** Line Chart with Confidence Bands

**Data:**
```javascript
const punishmentData = [
  {consequence: 0.00, cooperation: 0.727, ci: 0.038},
  {consequence: 0.25, cooperation: 0.600, ci: 0.047},
  {consequence: 0.50, cooperation: 0.497, ci: 0.050},
  {consequence: 0.75, cooperation: 0.382, ci: 0.057},
  {consequence: 1.00, cooperation: 0.289, ci: 0.053}
];
```

**Visual Elements:**
1. **Main line**: Thick, animated draw from left to right
2. **Confidence band**: Semi-transparent area around line
3. **Data points**: Circles at each measurement
4. **Trend annotation**: "r = -0.951" with arrow
5. **Background gradient**: Subtle red gradient increasing left to right

**Interactions:**
- Hover on points: Tooltip with exact values
- Hover on line: Show correlation coefficient
- Click: Expand to show methodology

**Animation Sequence:**
1. Axes appear (0.3s)
2. Confidence band fades in (0.3s)
3. Line draws from left to right (1.5s)
4. Points pop in sequentially (0.5s)
5. Annotation fades in (0.3s)

**D3 Implementation:**
```javascript
// Line generator with curve
const line = d3.line()
  .x(d => x(d.consequence))
  .y(d => y(d.cooperation))
  .curve(d3.curveMonotoneX);

// Area generator for confidence band
const area = d3.area()
  .x(d => x(d.consequence))
  .y0(d => y(d.cooperation - d.ci))
  .y1(d => y(d.cooperation + d.ci))
  .curve(d3.curveMonotoneX);

// Animated line drawing
path.attr('stroke-dasharray', totalLength + ' ' + totalLength)
    .attr('stroke-dashoffset', totalLength)
    .transition()
    .duration(1500)
    .attr('stroke-dashoffset', 0);
```

---

### Figure 3: Redemption Effect

**Type:** Slope Graph / Before-After Comparison

**Data:**
```javascript
const redemptionData = {
  without: {value: 0.393, ci: 0.060, label: 'Without Redemption'},
  with: {value: 0.600, ci: 0.075, label: 'With Redemption'},
  improvement: '+52.7%'
};
```

**Visual Design:**
```
Without          With
Redemption    Redemption
    │              │
   39%────────────60%
    │      ↗       │
    │   +52.7%     │
    │              │
```

**Interactions:**
- Hover: Show confidence intervals
- Animation: Line draws connecting the two points

---

### Figure 4: Dunbar Scaling (HERO)

**Type:** Dual-Axis Line Chart with Interactive Slider

**Data:**
```javascript
const dunbarData = [
  {population: 5, efficiency: 0.227, gini: 0.354},
  {population: 10, efficiency: 0.265, gini: 0.621},
  {population: 20, efficiency: 0.206, gini: 0.777},
  {population: 50, efficiency: 0.184, gini: 0.903},
  {population: 100, efficiency: 0.114, gini: 0.952},
  {population: 150, efficiency: 0.090, gini: 0.973},
  {population: 200, efficiency: 0.075, gini: 0.980}
];
```

**Visual Elements:**
1. **Left axis**: Efficiency (blue line, declining)
2. **Right axis**: Specialization/Gini (orange line, increasing)
3. **Crossover annotation**: Vertical line at ~100 agents
4. **Interactive slider**: Select population to highlight
5. **Tooltip**: Shows both values at selected point

**Interactions:**
- Slider: Drag to explore different population sizes
- Hover: Show exact values for both metrics
- Click on "100": Expand explanation of Dunbar limit

**D3 Implementation:**
```javascript
// Dual scales
const yLeft = d3.scaleLinear().domain([0, 0.3]).range([height, 0]);
const yRight = d3.scaleLinear().domain([0, 1]).range([height, 0]);
const x = d3.scaleLog().domain([5, 200]).range([0, width]);

// Slider implementation
const slider = d3.sliderBottom(x)
  .min(5).max(200)
  .step(1)
  .default(100)
  .on('onchange', val => updateHighlight(val));
```

---

### Figure 5: Protocol Comparison

**Type:** Bubble Scatter Plot

**Data:**
```javascript
const protocolData = [
  {name: 'CNP', efficiency: 0.824, messages: 109.5, success: 1.0},
  {name: 'FIPA-ACL', efficiency: 0.818, messages: 190.2, success: 1.0},
  {name: 'MARL-IQL', efficiency: 0.755, messages: 0, success: 1.0},
  {name: 'Auction', efficiency: 0.661, messages: 175.6, success: 1.0},
  {name: 'GCL', efficiency: 0.645, messages: 84.0, success: 1.0}
];
```

**Visual Design:**
- X-axis: Message complexity
- Y-axis: Efficiency
- Bubble size: Success rate (all 100% here)
- Color: Protocol type (GCL highlighted in blue)

**Interactions:**
- Hover: Show protocol details
- Click: Expand to show protocol description

---

### Figure 6: Trust Network

**Type:** Force-Directed Graph

**Data Structure:**
```javascript
const networkData = {
  nodes: [
    {id: 'A1', specialization: 0.8, tasks: 45},
    {id: 'A2', specialization: 0.6, tasks: 32},
    // ... more agents
  ],
  links: [
    {source: 'A1', target: 'A2', trust: 0.85},
    {source: 'A1', target: 'A3', trust: 0.72},
    // ... more trust relationships
  ]
};
```

**Visual Elements:**
- Nodes: Circles sized by task count, colored by specialization
- Edges: Lines with thickness proportional to trust level
- Clusters: Emerge naturally from force simulation

**Interactions:**
- Drag nodes: Reposition in simulation
- Hover node: Highlight connections
- Click node: Show agent details
- Zoom/pan: Explore large networks

**D3 Implementation:**
```javascript
const simulation = d3.forceSimulation(nodes)
  .force('link', d3.forceLink(links).id(d => d.id).strength(d => d.trust))
  .force('charge', d3.forceManyBody().strength(-100))
  .force('center', d3.forceCenter(width / 2, height / 2))
  .force('collision', d3.forceCollide().radius(d => nodeRadius(d.tasks)));
```

---

## Color Palette

```css
:root {
  /* Primary colors */
  --gcl-blue: #2563eb;           /* GCL brand, primary actions */
  --gcl-blue-light: #60a5fa;     /* Hover states */
  --gcl-blue-dark: #1d4ed8;      /* Active states */
  
  /* Semantic colors */
  --gcl-success: #16a34a;        /* Positive results, complete contracts */
  --gcl-warning: #ea580c;        /* Caution, paradox findings */
  --gcl-danger: #dc2626;         /* Failures, incomplete contracts */
  
  /* Neutral colors */
  --gcl-gray-100: #f3f4f6;       /* Background */
  --gcl-gray-300: #d1d5db;       /* Borders */
  --gcl-gray-500: #6b7280;       /* Secondary text */
  --gcl-gray-900: #111827;       /* Primary text */
  
  /* Chart-specific */
  --efficiency-color: #2563eb;   /* Blue for efficiency */
  --specialization-color: #ea580c; /* Orange for specialization */
  --confidence-band: rgba(37, 99, 235, 0.2); /* Transparent blue */
}
```

---

## Responsive Design

### Breakpoints

```javascript
const breakpoints = {
  mobile: 480,
  tablet: 768,
  desktop: 1024,
  wide: 1440
};

// Responsive chart dimensions
function getChartDimensions() {
  const containerWidth = document.getElementById('chart-container').clientWidth;
  
  if (containerWidth < breakpoints.mobile) {
    return {width: containerWidth - 20, height: 250, margin: {top: 20, right: 20, bottom: 40, left: 40}};
  } else if (containerWidth < breakpoints.tablet) {
    return {width: containerWidth - 40, height: 300, margin: {top: 30, right: 30, bottom: 50, left: 50}};
  } else {
    return {width: Math.min(containerWidth - 60, 800), height: 400, margin: {top: 40, right: 40, bottom: 60, left: 60}};
  }
}
```

### Mobile Adaptations

1. **Punishment Paradox**: Simplify to bar chart on mobile
2. **Dunbar Scaling**: Replace slider with tap-to-select
3. **Trust Network**: Reduce node count, increase touch targets
4. **All charts**: Larger fonts, simplified legends

---

## Animation Guidelines

### Timing

```javascript
const timing = {
  fast: 200,      // Hover effects
  medium: 500,    // Transitions
  slow: 1000,     // Initial animations
  stagger: 100    // Sequential element delays
};
```

### Easing

```javascript
const easing = {
  default: d3.easeCubicOut,
  bounce: d3.easeElastic,
  smooth: d3.easeSinInOut
};
```

### Scroll-Triggered Animations

```javascript
// Intersection Observer for scroll animations
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      animateChart(entry.target.id);
    }
  });
}, {threshold: 0.3});

document.querySelectorAll('.chart-container').forEach(el => observer.observe(el));
```

---

## Accessibility

### Requirements

1. **Color contrast**: All text meets WCAG AA (4.5:1 ratio)
2. **Keyboard navigation**: All interactive elements focusable
3. **Screen readers**: ARIA labels for all chart elements
4. **Reduced motion**: Respect `prefers-reduced-motion`

### Implementation

```javascript
// Check for reduced motion preference
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const animationDuration = prefersReducedMotion ? 0 : timing.slow;

// ARIA labels
svg.attr('role', 'img')
   .attr('aria-label', 'Chart showing punishment paradox: cooperation decreases as consequences increase');

// Keyboard navigation
circles.attr('tabindex', 0)
       .on('keydown', (event, d) => {
         if (event.key === 'Enter' || event.key === ' ') {
           showTooltip(d);
         }
       });
```

---

## Data Export

Each visualization should support data export:

```javascript
function exportData(chartId, format = 'csv') {
  const data = getChartData(chartId);
  
  if (format === 'csv') {
    const csv = d3.csvFormat(data);
    downloadFile(csv, `${chartId}.csv`, 'text/csv');
  } else if (format === 'json') {
    const json = JSON.stringify(data, null, 2);
    downloadFile(json, `${chartId}.json`, 'application/json');
  }
}
```

---

## Performance Optimization

1. **Canvas fallback**: For networks > 500 nodes
2. **Debounced resize**: Prevent excessive redraws
3. **Virtual scrolling**: For large data tables
4. **Lazy loading**: Load charts as they enter viewport

```javascript
// Debounced resize handler
let resizeTimeout;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimeout);
  resizeTimeout = setTimeout(() => {
    updateAllCharts();
  }, 250);
});
```
