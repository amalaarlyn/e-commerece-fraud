import re

# Read the old index.html
with open('index.html', 'r') as f:
    old_html = f.read()

# Extract old sections
def extract_section(section_id):
    pattern = f'<section id="{section_id}">(.*?)</section>'
    match = re.search(pattern, old_html, re.DOTALL)
    if match:
        return match.group(1)
    return ""

behavior_content = extract_section("behavior-section")
graph_content = extract_section("graph-section")
image_content = extract_section("image-section")
fusion_content = extract_section("fusion-section")
pipeline_content = extract_section("pipeline-section")
results_content = extract_section("results-section")

# Apply basic tailwind conversions to the old content
def apply_tailwind(html):
    # Convert section headers
    html = re.sub(r'<div class="section-header">', r'<div class="mb-lg border-b border-outline-variant pb-md mb-lg">', html)
    html = re.sub(r'<div class="section-tag">(.*?)</div>', r'<div class="text-primary text-label font-bold tracking-widest uppercase mb-sm">\1</div>', html)
    html = re.sub(r'<h2>(.*?)</h2>', r'<h2 class="font-headline-lg text-headline-lg text-on-surface">\1</h2>', html)
    html = re.sub(r'<p>(.*?)</p>', r'<p class="text-body-md text-on-surface-variant">\1</p>', html, count=1)
    
    # Buttons
    html = re.sub(r'class="score-btn"', r'class="bg-primary-container text-on-primary-container py-md px-lg rounded-xl font-headline-sm hover:opacity-90 transition-opacity mt-md inline-flex"', html)
    html = re.sub(r'class="add-order-btn"', r'class="bg-surface-variant text-on-surface py-sm px-md rounded-lg font-body-sm hover:bg-surface-bright transition-colors mt-md inline-flex"', html)
    
    return f'<div class="glass-card rounded-[16px] p-xl">{html}</div>'

behavior_tw = apply_tailwind(behavior_content)
graph_tw = apply_tailwind(graph_content)
image_tw = apply_tailwind(image_content)
fusion_tw = apply_tailwind(fusion_content)
dashboard_tw = apply_tailwind(pipeline_content + "<br><br>" + results_content)

# The user's snippet (we need to inject the tab logic into it)
# We will read it from a file or define it as a string
