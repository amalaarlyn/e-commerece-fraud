import re

with open('template.html', 'r') as f:
    template = f.read()

with open('user_queue_snippet.html', 'r') as f:
    queue_snippet = f.read()

with open('index.html', 'r') as f:
    old_html = f.read()

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

def apply_tailwind(html):
    if not html:
        return ""
    # Headers
    html = re.sub(r'<div class="section-header">', r'<div class="mb-lg border-b border-outline-variant pb-md mb-lg">', html)
    html = re.sub(r'<div class="section-tag">(.*?)</div>', r'<div class="text-primary text-label font-bold tracking-widest uppercase mb-sm">\1</div>', html)
    html = re.sub(r'<h2>(.*?)</h2>', r'<h2 class="font-headline-lg text-headline-lg text-on-surface">\1</h2>', html)
    html = re.sub(r'<p>(.*?)</p>', r'<p class="text-body-md text-on-surface-variant">\1</p>', html, count=1)
    
    # Buttons
    html = re.sub(r'class="score-btn"', r'class="score-btn bg-primary-container text-on-primary-container py-md px-lg rounded-xl font-headline-sm hover:opacity-90 transition-opacity mt-md inline-flex"', html)
    html = re.sub(r'class="add-order-btn"', r'class="add-order-btn bg-surface-variant text-on-surface py-sm px-md rounded-lg font-body-sm hover:bg-surface-bright transition-colors mt-md inline-flex"', html)
    
    return f'<div class="glass-card rounded-[16px] p-xl">{html}</div>'

behavior_tw = apply_tailwind(behavior_content)
graph_tw = apply_tailwind(graph_content)
image_tw = apply_tailwind(image_content)
fusion_tw = apply_tailwind(fusion_content)
dashboard_tw = apply_tailwind(pipeline_content + "<br><br>" + results_content)

new_html = template.replace('<!-- INSERT_QUEUE_HERE -->', queue_snippet)
new_html = new_html.replace('<!-- INSERT_DASHBOARD_HERE -->', dashboard_tw)
new_html = new_html.replace('<!-- INSERT_BEHAVIOR_HERE -->', behavior_tw)
new_html = new_html.replace('<!-- INSERT_GRAPH_HERE -->', graph_tw)
new_html = new_html.replace('<!-- INSERT_IMAGE_HERE -->', image_tw)
new_html = new_html.replace('<!-- INSERT_FUSION_HERE -->', fusion_tw)

with open('index_new.html', 'w') as f:
    f.write(new_html)

print("index_new.html created successfully.")
