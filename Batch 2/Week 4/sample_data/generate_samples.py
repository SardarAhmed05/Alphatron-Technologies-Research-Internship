"""
Generates rich offline test HTML pages for EDRIC testing without internet access.
"""

import os
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))

HTML_ECOMMERCE = """<!DOCTYPE html>
<html lang="en">
<head><title>TechNova Electronics - Flagship Store</title></head>
<body>
    <header><h1>TechNova Premium Gadgets</h1><nav><a href="#">Home</a></nav></header>
    <main>
        <h2>Featured Product Inventory (Q3 2026)</h2>
        <table border="1">
            <thead>
                <tr>
                    <th>Product Name</th>
                    <th>Category</th>
                    <th>Original Price</th>
                    <th>Discounted Price</th>
                    <th>Stock Units</th>
                    <th>Customer Rating</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>QuantumBook Pro 16</td>
                    <td>Laptops</td>
                    <td>,499.00</td>
                    <td>,199.00</td>
                    <td>42</td>
                    <td>4.9 / 5.0 (1,280 reviews)</td>
                </tr>
                <tr>
                    <td>NeuralEar Buds Ultra</td>
                    <td>Audio</td>
                    <td>.00</td>
                    <td>.00</td>
                    <td>150</td>
                    <td>4.7 / 5.0 (890 reviews)</td>
                </tr>
                <tr>
                    <td>VisionMatrix VR Headset</td>
                    <td>Wearables</td>
                    <td>,199.00</td>
                    <td>.00</td>
                    <td>18</td>
                    <td>4.8 / 5.0 (420 reviews)</td>
                </tr>
                <tr>
                    <td>PulseSmart Health Ring</td>
                    <td>Wearables</td>
                    <td>.00</td>
                    <td>.00</td>
                    <td>85</td>
                    <td>4.6 / 5.0 (610 reviews)</td>
                </tr>
            </tbody>
        </table>
    </main>
    <footer><p>© 2026 TechNova Inc. All rights reserved.</p></footer>
</body>
</html>"""

HTML_TECH_NEWS = """<!DOCTYPE html>
<html lang="en">
<head><title>AI Breakthrough Daily - August 2026 Report</title></head>
<body>
    <h1>AI Breakthrough Daily: Frontier Models & Agentic Workflows</h1>
    <article>
        <h2>LangGraph 2.0 Achieves 99.4% State Transition Reliability</h2>
        <p>By Sarah Lin, Chief Editor | Published: August 15, 2026</p>
        <p>Engineers at leading AI labs reported breakthrough benchmark results showing multi-agent reflection loops in LangGraph reduced hallucination rates from 14.2% down to less than 0.8%.</p>
        <p>Key metrics revealed an average latency reduction of 3.2x when using lightweight state checkpointing.</p>
    </article>
    <article>
        <h2>Quantum TPU Deployment Reaches 100,000 Nodes</h2>
        <p>By Alex Rivera | Published: August 18, 2026</p>
        <p>Global cloud infrastructure has integrated over 100,000 neuromorphic nodes, enabling sub-millisecond inference for 70B parameter models.</p>
    </article>
</body>
</html>"""

HTML_COMPANY_PROFILE = """<!DOCTYPE html>
<html lang="en">
<head><title>Aegis Health AI - Corporate Overview</title></head>
<body>
    <h1>Aegis Health AI Corporation</h1>
    <p><strong>Headquarters:</strong> Boston, MA</p>
    <p><strong>Founded:</strong> 2023</p>
    <p><strong>CEO:</strong> Dr. Elena Vance</p>
    <p><strong>Total Funding:</strong> .5 Million (Series B led by Sequoia Capital)</p>
    <p><strong>Valuation:</strong>  Million</p>
    <p><strong>Key Products:</strong> CardioVision AI (FDA Cleared), NeuroScan Assistant</p>
    <p><strong>Active Hospital Deployments:</strong> 120 Medical Centers across 14 Countries</p>
</body>
</html>"""


def generate_all():
    with open(os.path.join(SAMPLE_DIR, "sample_ecommerce.html"), "w", encoding="utf-8") as f:
        f.write(HTML_ECOMMERCE)
    with open(os.path.join(SAMPLE_DIR, "sample_tech_news.html"), "w", encoding="utf-8") as f:
        f.write(HTML_TECH_NEWS)
    with open(os.path.join(SAMPLE_DIR, "sample_company_profile.html"), "w", encoding="utf-8") as f:
        f.write(HTML_COMPANY_PROFILE)
    print("[SUCCESS] Generated sample test files in sample_data/")


if __name__ == "__main__":
    generate_all()
