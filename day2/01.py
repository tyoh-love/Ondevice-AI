# YOLO Evolution Timeline Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig, ax = plt.subplots(figsize=(14, 8))

# YOLO version information
yolo_versions = [
    {'version': 'YOLOv1', 'year': 2015, 'fps': 45, 'mAP': 63.4, 'creator': 'Joseph Redmon',
     'feature': 'Start of real-time object detection'},
    {'version': 'YOLOv2', 'year': 2016, 'fps': 67, 'mAP': 76.8, 'creator': 'Joseph Redmon',
     'feature': 'Introduction of Anchor Box'},
    {'version': 'YOLOv3', 'year': 2018, 'fps': 65, 'mAP': 82.0, 'creator': 'Joseph Redmon',
     'feature': 'Multi-scale prediction'},
    {'version': 'YOLOv4', 'year': 2020, 'fps': 65, 'mAP': 85.4, 'creator': 'Alexey Bochkovskiy',
     'feature': 'Bag of Freebies'},
    {'version': 'YOLOv5', 'year': 2020, 'fps': 140, 'mAP': 84.0, 'creator': 'Ultralytics',
     'feature': 'PyTorch implementation'},
    {'version': 'YOLOv6', 'year': 2022, 'fps': 150, 'mAP': 85.5, 'creator': 'Meituan',
     'feature': 'Industrial optimization'},
    {'version': 'YOLOv7', 'year': 2022, 'fps': 155, 'mAP': 86.7, 'creator': 'WongKinYiu',
     'feature': 'E-ELAN structure'},
    {'version': 'YOLOv8', 'year': 2023, 'fps': 160, 'mAP': 87.0, 'creator': 'Ultralytics',
     'feature': 'Anchor-free'},
    {'version': 'YOLOv9', 'year': 2024, 'fps': 170, 'mAP': 88.0, 'creator': 'Community',
     'feature': 'PGI & GELAN'},
    {'version': 'YOLOv10', 'year': 2024, 'fps': 175, 'mAP': 88.5, 'creator': 'THU',
     'feature': 'NMS-free'},
    {'version': 'YOLO11', 'year': 2024, 'fps': 180, 'mAP': 89.0, 'creator': 'Ultralytics',
     'feature': 'C3k2 block'},
    {'version': 'YOLOv12', 'year': 2025, 'fps': 200, 'mAP': 90.5, 'creator': 'OpenAI',
     'feature': 'ViT hybrid'}
]

# Calculate position by year
years = [v['year'] for v in yolo_versions]
positions = np.linspace(1, 10, len(yolo_versions))

# Draw timeline
for i, (version, pos) in enumerate(zip(yolo_versions, positions)):
    # Draw box
    if i < 3:
        color = '#FF6B6B'  # Redmon era
    elif i < 8:
        color = '#4ECDC4'  # Community era
    else:
        color = '#45B7D1'  # Modern era

    box = FancyBboxPatch((pos-0.4, 2), 0.8, 2,
                        boxstyle="round,pad=0.1",
                        facecolor=color,
                        edgecolor='black',
                        linewidth=2,
                        alpha=0.8)
    ax.add_patch(box)

    # Version info
    ax.text(pos, 3.5, version['version'], ha='center', fontweight='bold', fontsize=10)
    ax.text(pos, 3.2, f"{version['year']}", ha='center', fontsize=8)
    ax.text(pos, 2.8, f"FPS: {version['fps']}", ha='center', fontsize=8)
    ax.text(pos, 2.5, f"mAP: {version['mAP']}%", ha='center', fontsize=8)

    # Feature
    ax.text(pos, 1.5, version['feature'], ha='center', fontsize=8,
            wrap=True, style='italic')

    # Creator
    ax.text(pos, 0.5, version['creator'], ha='center', fontsize=7,
            color='gray')

# Draw arrows
for i in range(len(positions)-1):
    ax.arrow(positions[i]+0.4, 3, positions[i+1]-positions[i]-0.8, 0,
            head_width=0.2, head_length=0.1, fc='gray', ec='gray', alpha=0.5)

# Styling
ax.set_xlim(0, 11)
ax.set_ylim(0, 5)
ax.set_title('YOLO Evolution: 2015-2025', fontsize=16, fontweight='bold', pad=20)

# Legend
redmon_patch = mpatches.Patch(color='#FF6B6B', label='Joseph Redmon Era')
community_patch = mpatches.Patch(color='#4ECDC4', label='Community-driven')
modern_patch = mpatches.Patch(color='#45B7D1', label='Next-generation YOLO')
ax.legend(handles=[redmon_patch, community_patch, modern_patch], loc='upper right')

ax.axis('off')
plt.tight_layout()
plt.show()