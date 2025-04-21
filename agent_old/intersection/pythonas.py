import xml.etree.ElementTree as ET
import random

# Define the list of possible guiShapes (excluding "emergency")
gui_shapes = ["passenger", "bus", "truck", "motorcycle", "bicycle", "tram"]

# Filepath to the XML file
input_file = r"f:\P\HK8\Deep-QLearning-Agent-for-Traffic-Signal-Control\TLCS\intersection\realold\Deep-QLearning-Agent-for-Traffic-Signal-Control\TLCS\intersection\episode_routes.rou.xml"
output_file = r"f:\P\HK8\Deep-QLearning-Agent-for-Traffic-Signal-Control\TLCS\intersection\realold\Deep-QLearning-Agent-for-Traffic-Signal-Control\TLCS\intersection\episode_routes_random_guiShape.rou.xml"

# Parse the XML file
tree = ET.parse(input_file)
root = tree.getroot()

# Iterate through all <vType> elements and assign a random guiShape
for vtype in root.findall("vType"):
    random_shape = random.choice(gui_shapes)
    vtype.set("guiShape", random_shape)

# Write the modified XML to a new file
tree.write(output_file, encoding="utf-8", xml_declaration=True)

print(f"Random guiShapes have been assigned and saved to {output_file}")