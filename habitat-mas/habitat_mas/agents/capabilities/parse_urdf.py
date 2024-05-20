import os
from openai import OpenAI
from urchin import URDF, Link, Joint
from habitat_mas.agents.llm_api_keys import get_openai_client
import networkx as nx
from networkx.readwrite.text import generate_network_text

default_tasks = """
Cross-floor object rearrangement (pick and place objects from one floor to another);
Cooperative Perception for manipulation: Robots need to search for and perceive the object to get geometric information, and then manipulate the object;
Home arrangement: Robots need to rearrange the furniture in the home, especially the objects in abnormal positions like on the high shelf and under table.
"""


        
# Function to read and parse URDF file
def parse_urdf(file_path):
    robot = URDF.load(file_path)
    return robot

def generate_tree_text_with_edge_types(graph, root):
    """
    Generate a tree structure text including node names and edge types.
    
    Parameters:
    graph (nx.DiGraph): The directed graph representing the tree structure.
    root (node): The root node of the tree.
    
    Returns:
    str: A string representing the tree structure with node names and edge types.
    """
    def dfs(node, depth):
        """
        Perform a depth-first search to generate the tree structure text.
        
        Parameters:
        node (node): The current node in the DFS traversal.
        depth (int): The current depth in the tree.
        
        Returns:
        str: A string representing the subtree rooted at the current node.
        """
        indent = "\t" * depth
        lines = [f"{indent}Link({node})"]
        
        for child in graph.successors(node):
            edge_type = graph[node][child].get('type', 'unknown')
            lines.append(f"{indent}├── Joint of type({edge_type})──Link({child})")
            lines.extend(dfs(child, depth + 1))
        
        return lines
    
    # Generate the tree structure text starting from the root
    tree_structure = dfs(root, 0)
    
    return "\n".join(tree_structure)

# Function to generate physics capabilities summary using OpenAI API
def query_llm_with_urdf(urdf: URDF, model_name="gpt-4o", task_text=default_tasks):
    # Extract relevant information from URDF
    robot_name = urdf.name

    urdf_text = ""
    
    # convert urdf to networkx DiGraph
    urdf_nx_graph = nx.DiGraph()
    
    # Add nodes and edges from urdf._G to the graph
    for link in urdf.links:
        # only keep the name and mass of the node
        # urdf_nx_graph.add_node(node.name, **node.__dict__)
        urdf_nx_graph.add_node(link.name)
    
    for joint in urdf.joints:
        # urdf_nx_graph.add_edge(edge.parent, edge.child, **edge.__dict__)
        urdf_nx_graph.add_edge(joint.parent, joint.child, type=joint.joint_type) 
    
    # for line in generate_network_text(urdf_nx_graph):
    #     urdf_text += line + "\n"
    for node in urdf_nx_graph.nodes:
        if node == urdf.base_link.name:
            root_node = node
    urdf_text = generate_tree_text_with_edge_types(urdf_nx_graph, root_node)
    
    system_prompt = """
You are a robot urdf structure reasoner. You will be given a robot's urdf tree-structured text, and you need to provide a summary of the robot's physics capabilities.
Please pay attention to the task and summarize the mobility, perception, and manipulation capabilities of the robot that are relevant to the task.
The response should be a JSON object with each capability as a dictionary, containing a summary field:
{
    "mobility": {"summary": ...},
    "perception": {"summary": ...},
    "manipulation": {"summary": ...}
}
"""
    
    prompt = f"""
The robot's name is {robot_name}. Here is the tree structure of the robot's URDF:
{urdf_text}
The robot task includes: {task_text}
Please provide a summary of the robot's physics capabilities based on this information and task.
"""
    
    # print(prompt)
    
    # Call OpenAI API to generate the summary
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        response_format={ "type": "json_object"}
    )
    summary = response.choices[0].message.content
    return summary

def generate_physics_summary(urdf_file_path:str, save_path=None):
    urdf = parse_urdf(urdf_file_path)
    summary = query_llm_with_urdf(urdf)
    print("Physics Capabilities Summary:")
    print(summary)
    if save_path:
        with open(save_path, "w") as f:
            f.write(summary)

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(cur_dir, "../../../../data")
    urdf_file_path = os.path.join(data_dir, "robots/hab_spot_arm/urdf/hab_spot_arm.urdf")
    save_path = os.path.join(cur_dir, "../../data/robot_resume/hab_spot_arm.txt")
    generate_physics_summary(urdf_file_path, save_path)