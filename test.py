import requests
import base64

url = "http://localhost:10022/robot-chat"
headers = {
    "Content-Type": "application/json"
}


# image_paths = ["./video_dir_point/image_dir/episode_91/frame_1_agent_0_head_rgbFetchRobot_head_rgb.png", 
#                "./video_dir_point/image_dir/episode_91/frame_1_agent_0_head_rgbFetchRobot_head_rgb.png"]
# images = []

# for image_path in image_paths:
#     with open(image_path, 'rb') as image_file:
#         images.append(base64.b64encode(image_file.read()).decode('utf-8'))
images = []
data = {
    "id": "test123",
    "width": [100, 100],
    "height": [100, 100],
    "image": images,
    "prompt": (
        "You are now managing two robots. The actions they can perform are "
        "\"nav_to_point,\" \"pick,\" \"continue\" and \"place.\" They need to locate "
        "all the boxes in the scene, pick them up, and place them on the table next "
        "to the sofa in the living room. The first robot's view is <image>\n. "
        "The second robot's view is <image>\n. Please output the next action for each robot respectively."
    )
}

response = requests.post(url, headers=headers, json=data)

print(response.status_code)
print(response.json())