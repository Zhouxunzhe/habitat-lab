import re
output = "robot_1:pick_[38,-1,75]\nrobot_2:nav_to_point_[0,0]"
pattern = re.compile(r'robot_(\d+):(\w+)_\[(\-?\d+),(\-?\d+)(?:,(\-?\d+))?\]')
matches = re.findall(pattern,output)
for match in matches:
    robot,action,x,y,z = match
    z = z if z else None
    print(f"Robot:{robot},Action:{action},x:{x},y:{y},z:{z}")
    
    