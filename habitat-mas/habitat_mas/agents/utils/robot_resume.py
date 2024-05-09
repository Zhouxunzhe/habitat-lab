class RobotResume:
    """
    This class is used to store the resume of a robot. It contains the capabilities of the robot.
    - robot_id: the id of the robot
    - robot_type: the type of the robot
    - robot mobility capabilities: the mobility capabilities of the robot, including its motion type(Fixed, Legged, Wheeled, Tracked, Aerial, etc.).
    - robot perception capabilities: the perception capabilities of the robot, including its sensor type(Lidar, Camera, Depth Camera, etc.), and its perceivable area. 
    - robot manipulation capabilities: the manipulation capabilities of the robot, including its end-effector type(Gripper, Suction, etc.), and its manipulation workspace.
    """
    def __init__(self, robot_id: str, raw_text:str) -> None:
        self.robot_id = robot_id
        self.raw_text = raw_text
        
        # parsing results
        self.parse_success = False 
        self.robot_id = None
        self.robot_type = None
        self.robot_mobility_capabilities = None
        self.robot_perception_capabilities = None
        self.robot_manipulation_capabilities = None
        
        # parse the resume
        # self.parse_resume()
        
        
    def parse_resume(self):
        """
        Parse the resume of the robot.
        """
        raise NotImplementedError
    