#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
import cv2
from cv_bridge import CvBridge
import base64
import requests
from PIL import Image as PILImage, ImageDraw, ImageFont
import numpy as np
from io import BytesIO
from llm_vision_control.srv import ObjectPose, ObjectPoseResponse
from pathlib import Path
import xml.etree.ElementTree as ET

class LLMVisionNode:
    def __init__(self):
        rospy.init_node('llm_vision_node')
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)
        self.bridge = CvBridge()
        self.api_key = rospy.get_param('~api_key', '')  # Retrieve the API key from the parameter server
        self.latest_image = None
        self.latest_depth = None
        self.camera_info = None
        self.object_pose_service = rospy.Service('object_pose', ObjectPose, self.object_pose_callback)

    def image_callback(self, msg):
        self.latest_image = msg

    def depth_callback(self, msg):
        self.latest_depth = msg

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def object_pose_callback(self, req):
        if self.latest_image is None or self.latest_depth is None or self.camera_info is None:
            print(self.latest_image)
            print(self.latest_depth)
            print(self.camera_info)
            rospy.logwarn("No image, depth, or camera info available yet.")
            return ObjectPoseResponse(PoseStamped())

        object_name = req.object_name

        # Convert ROS image message to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='passthrough')

        # Convert OpenCV image to PIL format
        pil_image = PILImage.fromarray(cv_image)

        # Add grid to the image using PIL
        pil_image = self.add_grid(pil_image)

        # Save the image with grid to a BytesIO object
        image_data = BytesIO()
        pil_image.save(image_data, format='PNG')
        image_data.seek(0)

        # Encode the image as base64
        base64_image = self.encode_image(image_data)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": 
f"""Given the image with a grid that may contain the object: '{object_name}', 
You must respond in the following format:
```
<scene-description>
A brief description of the objects in the scene and what grid cells they are in.
</scene-description>
<grid-cell>
gridCell of the object
</grid-cell>
```

If the object you were asked to find is not in the description, you must return with none in the grid-cell xml block.

Some example responses might be:

for a `human face`
```
<scene-description>
A cluttered labratory with various people in it.
The peoples faces are in squares a6, b7, e4.
</scene-description>
<grid-cell>
b7
</grid-cell>
```

or for a professor
```
<scene-description>
A classroom with a computer at a6 and a professor at e4.
</scene-description>
<grid-cell>
e4
</grid-cell>
```

or for a cow
<scene-description>
There are several cars on e5, f2, and a3 in a busy city intersection
</scene-description>
<grid-cell>
none
</grid-cell>
```
"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1000
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        print("LLM RESPONSE: {}".format(response.json()))

        # Process the LLM response and extract the grid coordinates
        llm_response = response.json()['choices'][0]['message']['content']
        _, parsed_response = self.llm_response_parser(llm_response)
        if parsed_response.lower() == 'none':
            rospy.logwarn("LLM response returned 'None'. Skipping the rest of the logic.")
            return ObjectPoseResponse(PoseStamped())
        grid_coordinates = [parsed_response]

        # Convert grid coordinates to pixel values
        pixel_coordinates = self.grid_to_pixel(grid_coordinates, pil_image)

        # Draw bounding boxes around the object coordinates
        draw = ImageDraw.Draw(pil_image)
        for coord in pixel_coordinates:
            x, y = coord
            draw.rectangle([(x-5, y-5), (x+5, y+5)], outline=(255, 0, 0), width=2)

        # Save the marked image to a file
        marked_image_path = '/home/douglas/chaser_ws/src/llm_vision_control/src/scripts/marked_image.png'
        pil_image.save(marked_image_path)

        # Show the marked image
        cv2.imshow('Marked Image', cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Convert depth image to numpy array
        depth_image = self.bridge.imgmsg_to_cv2(self.latest_depth, desired_encoding='passthrough')

        # Get the depth value at the object coordinates
        x, y = pixel_coordinates[0]  # Assuming only one object coordinate
        depth_value = depth_image[y, x]

        # Calculate the 3D point in the camera frame
        fx = self.camera_info.K[0]
        fy = self.camera_info.K[4]
        cx = self.camera_info.K[2]
        cy = self.camera_info.K[5]

        z = depth_value / 1000.0  # Convert from millimeters to meters
        x = (x - cx) * z / fx
        y = (y - cy) * z / fy

        # Create and return the PoseStamped message as the service response
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.camera_info.header.frame_id
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.pose.position.x = x
        pose_stamped.pose.position.y = y
        pose_stamped.pose.position.z = z
        pose_stamped.pose.orientation.w = 1.0  # Default orientation (identity quaternion)
        return ObjectPoseResponse(pose_stamped)

    def llm_response_parser(self, response: str):
        try:
            # Wrap the XML response in a parent tag
            wrapped_response = f"<llm_response>{response}</llm_response>"

            # Parse the wrapped XML response
            root = ET.fromstring(wrapped_response)

            # Find the <scene-description> element and extract its text content
            scene_description = root.find('scene-description').text.strip()

            # Find the <grid-cell> element and extract its text content
            grid_cell = root.find('grid-cell').text.strip()

            # Return the scene description and grid cell as a tuple
            return scene_description, grid_cell

        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return "none", "none"

    def encode_image(self, image_data):
        return base64.b64encode(image_data.getvalue()).decode('utf-8')

    def grid_to_pixel(self, grid_coordinates, pil_image):
        width, height = pil_image.size
        grid_size = 10
        cell_width = width // grid_size
        cell_height = height // grid_size

        pixel_coordinates = []
        for coord in grid_coordinates:
            col = ord(coord[0]) - ord('A')
            row = int(coord[1])
            x = col * cell_width + cell_width // 2
            y = row * cell_height + cell_height // 2
            pixel_coordinates.append((x, y))

        return pixel_coordinates

    def encode_image(self, image_data):
        return base64.b64encode(image_data.getvalue()).decode('utf-8')

    def add_grid(self, image):
        width, height = image.size
        grid_size = 10
        cell_width = width // grid_size
        cell_height = height // grid_size

        draw = ImageDraw.Draw(image)

        # Use the pathlib library to get this files folder:
        this_folder = Path(__file__).parent.resolve()
        font = ImageFont.truetype(str(this_folder / "Arial.ttf"), 40)

        for i in range(grid_size):
            # Draw vertical lines
            draw.line([(i * cell_width, 0), (i * cell_width, height)], fill=(255, 255, 255), width=2)
            # Draw horizontal lines
            draw.line([(0, i * cell_height), (width, i * cell_height)], fill=(255, 255, 255), width=2)

            # Add labels on the top (A to J)
            label = chr(ord('A') + i)
            draw.text((i * cell_width + (cell_width) // 2, 5), label, fill=(255, 255, 255), font=font)

            # Add labels on the left (0 to 9)
            label = str(i)
            draw.text((5, i * cell_height + (cell_height) // 2), label, fill=(255, 255, 255), font=font)
        return image

if __name__ == '__main__':
    node = LLMVisionNode()
    rospy.spin()