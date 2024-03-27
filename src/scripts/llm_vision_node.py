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
        # rospy.logwarn("api key: " + str(self.api_key))
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
            rospy.logwarn("No image, depth, or camera info available yet.")
            rospy.logwarn(f"latest image is none? {self.latest_image is None}")
            rospy.logwarn(f"latest depth is none? {self.latest_depth is None}")
            rospy.logwarn(f"latest camera_info is none? {self.camera_info is None}")

            return ObjectPoseResponse(PoseStamped())

        object_name = req.object_name

        # Convert ROS image message to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='passthrough')

        # Convert OpenCV image to PIL format
        pil_image = PILImage.fromarray(cv_image)

        # Add grid to the image using PIL
        pil_image_with_grid = self.add_grid(pil_image.copy())

        # Perform initial object localization
        base64_image = self.encode_image(pil_image_with_grid)
        grid_coordinates = self.localize_object(base64_image, object_name)

        if grid_coordinates.lower() == 'none':
            rospy.logwarn("Object not found in the initial localization.")
            return ObjectPoseResponse(PoseStamped())

        # Convert grid coordinates to pixel values in the original image
        initial_pixel_coordinates = self.grid_to_pixel(grid_coordinates, pil_image, pil_image)

        # Draw bounding boxes around the initial object coordinates
        self.draw_bounding_box(pil_image_with_grid, initial_pixel_coordinates)

        # Save and display the initial marked image
        self.save_and_display_image(pil_image_with_grid, "initial_marked_image.png")

        # Crop the image around the initial object location
        cropped_image = self.crop_image(pil_image, grid_coordinates)

        # Add grid to the cropped image
        cropped_image_with_grid = self.add_grid(cropped_image.copy())

        # Perform refined object localization on the cropped image
        base64_cropped_image = self.encode_image(cropped_image_with_grid)
        refined_grid_coordinates = self.localize_object(base64_cropped_image, object_name)
        self.draw_selected_point(cropped_image, refined_grid_coordinates)

        if refined_grid_coordinates.lower() == 'none':
            rospy.logwarn("Object not found in the refined localization.")
            return ObjectPoseResponse(PoseStamped())

        # Convert refined grid coordinates to pixel values in the original image
        pixel_coordinates = self.grid_to_pixel(refined_grid_coordinates, pil_image, cropped_image)

        # Draw bounding boxes around the object coordinates
        self.draw_bounding_box(pil_image, pixel_coordinates)

        # Save and display the marked image
        self.save_and_display_image(pil_image, "refined_marked_image.png")

        # Calculate the 3D point in the camera frame
        pose_stamped = self.calculate_3d_point(pixel_coordinates)

        return ObjectPoseResponse(pose_stamped)

    def draw_selected_point(self, image, grid_coordinates):
        width, height = image.size
        grid_size = 10
        cell_width = width // grid_size
        cell_height = height // grid_size

        col = ord(grid_coordinates[0]) - ord('A')
        row = int(grid_coordinates[1])

        x = col * cell_width + cell_width // 2
        y = row * cell_height + cell_height // 2

        draw = ImageDraw.Draw(image)
        draw.ellipse([(x-5, y-5), (x+5, y+5)], fill=(255, 0, 0))
        # Show the image
        cv2.imshow("Selected Point", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def localize_object(self, image, object_name):
        # base64_image = self.encode_image(image)
        headers, payload = self.prepare_gpt4_request(image, object_name)
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        rospy.logwarn(response.json())
        llm_response = response.json()['choices'][0]['message']['content']
        _, parsed_response = self.llm_response_parser(llm_response)
        return parsed_response

    def crop_image(self, image, grid_coordinates):
        width, height = image.size
        grid_size = 10
        cell_width = width // grid_size
        cell_height = height // grid_size

        col = ord(grid_coordinates[0]) - ord('A')
        row = int(grid_coordinates[1])

        # Calculate the bounds for cropping (including half a square around the selected square)
        left = max(0, (col - 1) * cell_width)
        top = max(0, (row - 1) * cell_height)
        right = min(width, (col + 2) * cell_width)
        bottom = min(height, (row + 2) * cell_height)

        cropped_image = image.crop((left, top, right, bottom))
        return cropped_image

    def grid_to_pixel(self, grid_coordinates, original_image, reference_image):
        width, height = reference_image.size
        grid_size = 10
        cell_width = width // grid_size
        cell_height = height // grid_size

        col = ord(grid_coordinates[0]) - ord('A')
        row = int(grid_coordinates[1])

        x = col * cell_width + cell_width // 2
        y = row * cell_height + cell_height // 2

        if reference_image != original_image:
            # Calculate the top-left coordinates of the cropped image in the original image
            crop_left = (original_image.width - reference_image.width) // 2
            crop_top = (original_image.height - reference_image.height) // 2

            # Add the cropped image offset to the pixel coordinates
            x += crop_left
            y += crop_top

        return [(x, y)]

    def draw_bounding_box(self, image, pixel_coordinates):
        draw = ImageDraw.Draw(image)
        for coord in pixel_coordinates:
            x, y = coord
            draw.rectangle([(x-5, y-5), (x+5, y+5)], outline=(255, 0, 0), width=2)

    def save_and_display_image(self, image, image_name):
        marked_image_path = f'/home/douglas/chaser_ws/src/llm_vision_control/src/scripts/{image_name}'
        image.save(marked_image_path)
        cv2.imshow(image_name, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def calculate_3d_point(self, pixel_coordinates):
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

        # Create and return the PoseStamped message
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.camera_info.header.frame_id
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.pose.position.x = x
        pose_stamped.pose.position.y = y
        pose_stamped.pose.position.z = z
        pose_stamped.pose.orientation.w = 1.0  # Default orientation (identity quaternion)
        return pose_stamped

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

    def encode_image(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        # image.show()
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return base64_image

    def add_grid(self, image):
        width, height = image.size
        grid_size = 10
        cell_width = width // grid_size
        cell_height = height // grid_size

        draw = ImageDraw.Draw(image)

        # Use the pathlib library to get this files folder:
        this_folder = Path(__file__).parent.resolve()
        font_size = height // 13
        font = ImageFont.truetype(str(this_folder / "Arial.ttf"), font_size)

        for i in range(grid_size):
            # Draw vertical lines
            draw.line([(i * cell_width, 0), (i * cell_width, height)], fill=(255, 255, 255), width=1)
            # Draw horizontal lines
            draw.line([(0, i * cell_height), (width, i * cell_height)], fill=(255, 255, 255), width=1)

            # Add labels on the top (A to J)
            label = chr(ord('A') + i)
            draw.text((i * cell_width + (cell_width) // 2, 5), label, fill=(255, 255, 255), font=font)

            # Add labels on the left (0 to 9)
            label = str(i)
            draw.text((5, i * cell_height), label, fill=(255, 255, 255), font=font)
        return image


    def prepare_gpt4_request(self, base64_image, object_name):

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
        return headers, payload

if __name__ == '__main__':
    node = LLMVisionNode()
    rospy.spin()