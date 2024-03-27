#!/usr/bin/env python
import rospy
from llm_vision_control.srv import ObjectPose

class ObjectPoseClient:
    def __init__(self):
        rospy.init_node('object_pose_client')
        self.object_name = rospy.get_param('~object_name', 'cup')
        self.service_name = rospy.get_param('~service_name', 'object_pose')
        self.interval = rospy.get_param('~interval', 10.0)
        self.timer = rospy.Timer(rospy.Duration(self.interval), self.timer_callback)

    def timer_callback(self, event):
        rospy.wait_for_service(self.service_name)
        try:
            object_pose_service = rospy.ServiceProxy(self.service_name, ObjectPose)
            response = object_pose_service(self.object_name)
            rospy.loginfo(f"Object pose: {response.object_pose}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

if __name__ == '__main__':
    client = ObjectPoseClient()
    rospy.spin()