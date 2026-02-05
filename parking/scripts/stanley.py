#!/usr/bin/env python3
import rospy, math
import numpy as np
from nav_msgs.msg import Path
from std_msgs.msg import Int16, ColorRGBA
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from jsk_rviz_plugins.msg import OverlayText
from tf.transformations import euler_from_quaternion


K = 1.0     # lateral gain
L = 0.8
V = 1.0     # long vel
k_h = 1.0   # heading P gain
k_l = 1.0   # lateral P gain

class Stanley:
    def __init__(self):
        rospy.init_node("stanley")
        rospy.Subscriber("/stanley_path", Path, self.cb_path, queue_size=1)
        self.steer_pub = rospy.Publisher("/parking_stanley_steer", Int16, queue_size=1)
        self.marker_pub = rospy.Publisher("/stanley_debug", Marker, queue_size=10)
        self.debug_text_pub = rospy.Publisher("/stanley_debug_text",OverlayText,queue_size=1,latch=True)
        self.herr_pub = rospy.Publisher("/herror",Int16,queue_size=1)
        self.path = None
        
        
    def cb_path(self, msg: Path):

        if not msg.poses:
            self.path = None
            self.clear_stanley_markers()
            self.steer_pub.publish(Int16(0))
            rospy.logwarn_throttle(1.0, "Stanley path invalid → clearing RViz markers")
            return

        self.path_yaw = self.get_path_yaw(msg)

        # ===== 추가: yaw NaN 방어 =====
        if not np.isfinite(self.path_yaw):
            rospy.logwarn_throttle(1.0, "[STANLEY] path_yaw is NaN → ignore path")
            self.path = None
            self.steer_pub.publish(Int16(0))
            return
        # ==============================

        path = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in msg.poses],
            dtype=float
        )

        # ===== 추가: path NaN 방어 =====
        if not np.all(np.isfinite(path)):
            rospy.logwarn_throttle(1.0, "[STANLEY] path contains NaN → ignore")
            self.path = None
            self.steer_pub.publish(Int16(0))
            return
        # ==============================

        # 기존 로직 그대로
        dir_pts = path[-1] - path[0]
        dir_yaw = np.array([math.cos(self.path_yaw), math.sin(self.path_yaw)])
        if np.dot(dir_pts, dir_yaw) < 0:
            path = path[::-1]

        self.path = path
        self.stanley()



    def stanley(self):
        
        if self.path is None:
            return
        motion_yaw = math.pi

        dists = np.linalg.norm(self.path, axis=1)
        idx = np.argmin(dists)
        
        ref = self.path[idx]

        t = np.array([math.cos(self.path_yaw),math.sin(self.path_yaw)]) # path 방향 단위벡터
        n = np.array([-t[1], t[0]])   # 경로 법선 벡터

        lateral_error = np.dot(n, -ref)
        
        if lateral_error > 0:
            lateral_debug_msg = f"ego is on leftside of path, error: {lateral_error}"
        else:
            lateral_debug_msg = f"ego is on rightside of path, error: {lateral_error}"
        
        heading_error = self.wrap(self.path_yaw - motion_yaw)
        
        if heading_error > 0:
            heading_debug_msg = f"path direction is on leftside of ego direction, error: {heading_error}"
        else:
            heading_debug_msg = f"path direction is on rightside of ego direction, error: {heading_error}"
        
        rospy.loginfo_throttle(0.5,f"{lateral_debug_msg}\n{heading_debug_msg}")
        
        """
        lateral error:
        양수
        → 경로 방향(path_yaw)을 바라봤을 때 경로가 차량(ref) 기준 오른쪽 -> steer left(+)

        음수
        → 경로 방향(path_yaw)을 바라봤을 때 경로가 차량(ref) 기준 왼쪽 -> steer right(-)
        
        heading error:
        양수
        → 경로 방향이 차량 이동 방향 기준 왼쪽 -> steer right(-)
        
        음수
        → 경로 방향이 차량 이동 방향 기준 오른쪽 -> steer left(+)
        """
        
        steer = (- k_h * heading_error+ k_l * math.atan2(K * lateral_error, V))
        
        invalid = (
            not np.isfinite(lateral_error) or
            not np.isfinite(heading_error) or
            not np.isfinite(steer)
        )
        
        if invalid:
            rospy.logwarn_throttle(1.0, "[STANLEY] NaN detected → steer=0 (debug kept)")
            steer = 0.0
            
        MAX_STEER_RAD = math.radians(22.5)
        steer = max(min(steer, MAX_STEER_RAD), -MAX_STEER_RAD)
        self.steer_pub.publish(Int16(int(math.degrees(steer))))
        print(f"pub: {int(math.degrees(steer))}")


        # for debug
        
        self.last_lateral_error = lateral_error
        self.last_heading_error = heading_error
        heror_msg = Int16()
        heror_msg.data = int(math.degrees(self.last_heading_error))
        self.herr_pub.publish(heror_msg)
        self.last_motion_yaw = motion_yaw
        self.last_path_yaw = self.path_yaw
        self.last_steer = steer
        
        self.publish_markers(ref, steer)
        self.publish_debug_vectors(ref)
        self.publish_debug_text()   


    def get_path_yaw(self, msg: Path):
        # parking.py에서 orientation을 z,w만 넣어줬으니(roll=pitch=0) yaw는 이렇게 복원 가능
        q = msg.poses[0].pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        return yaw


    def wrap(self, ang):
        # range : [-π, π)
        return (ang + math.pi) % (2*math.pi) - math.pi

    # ---------------- markers ----------------
    def publish_markers(self, closest, steer):
        now = rospy.Time.now()

        # 1. closest point
        m = Marker()
        m.header.frame_id = "laser"
        m.header.stamp = now
        m.ns = "closest"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = closest[0]
        m.pose.position.y = closest[1]
        m.scale.x = m.scale.y = m.scale.z = 0.2
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 1.0
        self.marker_pub.publish(m)

        # 2. predicted reverse curve
        self.publish_curve(steer, now)


    def publish_curve(self, steer, stamp):
        m = Marker()
        m.header.frame_id = "laser"
        m.header.stamp = stamp
        m.ns = "predicted_curve"
        m.id = 1
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.05
        m.color.r = 1.0
        m.color.a = 1.0

        pts = []
        ds = 0.10          # 진행거리 step (m)
        N = 10             # 점 개수 (길이 = ds*N)
        Lwb = L            # wheelbase

        # 거의 직진이면 그냥 직선으로 표시
        if abs(steer) < 1e-3:
            for i in range(1, N+1):
                s = i * ds
                x = -s
                y = 0.0
                pts.append(Point(x=x, y=y, z=0.0))
            m.points = pts
            self.marker_pub.publish(m)
            return

        # 곡률(κ) 기반 적분: x,y,psi를 한 스텝씩 누적
        kappa = math.tan(steer) / Lwb   # curvature
        x, y = 0.0, 0.0
        psi = 0.0                       # 차량 기준 heading

        # 후진이면 진행거리 부호만 반대로
        v_sign = -1.0

        for i in range(N):
            # 작은 구간에서 heading 변화
            dpsi = v_sign * kappa * ds
            psi += dpsi

            # 현재 heading 방향으로 이동
            x += v_sign * ds * math.cos(psi)
            y += v_sign * ds * math.sin(psi)

            pts.append(Point(x=x, y=y, z=0.0))

        m.points = pts
        self.marker_pub.publish(m)


    def publish_debug_vectors(self, ref):
        now = rospy.Time.now()

        # ---------- motion yaw (blue) ----------
        self.publish_arrow(
            ns="motion_yaw",
            mid=10,
            yaw=self.last_motion_yaw,
            color=(0.2, 0.4, 1.0),
            length=1.2,
            stamp=now
        )

        # ---------- path yaw (green) ----------
        self.publish_arrow(
            ns="path_yaw",
            mid=11,
            yaw=self.last_path_yaw,
            color=(0.2, 1.0, 0.2),
            length=1.5,
            stamp=now
        )

        # ---------- lateral error (red line) ----------
        self.publish_lateral_error_line(ref, now)

        # ---------- heading error arc ----------
        self.publish_heading_arc(now)


        # ---------------- utils ----------------
    
    def publish_arrow(self, ns, mid, yaw, color, length, stamp):
        m = Marker()
        m.header.frame_id = "laser"
        m.header.stamp = stamp
        m.ns = ns
        m.id = mid
        m.type = Marker.ARROW
        m.action = Marker.ADD

        p0 = Point(0.0, 0.0, 0.0)
        p1 = Point(length * math.cos(yaw), length * math.sin(yaw), 0.0)

        m.points = [p0, p1]

        m.scale.x = 0.05
        m.scale.y = 0.10
        m.scale.z = 0.10

        m.color.r, m.color.g, m.color.b = color
        m.color.a = 1.0

        self.marker_pub.publish(m)


    def publish_lateral_error_line(self, ref, stamp):
        m = Marker()
        m.header.frame_id = "laser"
        m.header.stamp = stamp
        m.ns = "lateral_error"
        m.id = 12
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD

        m.scale.x = 0.04
        m.color.r = 0.7
        m.color.g = 0.2
        m.color.b = 0.8
        m.color.a = 1.0

        p0 = Point(0.0, 0.0, 0.0)
        p1 = Point(ref[0], ref[1], 0.0)

        m.points = [p0, p1]
        self.marker_pub.publish(m)
        
        
    def publish_heading_arc(self, stamp):
        m = Marker()
        m.header.frame_id = "laser"
        m.header.stamp = stamp
        m.ns = "heading_error_arc"
        m.id = 13
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD

        m.scale.x = 0.04
        m.color.r = 0.7
        m.color.g = 0.2
        m.color.b = 0.8
        m.color.a = 1.0

        R = 0.8
        N = 20
        a0 = self.last_motion_yaw
        a1 = self.last_motion_yaw + self.last_heading_error

        for i in range(N + 1):
            t = a0 + (a1 - a0) * i / N
            m.points.append(Point(R * math.cos(t), R * math.sin(t), 0.0))

        self.marker_pub.publish(m)


    def publish_debug_text(self):

        msg = OverlayText()

        msg.text = (
            f"[ STANLEY DEBUG ]\n\n"
            f"path yaw           : {math.degrees(self.last_path_yaw):+.1f} deg\n"
            f"motion yaw         : {math.degrees(self.last_motion_yaw):+.1f} deg\n"
            f"heading error      : {math.degrees(self.last_heading_error):+.1f} deg\n"
            f"heading effect     : {math.degrees(-k_h*self.last_heading_error):+.1f} deg\n\n"
            f"lateral error      : {self.last_lateral_error:+.2f} m\n"
            f"lateral effect     : "
            f"{math.degrees(k_l * math.atan2(K * self.last_lateral_error, V)):+.2f} deg\n\n"
            f"stanley steer cmd  : {math.degrees(self.last_steer):+.1f} deg"
        )

        # ---- 화면 고정 위치 & 스타일 ----
        msg.width  = 520
        msg.height = 360   
        msg.left   = 200   # 우측 상단
        msg.top    = 10
        msg.text_size = 10

        msg.fg_color = ColorRGBA(1.0, 1.0, 1.0, 1.0)   # 글자 흰색
        msg.bg_color = ColorRGBA(0.0, 0.0, 0.0, 0.0)   # 반투명 검정 배경

        self.debug_text_pub.publish(msg)



    def clear_stanley_markers(self):
        now = rospy.Time.now()

        def delete(ns, mid):
            m = Marker()
            m.header.frame_id = "laser"
            m.header.stamp = now
            m.ns = ns
            m.id = mid
            m.action = Marker.DELETE
            return m

        # ----- 모두 지우기 -----
        self.marker_pub.publish(delete("closest", 0))
        self.marker_pub.publish(delete("predicted_curve", 1))
        self.marker_pub.publish(delete("motion_yaw", 10))
        self.marker_pub.publish(delete("path_yaw", 11))
        self.marker_pub.publish(delete("lateral_error", 12))
        self.marker_pub.publish(delete("heading_error_arc", 13))


if __name__ == "__main__":
    try:
        Stanley()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass