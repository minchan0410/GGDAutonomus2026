#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
from collections import deque
from std_msgs.msg import Int32MultiArray, Float32, Int16

# =========================
# Tuning Defaults (params)
# =========================
DEFAULT_LANE_LINES_TOPIC = '/lane_lines_px'
DEFAULT_RATE_HZ = 20.0
DEFAULT_Y_MID = 240
DEFAULT_MA_WINDOW = 10
DEFAULT_EMA_WINDOW = 20
DEFAULT_INVALID_VALUE = float('nan')

# Thresholds (<=0 disables each check)
DEFAULT_LEFT_X_THRESH = -0.1
DEFAULT_LEFT_DX_THRESH = -20.0
DEFAULT_LEFT_DDX_THRESH = -50.0
DEFAULT_LEFT_MA_THRESH = -0.1
DEFAULT_LEFT_EMA_THRESH = -0.1
DEFAULT_LEFT_EMA_DX_THRESH = -0.1
DEFAULT_LEFT_EMA_DDX_THRESH = -5000.0

DEFAULT_RIGHT_X_THRESH = -0.1
DEFAULT_RIGHT_DX_THRESH = -0.0
DEFAULT_RIGHT_DDX_THRESH = -0.0
DEFAULT_RIGHT_MA_THRESH = -0.1
DEFAULT_RIGHT_EMA_THRESH = -0.1
DEFAULT_RIGHT_EMA_DX_THRESH = -0.1
DEFAULT_RIGHT_EMA_DDX_THRESH = -0.1

# Absolute comparison options
#값의 변화를 절대값으로 볼 것인지
DEFAULT_USE_ABS_X = False
DEFAULT_USE_ABS_DX = True
DEFAULT_USE_ABS_DDX = True
DEFAULT_USE_ABS_MA = False
DEFAULT_USE_ABS_EMA = False
DEFAULT_USE_ABS_EMA_DX = False
DEFAULT_USE_ABS_EMA_DDX = False


def _finite(val):
    return val is not None and not math.isnan(val) and not math.isinf(val)


def _get_param(ns, name, default):
    return rospy.get_param(ns + name, default)


class SideState:
    def __init__(self, ma_window, ema_window):
        self.ma_window = max(1, int(ma_window))
        self.ema_window = max(1, int(ema_window))
        self.ma_buf = deque(maxlen=self.ma_window)
        self.ema = None
        self.last_x = None
        self.last_dx = None
        self.last_ema = None
        self.last_ema_dx = None
        self.last_valid = False

    def reset(self):
        self.ma_buf.clear()
        self.ema = None
        self.last_x = None
        self.last_dx = None
        self.last_ema = None
        self.last_ema_dx = None
        self.last_valid = False

    def update(self, x, dt):
        # Returns x, dx, ddx, ma, ema, ema_dx, ema_ddx
        if not _finite(x) or dt <= 0.0:
            self.reset()
            return None

        dx = None
        ddx = None
        if self.last_x is not None:
            dx = (x - self.last_x) / dt
            if self.last_dx is not None:
                ddx = (dx - self.last_dx) / dt
        self.last_x = x
        self.last_dx = dx

        self.ma_buf.append(x)
        ma = sum(self.ma_buf) / float(len(self.ma_buf))

        # EMA with window-based alpha
        alpha = 2.0 / (self.ema_window + 1.0)
        if self.ema is None:
            self.ema = x
        else:
            self.ema = (alpha * x) + (1.0 - alpha) * self.ema

        ema_dx = None
        ema_ddx = None
        if self.last_ema is not None:
            ema_dx = (self.ema - self.last_ema) / dt
            if self.last_ema_dx is not None:
                ema_ddx = (ema_dx - self.last_ema_dx) / dt
        self.last_ema = self.ema
        self.last_ema_dx = ema_dx

        self.last_valid = True
        return x, dx, ddx, ma, self.ema, ema_dx, ema_ddx


class LaneChangeChecker:
    def __init__(self):
        rospy.init_node('lane_change_checker', anonymous=True)
        ns = '~'

        self.lane_lines_topic = _get_param(ns, 'lane_lines_topic', DEFAULT_LANE_LINES_TOPIC)
        self.rate_hz = float(_get_param(ns, 'rate_hz', DEFAULT_RATE_HZ))
        self.y_mid = int(_get_param(ns, 'y_mid', DEFAULT_Y_MID))
        self.ma_window = int(_get_param(ns, 'ma_window', DEFAULT_MA_WINDOW))
        self.ema_window = int(_get_param(ns, 'ema_window', DEFAULT_EMA_WINDOW))
        self.invalid_value = float(_get_param(ns, 'invalid_value', DEFAULT_INVALID_VALUE))

        # Thresholds (<=0 disables check)
        self.left_thresh = {
            'x': float(_get_param(ns, 'left_x_thresh', DEFAULT_LEFT_X_THRESH)),
            'dx': float(_get_param(ns, 'left_dx_thresh', DEFAULT_LEFT_DX_THRESH)),
            'ddx': float(_get_param(ns, 'left_ddx_thresh', DEFAULT_LEFT_DDX_THRESH)),
            'ma': float(_get_param(ns, 'left_ma_thresh', DEFAULT_LEFT_MA_THRESH)),
            'ema': float(_get_param(ns, 'left_ema_thresh', DEFAULT_LEFT_EMA_THRESH)),
            'ema_dx': float(_get_param(ns, 'left_ema_dx_thresh', DEFAULT_LEFT_EMA_DX_THRESH)),
            'ema_ddx': float(_get_param(ns, 'left_ema_ddx_thresh', DEFAULT_LEFT_EMA_DDX_THRESH)),
        }
        self.right_thresh = {
            'x': float(_get_param(ns, 'right_x_thresh', DEFAULT_RIGHT_X_THRESH)),
            'dx': float(_get_param(ns, 'right_dx_thresh', DEFAULT_RIGHT_DX_THRESH)),
            'ddx': float(_get_param(ns, 'right_ddx_thresh', DEFAULT_RIGHT_DDX_THRESH)),
            'ma': float(_get_param(ns, 'right_ma_thresh', DEFAULT_RIGHT_MA_THRESH)),
            'ema': float(_get_param(ns, 'right_ema_thresh', DEFAULT_RIGHT_EMA_THRESH)),
            'ema_dx': float(_get_param(ns, 'right_ema_dx_thresh', DEFAULT_RIGHT_EMA_DX_THRESH)),
            'ema_ddx': float(_get_param(ns, 'right_ema_ddx_thresh', DEFAULT_RIGHT_EMA_DDX_THRESH)),
        }

        # Absolute comparison options
        self.use_abs = {
            'x': bool(_get_param(ns, 'use_abs_x', DEFAULT_USE_ABS_X)),
            'dx': bool(_get_param(ns, 'use_abs_dx', DEFAULT_USE_ABS_DX)),
            'ddx': bool(_get_param(ns, 'use_abs_ddx', DEFAULT_USE_ABS_DDX)),
            'ma': bool(_get_param(ns, 'use_abs_ma', DEFAULT_USE_ABS_MA)),
            'ema': bool(_get_param(ns, 'use_abs_ema', DEFAULT_USE_ABS_EMA)),
            'ema_dx': bool(_get_param(ns, 'use_abs_ema_dx', DEFAULT_USE_ABS_EMA_DX)),
            'ema_ddx': bool(_get_param(ns, 'use_abs_ema_ddx', DEFAULT_USE_ABS_EMA_DDX)),
        }

        self.left_state = SideState(self.ma_window, self.ema_window)
        self.right_state = SideState(self.ma_window, self.ema_window)

        self.latest_lines = None

        # Publishers for left
        self.pub_left_x = rospy.Publisher('lane_change/left/x', Float32, queue_size=10)
        self.pub_left_dx = rospy.Publisher('lane_change/left/dx', Float32, queue_size=10)
        self.pub_left_ddx = rospy.Publisher('lane_change/left/ddx', Float32, queue_size=10)
        self.pub_left_ma = rospy.Publisher('lane_change/left/ma', Float32, queue_size=10)
        self.pub_left_ema = rospy.Publisher('lane_change/left/ema', Float32, queue_size=10)
        self.pub_left_ema_dx = rospy.Publisher('lane_change/left/ema_dx', Float32, queue_size=10)
        self.pub_left_ema_ddx = rospy.Publisher('lane_change/left/ema_ddx', Float32, queue_size=10)
        self.pub_left_flag = rospy.Publisher('lane_change/left/flag', Int16, queue_size=10)

        # Publishers for right
        self.pub_right_x = rospy.Publisher('lane_change/right/x', Float32, queue_size=10)
        self.pub_right_dx = rospy.Publisher('lane_change/right/dx', Float32, queue_size=10)
        self.pub_right_ddx = rospy.Publisher('lane_change/right/ddx', Float32, queue_size=10)
        self.pub_right_ma = rospy.Publisher('lane_change/right/ma', Float32, queue_size=10)
        self.pub_right_ema = rospy.Publisher('lane_change/right/ema', Float32, queue_size=10)
        self.pub_right_ema_dx = rospy.Publisher('lane_change/right/ema_dx', Float32, queue_size=10)
        self.pub_right_ema_ddx = rospy.Publisher('lane_change/right/ema_ddx', Float32, queue_size=10)
        self.pub_right_flag = rospy.Publisher('lane_change/right/flag', Int16, queue_size=10)

        self.left_triggered = False
        self.right_triggered = False

        rospy.Subscriber(self.lane_lines_topic, Int32MultiArray, self.lines_callback, queue_size=1)

        self.last_timer_time = None
        period = 1.0 / self.rate_hz if self.rate_hz > 0.0 else 0.05
        rospy.Timer(rospy.Duration(period), self.timer_callback)

    def lines_callback(self, msg):
        if not msg.data or len(msg.data) < 8:
            self.latest_lines = None
            return
        self.latest_lines = list(msg.data[:8])

    @staticmethod
    def _line_valid(line):
        return line and len(line) == 4 and all(v != -1 for v in line)

    @staticmethod
    def _interp_x_at_y(line, y):
        x1, y1, x2, y2 = line
        if y2 == y1:
            return None
        return (float(y) - y1) * (float(x2) - x1) / (float(y2) - y1) + x1

    def _check_thresholds(self, values, thresh_map):
        # values: dict with keys x, dx, ddx, ma, ema
        for k, v in values.items():
            t = thresh_map.get(k, 0.0)
            if t <= 0.0:
                continue
            if v is None or not _finite(v):
                continue
            val = abs(v) if self.use_abs.get(k, False) else v
            if val > t:
                return True
        return False

    def _publish_metrics(self, pubs, values, invalid_value):
        # pubs: list [x, dx, ddx, ma, ema, ema_dx, ema_ddx]
        out = []
        for v in values:
            out.append(v if v is not None and _finite(v) else invalid_value)
        pubs[0].publish(Float32(out[0]))
        pubs[1].publish(Float32(out[1]))
        pubs[2].publish(Float32(out[2]))
        pubs[3].publish(Float32(out[3]))
        pubs[4].publish(Float32(out[4]))
        pubs[5].publish(Float32(out[5]))
        pubs[6].publish(Float32(out[6]))

    def timer_callback(self, event):
        if self.latest_lines is None:
            return

        if self.last_timer_time is None:
            self.last_timer_time = event.current_real
            return

        dt = (event.current_real - self.last_timer_time).to_sec()
        self.last_timer_time = event.current_real
        if dt <= 0.0:
            return

        left_line = self.latest_lines[0:4]
        right_line = self.latest_lines[4:8]

        left_x = None
        right_x = None

        if self._line_valid(left_line):
            left_x = self._interp_x_at_y(left_line, self.y_mid)
        if self._line_valid(right_line):
            right_x = self._interp_x_at_y(right_line, self.y_mid)

        left_metrics = self.left_state.update(left_x, dt) if _finite(left_x) else None
        right_metrics = self.right_state.update(right_x, dt) if _finite(right_x) else None

        # Publish metrics
        self._publish_metrics(
            [
                self.pub_left_x, self.pub_left_dx, self.pub_left_ddx,
                self.pub_left_ma, self.pub_left_ema, self.pub_left_ema_dx, self.pub_left_ema_ddx
            ],
            left_metrics if left_metrics else [None] * 7,
            self.invalid_value
        )
        self._publish_metrics(
            [
                self.pub_right_x, self.pub_right_dx, self.pub_right_ddx,
                self.pub_right_ma, self.pub_right_ema, self.pub_right_ema_dx, self.pub_right_ema_ddx
            ],
            right_metrics if right_metrics else [None] * 7,
            self.invalid_value
        )

        # Threshold check (OR)
        left_values = None
        right_values = None
        if left_metrics:
            left_values = {
                'x': left_metrics[0],
                'dx': left_metrics[1],
                'ddx': left_metrics[2],
                'ma': left_metrics[3],
                'ema': left_metrics[4],
                'ema_dx': left_metrics[5],
                'ema_ddx': left_metrics[6],
            }
        if right_metrics:
            right_values = {
                'x': right_metrics[0],
                'dx': right_metrics[1],
                'ddx': right_metrics[2],
                'ma': right_metrics[3],
                'ema': right_metrics[4],
                'ema_dx': right_metrics[5],
                'ema_ddx': right_metrics[6],
            }

        left_hit = self._check_thresholds(left_values, self.left_thresh) if left_values else False
        right_hit = self._check_thresholds(right_values, self.right_thresh) if right_values else False

        self.pub_left_flag.publish(Int16(1 if left_hit else 0))
        self.pub_right_flag.publish(Int16(1 if right_hit else 0))

        if left_hit and not self.left_triggered:
            rospy.logwarn('lane_change_checker: lane change to left')
        if right_hit and not self.right_triggered:
            rospy.logwarn('lane_change_checker: lane change to right')

        self.left_triggered = left_hit
        self.right_triggered = right_hit


if __name__ == '__main__':
    try:
        LaneChangeChecker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
