# hydro_mpc/navigation/state_machine.py
from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass

class NavState(Enum):
    IDLE = auto()
    TAKEOFF = auto()
    LOITER = auto()
    FOLLOW_TARGET = auto()
    LANDING = auto()

@dataclass
class NavEvents:
    have_odom: bool
    auto_start: bool
    target_fresh: bool
    at_takeoff_wp: bool
    landing_needed: bool
    landing_done: bool

class NavStateMachine:
    """Pure transition logic; no ROS, no planning side effects."""
    def __init__(self):
        self.state = NavState.IDLE

    def reset(self, state: NavState = NavState.IDLE):
        self.state = state

    def step(self, ev: NavEvents) -> NavState:
        s = self.state
        if s == NavState.IDLE:
            if ev.have_odom and ev.auto_start:
                self.state = NavState.TAKEOFF

        elif s == NavState.TAKEOFF:
            if ev.at_takeoff_wp:
                self.state = NavState.FOLLOW_TARGET if ev.target_fresh else NavState.LOITER

        elif s == NavState.LOITER:
            if ev.target_fresh:
                self.state = NavState.FOLLOW_TARGET

        elif s == NavState.FOLLOW_TARGET:
            if ev.landing_needed:
                self.state = NavState.LANDING
            elif not ev.target_fresh:
                self.state = NavState.LOITER

        elif s == NavState.LANDING:
            if ev.landing_done:
                self.state = NavState.IDLE

        return self.state
