# hydro_mpc/navigation/state_machine.py
from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass

class NavState(Enum):
    IDLE = auto()
    TAKEOFF = auto()
    LOITER = auto()
    FOLLOW_TARGET = auto()
    MISSION = auto()
    LANDING = auto()
    EMERGENCY = auto()
    MANUAL = auto()

@dataclass
class NavEvents:
    have_odom: bool
    auto_start: bool
    target_fresh: bool
    trajectory_fresh: bool
    at_takeoff_wp: bool
    at_destination: bool
    landing_needed: bool
    landing_done: bool
    start_requested: bool
    halt_condition: bool
    mission_valid: bool
    manual_requested: bool

class NavStateMachine:
    """Pure transition logic; no ROS, no planning side effects."""
    def __init__(self):
        self.state = NavState.IDLE

    def reset(self, state: NavState = NavState.IDLE):
        self.state = state

    def step(self, ev: NavEvents) -> NavState:

        # Global halt: drop to IDLE from any non-emergency state
        if ev.halt_condition:
            if self.state != NavState.EMERGENCY:
                self.state = NavState.IDLE
            else:
                self.state = NavState.LANDING
            return self.state
    
        if ev.manual_requested:
            self.state = NavState.MANUAL
            return self.state
        
        s = self.state
        if s == NavState.IDLE:
            if ev.have_odom and (ev.auto_start or ev.start_requested):
                self.state = NavState.TAKEOFF

        elif s == NavState.TAKEOFF:
            if ev.at_takeoff_wp:
                if ev.target_fresh:
                    self.state = NavState.FOLLOW_TARGET
                elif ev.mission_valid:
                    self.state = NavState.MISSION
                else:
                    self.state = NavState.LOITER 

        elif s == NavState.LOITER:
            if ev.target_fresh:
                self.state = NavState.FOLLOW_TARGET

        elif s == NavState.MISSION:
            if ev.landing_needed:
                self.state = NavState.LANDING
            elif not ev.mission_valid:
                self.state = NavState.LOITER          # mission was invalidated; fall back
            elif ev.at_destination:
                self.state = NavState.LOITER          # finished the track; loiter

        elif s == NavState.FOLLOW_TARGET:
            if ev.landing_needed:
                self.state = NavState.LANDING
            elif not ev.target_fresh:
                self.state = NavState.LOITER

        elif s == NavState.LANDING:
            if (ev.landing_done or ev.at_destination):
                self.state = NavState.IDLE

        elif s == NavState.MANUAL:
            if ev.have_odom and (ev.auto_start or ev.start_requested):
                self.state = NavState.TAKEOFF

        return self.state
