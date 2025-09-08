# hydro_mpc/navigation/state_machine.py
from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass

class NavState(Enum):
    IDLE = 1
    HOLD = 2
    TAKEOFF = 3
    LOITER = 4
    FOLLOW_TARGET = 5
    MISSION = 6
    LANDING = 7
    EMERGENCY = 8
    MANUAL = 9

@dataclass
class NavEvents:
    have_odom: bool
    grounded: bool
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

        # Global halt: go to HOLD (air) unless EMERGENCY forces landing
        if ev.halt_condition:
            if self.state == NavState.EMERGENCY:
                self.state = NavState.LANDING
            else:
                self.state = NavState.HOLD
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
                    self.state = NavState.HOLD   # reached TO altitude but nothing to do → HOLD

        elif s == NavState.LOITER:
            if ev.target_fresh:
                self.state = NavState.FOLLOW_TARGET

        elif s == NavState.HOLD:
            # HOLD is a “stable hover”. From here we can pick up a mission/target.
            if ev.target_fresh:
                self.state = NavState.FOLLOW_TARGET
            elif ev.mission_valid and ev.start_requested:
                self.state = NavState.MISSION
            # If we end up grounded (e.g., disarm/landed), drop to IDLE.
            # elif ev.grounded:
            #     self.state = NavState.IDLE


        elif s == NavState.MISSION:
            if ev.landing_needed:
                self.state = NavState.LANDING
            elif not ev.mission_valid:
                self.state = NavState.HOLD          # mission invalid → HOLD
            elif ev.at_destination:
                self.state = NavState.HOLD          # finished track → HOLD

        elif s == NavState.FOLLOW_TARGET:
            if ev.landing_needed:
                self.state = NavState.LANDING
            elif not ev.target_fresh:
                self.state = NavState.HOLD

        elif s == NavState.LANDING:
            if (ev.landing_done or ev.at_destination):
                self.state = NavState.IDLE

        elif s == NavState.MANUAL:
            if (not ev.manual_requested):
                if ev.grounded:
                    # self.state = NavState.IDLE 
                    self.state = NavState.HOLD
                else:
                    self.state = NavState.HOLD
            elif ev.have_odom and ev.start_requested:
                self.state = NavState.TAKEOFF


        return self.state
