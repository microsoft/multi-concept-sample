#!/usr/bin/env python3

"""
MSFT Bonsai SDK3 Template for Simulator Integration using Python
Copyright 2020 Microsoft

Usage:
  For registering simulator with the Bonsai service for training:
    python main.py \
           --workspace <workspace_id> \
           --accesskey="<access_key> \
  Then connect your registered simulator to a Brain via UI
  Alternatively, one can set the SIM_ACCESS_KEY and SIM_WORKSPACE as
  environment variables.
"""

import json
import time
from typing import Dict, Any, Optional
from microsoft_bonsai_api.simulator.client import BonsaiClientConfig, BonsaiClient
from microsoft_bonsai_api.simulator.generated.models import (
    SimulatorState,
    SimulatorInterface,
)

import argparse
from sim.moab_model import MoabModel
from jinja2 import Template
from concept_orchestration import ExportedBrainPredictor, launch_predictor_server
from pyrr import matrix33, vector

class TemplateSimulatorSession():
    def __init__(self, render):
        ## Initialize python api for simulator
        self.simulator = MoabModel()
        
        C1_url = 'http://localhost:1111'
        C2_url = 'http://localhost:2222'
        self.C1 = ExportedBrainPredictor(predictor_url=C1_url, control_period=1)
        self.C2 = ExportedBrainPredictor(predictor_url=C2_url, control_period=1)

    def get_state(self) -> Dict[str, Any]:
        """Called to retreive the current state of the simulator. """
        state = self.simulator.state()
        for key, value in state.items():
            state[key] = float(value)

        return state

    def episode_start(self, config: Dict[str, Any]):
        """ Called at the start of each episode """
        
        # Return to pre-determined good state to avoid accidental episode-episode dependencies
        self.simulator.reset()

        if config is None:
            self.sim_config = {}  
        else:
            self.sim_config = config      
        # Initial control state which are all unitless in [-1..1] 
        self.simulator.roll = self.sim_config.get("initial_roll", self.simulator.roll)
        self.simulator.pitch = self.sim_config.get("initial_pitch", self.simulator.pitch)
        self.simulator.height_z = self.sim_config.get("initial_height_z", self.simulator.height_z)

        # Constants, SI units
        self.simulator.time_delta = self.sim_config.get("time_delta", self.simulator.time_delta)
        self.simulator.jitter = self.sim_config.get("jitter", self.simulator.jitter)
        self.simulator.gravity = self.sim_config.get("gravity", self.simulator.gravity)
        self.simulator.plate_theta_vel_limit = self.sim_config.get(
            "plate_theta_vel_limit", self.simulator.plate_theta_vel_limit
        )
        self.simulator.plate_theta_acc = self.sim_config.get(
            "plate_theta_acc", self.simulator.plate_theta_acc
        )
        self.simulator.plate_theta_limit = self.sim_config.get(
            "plate_theta_limit", self.simulator.plate_theta_limit
        )
        self.simulator.plate_z_limit = self.sim_config.get("plate_z_limit", self.simulator.plate_z_limit)

        self.simulator.ball_mass = self.sim_config.get("ball_mass", self.simulator.ball_mass)
        self.simulator.ball_radius = self.sim_config.get("ball_radius", self.simulator.ball_radius)
        self.simulator.ball_shell = self.sim_config.get("ball_shell", self.simulator.ball_shell)

        self.simulator.obstacle_radius = self.sim_config.get(
            "obstacle_radius", self.simulator.obstacle_radius
        )
        self.simulator.obstacle_x = self.sim_config.get("obstacle_x", self.simulator.obstacle_x)
        self.simulator.obstacle_y = self.sim_config.get("obstacle_y", self.simulator.obstacle_y)

        # A target position the AI can try and move the ball to
        self.simulator.target_x = self.sim_config.get("target_x", self.simulator.target_x)
        self.simulator.target_y = self.sim_config.get("target_y", self.simulator.target_y)

        # Observation config
        self.simulator.ball_noise = self.sim_config.get("ball_noise", self.simulator.ball_noise)
        self.simulator.plate_noise = self.sim_config.get("plate_noise", self.simulator.plate_noise)

        # Update the initial plate metrics from the constants and the controls
        self.simulator.update_plate(plate_reset=True)

        # Initial ball state after updating plate
        self.simulator.set_initial_ball(
            self.sim_config.get("initial_x", self.simulator.ball.x),
            self.sim_config.get("initial_y", self.simulator.ball.y),
            self.sim_config.get("initial_z", self.simulator.ball.z),
        )

        # Velocity set as a vector
        self.simulator.ball_vel.x = self.sim_config.get("initial_vel_x", self.simulator.ball_vel.x)
        self.simulator.ball_vel.y = self.sim_config.get("initial_vel_y", self.simulator.ball_vel.y)
        self.simulator.ball_vel.z = self.sim_config.get("initial_vel_z", self.simulator.ball_vel.z)

        # Velocity set as a speed/direction towards target
        initial_speed = self.sim_config.get("initial_speed", None)
        initial_direction = self.sim_config.get("initial_direction", None)
        if initial_speed is not None and initial_direction is not None:
            self._set_velocity_for_speed_and_direction(initial_speed, initial_direction)

    def episode_step(self, action: Dict[str, Any]):
        """ Called for each step of the episode """
        ## Add simulator step api here using action from Bonsai platform
        
        # Use new syntax or fall back to old parameter names
        if action.get('concept_index') == 1: # selector
            action = self.C1.get_action(self.get_state())
        else:
            action = self.C2.get_action(self.get_state())
        
        # Clamp inputs to legal ranges
        self.simulator.roll = self.clamp(
            action.get("input_roll", self.simulator.roll), -1.0, 1.0)
        self.simulator.pitch = self.clamp(
            action.get("input_pitch", self.simulator.pitch), -1.0, 1.0)
        self.simulator.height_z = self.clamp(
            action.get("input_height_z", self.simulator.height_z), -1.0, 1.0)

        self.simulator.step()

    def halted(self) -> bool:
        """
        Should return True if the simulator cannot continue for some reason
        """
        return self.simulator.halted() 

    def clamp(self, val: float, min_val: float, max_val: float):
        """Clamp the values to defined ranges.
        """
        return min(max_val, max(min_val, val))
    
    def _set_velocity_for_speed_and_direction(self, speed: float, direction: float):
        """Set the direction and speed.       
        """
        # Get the heading
        dx = self.simulator.target_x - self.simulator.ball.x
        dy = self.simulator.target_y - self.simulator.ball.y

        # Direction is meaningless if we're already at the target
        if (dx != 0) or (dy != 0):

            # Set the magnitude
            vel = vector.set_length([dx, dy, 0.0], speed)

            # Rotate by direction around Z-axis at ball position
            rot = matrix33.create_from_axis_rotation([0.0, 0.0, 1.0], direction)
            vel = matrix33.apply_to_vector(rot, vel)

            # Unpack into ball velocity
            self.simulator.ball_vel.x = vel[0]
            self.simulator.ball_vel.y = vel[1]
            self.simulator.ball_vel.z = vel[2]


def main(render=False):
    # Grab standardized way to interact with sim API
    sim = TemplateSimulatorSession(render= render)

    # Configure client to interact with Bonsai service
    config_client = BonsaiClientConfig()
    client = BonsaiClient(config_client)

    # Load json file as simulator integration config type file
    with open('moab_interface.json', "r") as file:
        template_str = file.read()

    # render the template with our constants
    template = Template(template_str)
    interface_str = template.render(
        initial_pitch=sim.simulator.pitch,
        initial_roll=sim.simulator.roll,
        initial_height_z=sim.simulator.height_z,
        time_delta=sim.simulator.time_delta,
        gravity=sim.simulator.time_delta,
        plate_radius=sim.simulator.plate_radius,
        plate_theta_vel_limit=sim.simulator.plate_theta_vel_limit,
        plate_theta_acc=sim.simulator.plate_theta_acc,
        plate_theta_limit=sim.simulator.plate_theta_limit,
        plate_z_limit=sim.simulator.plate_z_limit,
        ball_mass=sim.simulator.ball_mass,
        ball_radius=sim.simulator.ball_radius,
        ball_shell=sim.simulator.ball_shell,
        obstacle_radius=sim.simulator.obstacle_radius,
        obstacle_x=sim.simulator.obstacle_x,
        obstacle_y=sim.simulator.obstacle_y,
        target_x=sim.simulator.target_x,
        target_y=sim.simulator.target_y,
        initial_x=sim.simulator.ball.x,
        initial_y=sim.simulator.ball.y,
        initial_z=sim.simulator.ball.z,
        initial_vel_x=sim.simulator.ball_vel.x,
        initial_vel_y=sim.simulator.ball_vel.y,
        initial_vel_z=sim.simulator.ball_vel.z,
        initial_speed=0,
        initial_direction=0,
        ball_noise=sim.simulator.ball_noise,
        plate_noise=sim.simulator.plate_noise,
    )
    interface = json.loads(interface_str)
    
    # Create simulator session and init sequence id
    
    registration_info =  SimulatorInterface(
        name=interface["name"],
        timeout=interface["timeout"],
        simulator_context=config_client.simulator_context,
        description=interface["description"],
    )

    registered_session = client.session.create(
                            workspace_name=config_client.workspace, 
                            body=registration_info
    )
    print("Registered simulator.")
    sequence_id = 1

    try:
        while True:
            # Advance by the new state depending on the event type
            sim_state = SimulatorState(
                            sequence_id=sequence_id, state=sim.get_state(), 
                            halted=sim.halted()
            )
            event = client.session.advance(
                        workspace_name=config_client.workspace, 
                        session_id=registered_session.session_id, 
                        body=sim_state
            )
            sequence_id = event.sequence_id
            print("[{}] Last Event: {}".format(time.strftime('%H:%M:%S'), 
                                               event.type))

            # Event loop
            if event.type == 'Idle':
                time.sleep(event.idle.callback_time)
                print('Idling...')
            elif event.type == 'EpisodeStart':
                sim.episode_start(event.episode_start.config)
            elif event.type == 'EpisodeStep':
                sim.episode_step(event.episode_step.action)
            elif event.type == 'EpisodeFinish':
                print('Episode Finishing...')
            elif event.type == 'Unregister':
                client.session.delete(
                    workspace_name=config_client.workspace, 
                    session_id=registered_session.session_id
                )
                print("Unregistered simulator.")
            else:
                pass
    except KeyboardInterrupt:
        # Gracefully unregister with keyboard interrupt
        client.session.delete(
            workspace_name=config_client.workspace, 
            session_id=registered_session.session_id
        )
        print("Unregistered simulator.")
    except Exception as err:
        # Gracefully unregister for any other exceptions
        client.session.delete(
            workspace_name=config_client.workspace, 
            session_id=registered_session.session_id
        )
        print("Unregistered simulator because: {}".format(err))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='args for sim integration',
                                     allow_abbrev=False)
    parser.add_argument('--render', action='store_true')
    args, _ = parser.parse_known_args()
    main(render=args.render)